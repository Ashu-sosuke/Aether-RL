from google import genai
from google.genai import types
from config import settings
from models import TaskPlan
import json
import logging
import asyncio
import httpx

logger = logging.getLogger("AetherIntent")

RETRYABLE_ERROR_MARKERS = ("429", "RESOURCE_EXHAUSTED", "500", "502", "503", "504", "UNAVAILABLE")
MODEL_NOT_FOUND_MARKERS = ("404", "NOT_FOUND")
ACCESS_DENIED_MARKERS = ("403", "PERMISSION_DENIED", "DENIED ACCESS")


class GenerationError(RuntimeError):
    """Raised when every configured model fails."""


class IntentParser:
    def __init__(self):
        self.gemini_client = genai.Client(api_key=settings.gemini_api_key)
        self.gemini_models = settings.gemini_models

    async def parse_goal(self, goal: str, task_id: str) -> TaskPlan:
        prompt = f"""
        Analyze the user goal: "{goal}"
        Break it down into a list of high-level logical steps for an Android agent.
        Return ONLY a JSON object:
        {{
            "task_id": "{task_id}",
            "goal": "{goal}",
            "steps": ["step 1", "step 2"],
            "status": "pending",
            "context": {{}}
        }}
        """
        try:
            data = await self._generate_json_with_fallback(prompt)
            data["task_id"] = task_id
            data["goal"] = goal
            return TaskPlan(**data)
        except Exception as e:
            logger.error(f"IntentParser: Failed to parse goal: {e}")
            raise

    async def get_next_action(self, 
                             goal: str, 
                             steps: list, 
                             nodes: list, 
                             active_app: str) -> dict:
        node_context = "\n".join([n.to_text_repr() for n in nodes[:50]])
        
        prompt = f"""
        Task: {goal}
        Remaining Steps: {steps}
        Active App: {active_app}
        UI Tree:
        {node_context}

        Based on the UI tree, determine the single next best action.
        Return ONLY a JSON object:
        {{
            "thought": "description",
            "action": {{
                "type": "tap",
                "node_id": "id",
                "text": null
            }}
        }}
        """
        return await self._generate_json_with_fallback(prompt)

    async def _generate_json_with_fallback(self, prompt: str) -> dict:
        # 1. Try Nemotron (Primary)
        if settings.nvidia_api_key:
            try:
                logger.info("Attempting generation with Nemotron (Primary)...")
                return await self._generate_json_with_nemotron(prompt)
            except Exception as e:
                logger.warning(f"Nemotron failed: {e}. Falling back to Gemini...")

        # 2. Try Gemini (Secondary)
        last_err = None
        for model_id in self.gemini_models:
            try:
                response = await asyncio.to_thread(
                    self.gemini_client.models.generate_content,
                    model=model_id,
                    contents=prompt,
                    config=types.GenerateContentConfig(response_mime_type="application/json"),
                )
                return self._parse_json_response(response.text)
            except Exception as e:
                last_err = e
                err_str = str(e)
                if self._contains_any(err_str, MODEL_NOT_FOUND_MARKERS):
                    logger.warning("Gemini model %s is not available.", model_id)
                    continue
                if self._contains_any(err_str, ACCESS_DENIED_MARKERS):
                    logger.error("Gemini access denied for model %s", model_id)
                    continue # Try next gemini model or raise at end
                if self._contains_any(err_str, RETRYABLE_ERROR_MARKERS):
                    logger.warning("Gemini model %s rate-limited. Trying next...", model_id)
                    await asyncio.sleep(1)
                    continue
                logger.error(f"Gemini error with {model_id}: {e}")

        # 3. Try Ollama (Local Backup)
        if settings.ollama_base_url:
            try:
                logger.info("Attempting generation with local Ollama...")
                return await self._generate_json_with_ollama(prompt)
            except Exception as e:
                logger.error(f"Ollama also failed: {e}")

        raise GenerationError(f"All LLM providers failed. Last error: {last_err}")

    async def _generate_json_with_nemotron(self, prompt: str) -> dict:
        url = "https://integrate.api.nvidia.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {settings.nvidia_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": settings.nvidia_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "top_p": 0.7,
            "max_tokens": 1024,
            "stream": False
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            return self._parse_json_response(content)

    async def _generate_json_with_ollama(self, prompt: str) -> dict:
        url = f"{settings.ollama_base_url}/api/generate"
        payload = {
            "model": "gemma:2b", # Defaulting to a small fast model
            "prompt": prompt,
            "stream": False,
            "format": "json"
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            result = response.json()
            return json.loads(result["response"])

    def _parse_json_response(self, raw: str) -> dict:
        text = raw.strip()
        # Handle markdown blocks if present
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        return json.loads(text)

    @staticmethod
    def _contains_any(value: str, markers: tuple[str, ...]) -> bool:
        upper_value = value.upper()
        return any(marker in upper_value for marker in markers)

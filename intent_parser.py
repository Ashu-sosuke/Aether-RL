from google import genai
from google.genai import types
from config import settings
from models import TaskPlan
import json
import logging
import asyncio

logger = logging.getLogger("AetherIntent")

RETRYABLE_ERROR_MARKERS = ("429", "RESOURCE_EXHAUSTED", "500", "502", "503", "504", "UNAVAILABLE")
MODEL_NOT_FOUND_MARKERS = ("404", "NOT_FOUND")


class GeminiGenerationError(RuntimeError):
    """Raised when every configured Gemini model fails for a user-facing reason."""


class IntentParser:
    def __init__(self):
        self.client = genai.Client(api_key=settings.gemini_api_key)
        self.models_to_try = settings.gemini_models

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
            return await self._generate_with_fallback(prompt, task_id)
        except Exception as e:
            logger.error(f"IntentParser: Failed to parse goal: {e}")
            raise

    async def _generate_with_fallback(self, prompt: str, task_id: str) -> TaskPlan:
        data = await self._generate_json_with_fallback(prompt)
        data["task_id"] = task_id
        return TaskPlan(**data)

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
        last_err = None
        saw_retryable_error = False
        saw_model_not_found = False

        for model_id in self.models_to_try:
            try:
                response = await asyncio.to_thread(
                    self.client.models.generate_content,
                    model=model_id,
                    contents=prompt,
                    config=types.GenerateContentConfig(response_mime_type="application/json"),
                )
                return self._parse_json_response(response.text)
            except Exception as e:
                last_err = e
                err_str = str(e)
                if self._contains_any(err_str, MODEL_NOT_FOUND_MARKERS):
                    saw_model_not_found = True
                    logger.warning("Gemini model %s is not available for generateContent.", model_id)
                    continue
                if self._contains_any(err_str, RETRYABLE_ERROR_MARKERS):
                    saw_retryable_error = True
                    logger.warning("Gemini model %s is temporarily unavailable or rate-limited: %s", model_id, err_str[:160])
                    await asyncio.sleep(1)
                    continue
                raise

        configured = ", ".join(self.models_to_try)
        if saw_retryable_error:
            raise GeminiGenerationError(
                "Gemini API quota is exhausted or temporarily rate-limited. "
                f"Configured models tried: {configured}. Check Google AI Studio quota/billing, "
                "or set GEMINI_MODELS to other models that support generateContent."
            ) from last_err
        if saw_model_not_found:
            raise GeminiGenerationError(
                "No configured Gemini model is available for generateContent. "
                f"Configured models tried: {configured}. Update GEMINI_MODELS with model names "
                "returned by the Gemini models.list endpoint."
            ) from last_err
        raise GeminiGenerationError(f"Gemini generation failed. Configured models tried: {configured}.") from last_err

    def _parse_json_response(self, raw: str) -> dict:
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        return json.loads(text)

    def _parse_response(self, raw: str, goal: str) -> TaskPlan:
        data = self._parse_json_response(raw)
        data.setdefault("task_id", "")
        data.setdefault("goal", goal)
        return TaskPlan(**data)

    @staticmethod
    def _contains_any(value: str, markers: tuple[str, ...]) -> bool:
        upper_value = value.upper()
        return any(marker in upper_value for marker in markers)

from google import genai
from google.genai import types
from config import settings
from models import TaskPlan
from typing import Optional
import json
import logging
import asyncio
import openai

logger = logging.getLogger("AetherIntent")

RETRYABLE_ERROR_MARKERS = ("429", "RESOURCE_EXHAUSTED", "500", "502", "503", "504", "UNAVAILABLE")
MODEL_NOT_FOUND_MARKERS = ("404", "NOT_FOUND")


class GeminiGenerationError(RuntimeError):
    """Raised when every configured Gemini model fails for a user-facing reason."""


import base64

class IntentParser:
    def __init__(self):
        self.gemini_client = genai.Client(api_key=settings.gemini_api_key)
        self.groq_client = None
        if settings.groq_api_key and settings.groq_api_key != "gsk_placeholder_replace_me":
            self.groq_client = openai.AsyncOpenAI(
                api_key=settings.groq_api_key,
                base_url=settings.groq_base_url
            )
        self.gemini_models = settings.gemini_models
        self.primary_model = settings.primary_model

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
                             active_app: str,
                             history: list = [],
                             screenshot_description: Optional[str] = None,
                             screenshot: Optional[str] = None) -> dict:
        node_context = "\n".join([n.to_text_repr() for n in nodes[:50]])
        
        prompt = f"""
### ROLE
You are the "Aether Brain," the central intelligence of an autonomous Android agent. Your goal is to complete complex user tasks by interacting with the Android UI via an Accessibility Service.

### INPUT DATA
1. USER_GOAL: {goal}
2. UI_TREE:
{node_context}
3. SCREENSHOT_DESCRIPTION: {screenshot_description or "N/A"}
4. ACTION_HISTORY: {history}

### OPERATIONAL RULES
1. REASONING: Before acting, analyze the screen. Does the current screen match the task goal? 
2. STEP-BY-STEP: Only perform ONE action per turn. 
3. HIERARCHY: 
   - Prefer Direct Intent (e.g., Opening an app directly).
   - Use Resource IDs if available.
   - Use Text labels if IDs are missing.
   - Use Coordinates (X, Y) if the UI tree is empty/null.
4. SELF-HEALING: If an action failed in the history, try a different approach (e.g., scroll down or go back).

### ACTION SCHEMA
You must respond ONLY in the following JSON format:
{{
  "thought": "Brief explanation of what you see and why you are taking this step.",
  "action": "CLICK | TYPE | SCROLL_DOWN | SCROLL_UP | BACK | OPEN_APP",
  "params": {{
    "target_id": "string (resource-id)",
    "text": "string (for TYPE or APP_NAME)",
    "coords": {{"x": int, "y": int}}
  }},
  "is_complete": boolean,
  "status_message": "User-friendly update"
}}

### TASK GUIDELINES
- YouTube: Launch app -> Search -> Type -> Click video.
- Swiggy/Zomato: Launch app -> Search Hotel -> Select Item -> Add to Cart -> Stop at Payment.
- Gmail: Open -> Click 'Compose' -> Fill fields -> Send.
        """
        return await self._generate_json_with_fallback(prompt, screenshot)

    async def _generate_json_with_fallback(self, prompt: str, screenshot: Optional[str] = None) -> dict:
        # 1. Try Groq (Primary) - Only if no screenshot (Groq is text-only usually)
        if self.groq_client and not screenshot:
            try:
                logger.info(f"Consulting Groq ({self.primary_model})...")
                response = await self.groq_client.chat.completions.create(
                    model=self.primary_model,
                    messages=[{"role": "user", "content": prompt}],
                    # format not supported on some groq models, but typically we want it
                )
                return self._parse_json_response(response.choices[0].message.content)
            except Exception as e:
                logger.warning(f"Groq API failed: {e}. Falling back to Gemini.")

        # 2. Try Gemini Fallbacks
        last_err = None
        saw_retryable_error = False
        saw_model_not_found = False

        # Prepare Gemini content (multimodal if screenshot exists)
        contents = [prompt]
        if screenshot:
            try:
                contents.append(types.Part.from_bytes(
                    data=base64.b64decode(screenshot),
                    mime_type="image/jpeg"
                ))
            except Exception as e:
                logger.error(f"Failed to decode screenshot: {e}")

        for model_id in self.gemini_models:
            try:
                logger.info(f"Consulting Gemini ({model_id}) - Vision: {screenshot is not None}...")
                response = await asyncio.to_thread(
                    self.gemini_client.models.generate_content,
                    model=model_id,
                    contents=contents,
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

        configured = ", ".join(self.gemini_models)
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

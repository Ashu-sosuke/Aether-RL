from google import genai
from google.genai import types
from config import settings
from models import ActionCommand, TaskPlan
import json
import logging

logger = logging.getLogger("AetherIntent")

class IntentParser:
    def __init__(self):
        self.client = genai.Client(api_key=settings.gemini_api_key)
        self.model_id = "gemini-2.0-flash"

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
        import asyncio
        # Try latest aliases which often resolve 404s in v1beta
        models_to_try = [
            "gemini-2.0-flash", 
            "gemini-1.5-flash", 
            "gemini-1.5-flash-latest", 
            "gemini-1.5-pro-latest"
        ]
        last_err = None

        for model_id in models_to_try:
            try:
                response = self.client.models.generate_content(
                    model=model_id,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json"
                    )
                )
                data = json.loads(response.text)
                data["task_id"] = task_id
                return TaskPlan(**data)
            except Exception as e:
                last_err = e
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "404" in err_str:
                    logger.warning(f"Model {model_id} failed ({err_str[:50]}). Trying next in 1s...")
                    await asyncio.sleep(1) # Small delay to respect rate limits
                    continue
                raise e
        raise last_err

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
        models_to_try = [
            "gemini-2.0-flash", 
            "gemini-1.5-flash", 
            "gemini-1.5-flash-latest", 
            "gemini-1.5-pro-latest"
        ]
        for model_id in models_to_try:
            try:
                response = self.client.models.generate_content(
                    model=model_id,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json"
                    )
                )
                return json.loads(response.text)
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "404" in err_str:
                    logger.warning(f"Model {model_id} failed. Trying next in 1s...")
                    await asyncio.sleep(1)
                    continue
                raise e
        raise last_err

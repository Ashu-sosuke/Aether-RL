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
        models_to_try = ["gemini-2.0-flash", "gemini-1.5-flash-002", "gemini-1.5-flash", "gemini-1.5-pro"]
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
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e) or "404" in str(e):
                    logger.warning(f"Model {model_id} failed ({str(e)[:50]}). Trying next...")
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
        models_to_try = ["gemini-2.0-flash", "gemini-1.5-flash-002", "gemini-1.5-flash", "gemini-1.5-pro"]
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
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e) or "404" in str(e):
                    logger.warning(f"Model {model_id} failed. Trying next...")
                    continue
                raise e
        raise last_err

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
        response = self.client.models.generate_content(
            model=self.model_id,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )
        data = json.loads(response.text)
        # Ensure task_id matches what we generated
        data["task_id"] = task_id
        return TaskPlan(**data)

    async def get_next_action(self, 
                             goal: str, 
                             steps: list, 
                             nodes: list, 
                             active_app: str) -> dict:
        node_context = "\n".join([n.to_text_repr() for n in nodes[:50]]) # Limit to 50 nodes for context
        
        prompt = f"""
        Task: {goal}
        Remaining Steps: {steps}
        Active App: {active_app}
        UI Tree:
        {node_context}

        Based on the UI tree, determine the single next best action.
        Return ONLY a JSON object:
        {{
            "thought": "I need to click the search bar to find the video",
            "action": {{
                "type": "tap",
                "node_id": "com.google.android.youtube:id/search_edit_text",
                "text": null
            }}
        }}
        """
        response = self.client.models.generate_content(
            model=self.model_id,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )
        return json.loads(response.text)

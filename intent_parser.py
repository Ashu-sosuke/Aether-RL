import google.generativeai as genai
import openai
import json
from config import settings
from models import TaskPlan, PlannedStep

class IntentParser:
    def __init__(self):
        genai.configure(api_key=settings.gemini_api_key)
        self.model = genai.GenerativeModel("gemini-1.5-pro")
        self._fallback = openai.AsyncOpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None

    async def parse_goal(self, goal: str, memory_context: dict) -> TaskPlan:
        prompt = self._build_prompt(goal, memory_context)

        # Try Gemini first
        try:
            response = self.model.generate_content(prompt)
            raw = response.text
            return self._parse_response(raw, goal)
        except Exception as e:
            print(f"Gemini failed: {e}")
            if not self._fallback:
                raise RuntimeError("Gemini failed and no OpenAI API key provided for fallback.")

        # Fallback to GPT-4o
        try:
            resp = await self._fallback.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            raw = resp.choices[0].message.content
            return self._parse_response(raw, goal)
        except Exception as e:
            raise RuntimeError(f"Both LLMs failed: {e}")

    def _build_prompt(self, goal: str, memory: dict) -> str:
        memory_str = json.dumps(memory, indent=2) if memory else "{}"
        return f"""
You are an Android automation planner. Break down the user's goal into
a sequence of atomic UI steps that an agent can execute.

USER GOAL: {goal}

USER MEMORY (preferences and history):
{memory_str}

Return ONLY a valid JSON object in this exact format, no other text:
{{
  "steps": [
    {{
      "description"  : "Tap the search bar",
      "appPackage"   : "com.swiggy.android",
      "requiresHitl" : false,
      "actionType"   : "TAP",
      "inputText"    : null
    }}
  ],
  "context": {{
    "extracted_entities": {{}}
  }}
}}

Rules:
- Each step is a single atomic UI action (one tap, one type, one scroll)
- requiresHitl must be true for: payments, sending messages, deleting
- actionType must be one of: TAP, LONG_TAP, TYPE, SCROLL_UP, SCROLL_DOWN, BACK, HOME
- inputText is only set when actionType is TYPE
- Use memory context to resolve "my usual", contact names, addresses
- Maximum 20 steps per plan
- appPackage is the Android package name if known, otherwise null
"""

    def _parse_response(self, raw: str, goal: str) -> TaskPlan:
        clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```")
        data  = json.loads(clean)
        steps = [PlannedStep(**s) for s in data.get("steps", [])]
        return TaskPlan(
            goal    = goal,
            steps   = steps,
            context = data.get("context", {})
        )

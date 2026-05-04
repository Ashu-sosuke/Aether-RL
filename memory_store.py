from supabase import create_client, Client
from config import settings
from models import TaskPlan
import json
import traceback

class MemoryStore:
    def __init__(self):
        self.client: Client = create_client(
            settings.supabase_url,
            settings.supabase_key
        )

    async def get_context(self, user_id: str) -> dict:
        try:
            result = self.client.table("user_memory") \
                .select("key, value") \
                .eq("user_id", user_id) \
                .execute()
            return {row["key"]: row["value"] for row in (result.data or [])}
        except Exception as e:
            print(f"MemoryStore: Error fetching context for {user_id}: {e}")
            return {}

    async def set_memory(self, user_id: str, key: str, value: str):
        try:
            self.client.table("user_memory").upsert({
                "user_id"   : user_id,
                "key"       : key,
                "value"     : value,
                "updated_at": "now()"
            }).execute()
        except Exception as e:
            print(f"MemoryStore: Error setting memory: {e}")

    async def log_task(self, task: TaskPlan, status: str, user_id: str):
        try:
            self.client.table("task_history").insert({
                "task_id"   : str(task.task_id), # Updated from taskId
                "user_id"   : user_id,
                "goal"      : task.goal,
                "status"    : status,
                "created_at": "now()"
            }).execute()
        except Exception as e:
            print(f"MemoryStore: Error logging task: {e}")

    async def extract_and_save(self, task: TaskPlan, user_id: str, outcome: str):
        try:
            ctx = task.context
            # Save last used app
            if "app_package" in ctx:
                await self.set_memory(user_id, "last_used_app", ctx["app_package"])
            # Save last order if food delivery
            if "order_items" in ctx:
                await self.set_memory(user_id, "last_order",
                                       json.dumps(ctx["order_items"]))
            # Save contact UPI if payment
            if "contact_name" in ctx and "upi_id" in ctx:
                await self.set_memory(
                    user_id,
                    f"contact_{ctx['contact_name'].lower()}_upi",
                    ctx["upi_id"]
                )
        except Exception as e:
            print(f"MemoryStore: Error in extract_and_save: {e}")

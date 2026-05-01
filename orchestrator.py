import asyncio
import json
import uuid
from fastapi import WebSocket
from sqlalchemy.ext.asyncio import AsyncSession
from models import TaskPlan, PlannedStep, NodeData, ActionCommand, ObservationPayload
from semantic_mapper import SemanticMapper
from memory_store import MemoryStore
from token_bucket import TokenBucket, ACTION_COST
from safety import classify_step_risk, build_hitl_description
from db import ActionLogEntry

# Module-level ack store shared across coroutines
_ack_store: dict[str, str] = {}

class TaskOrchestrator:
    def __init__(self,
                 ws         : WebSocket,
                 task       : TaskPlan,
                 session_id : str,
                 user_id    : str,
                 mapper     : SemanticMapper,
                 memory     : MemoryStore,
                 bucket     : TokenBucket,
                 db         : AsyncSession):
        self.ws         = ws
        self.task       = task
        self.session_id = session_id
        self.user_id    = user_id
        self.mapper     = mapper
        self.memory     = memory
        self.bucket     = bucket
        self.db         = db
        self._obs_queue : asyncio.Queue[ObservationPayload] = asyncio.Queue(maxsize=1)
        self._hitl_queue: asyncio.Queue[bool]               = asyncio.Queue(maxsize=1)

    async def run(self) -> str:
        """
        Execute all steps in the task plan.
        Returns final status: "completed" | "failed" | "aborted"
        """
        outcome = "completed"
        for step in self.task.steps:
            step_result = await self._execute_step(step)
            if step_result == "aborted":
                outcome = "aborted"
                break
            if step_result == "failed":
                outcome = "failed"
                break

        await self._send({
            "type"   : "task_complete" if outcome == "completed" else "task_failed",
            "taskId" : self.task.taskId,
            "payload": {"status": outcome, "message": ""}
        })
        await self.memory.log_task(self.task, outcome, self.user_id)
        if outcome == "completed":
            await self.memory.extract_and_save(
                self.task, self.user_id, outcome)
        return outcome

    async def _execute_step(self, step: PlannedStep) -> str:
        # -- STEP 1: OBSERVE --
        await self._send({
            "type"   : "status",
            "taskId" : self.task.taskId,
            "payload": {"status": "observing",
                        "message": step.description}
        })
        obs = await self._wait_for_observation(timeout=10.0)
        if obs is None:
            return "failed"

        # -- STEP 2: ANALYSE --
        node = self.mapper.find_best_node(step.description, obs.nodes)
        if node is None:
            await self._log_action(step, None, obs.activePackage, "no_match")
            return "failed"

        # -- STEP 3: SAFETY CHECK --
        is_risky = classify_step_risk(step, obs.activePackage) or step.requiresHitl
        approved = True
        if is_risky:
            if not self.bucket.consume(ACTION_COST):
                await self._send_token_error()
                return "failed"
            approved = await self._request_hitl(step, node)
            if not approved:
                await self._log_action(step, node, obs.activePackage, "hitl_denied")
                return "aborted"

        # -- STEP 4: EXECUTE --
        if not self.bucket.consume(ACTION_COST):
            await self._send_token_error()
            return "failed"

        command = ActionCommand(
            nodeId = node.nodeId,
            type   = step.actionType,
            text   = step.inputText,
            x      = node.center_x(),
            y      = node.center_y()
        )
        await self._send({
            "type"   : "command",
            "taskId" : self.task.taskId,
            "payload": {"action": command.model_dump()}
        })

        # Wait for ack
        ack_status = await self._wait_for_ack(command.actionId, timeout=15.0)
        status = "success" if ack_status == "success" else "failed"

        # -- STEP 5: LOG --
        await self._log_action(step, node, obs.activePackage, status,
                                is_risky, approved if is_risky else None)

        # Retry once on failure
        if status == "failed":
            await asyncio.sleep(1.0)
            # return await self._execute_step(step) # simplified retry for demo
            pass 

        return "success" if status == "success" else "failed"

    async def _wait_for_observation(self, timeout: float) -> ObservationPayload | None:
        try:
            return await asyncio.wait_for(self._obs_queue.get(), timeout)
        except asyncio.TimeoutError:
            return None

    async def _wait_for_ack(self, action_id: str, timeout: float) -> str:
        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            ack = _ack_store.get(action_id)
            if ack:
                del _ack_store[action_id]
                return ack
            await asyncio.sleep(0.1)
        return "timeout"

    async def _request_hitl(self, step: PlannedStep, node: NodeData) -> bool:
        desc = build_hitl_description(step, node)
        await self._send({
            "type"   : "hitl_required",
            "taskId" : self.task.taskId,
            "payload": {"description": desc}
        })
        try:
            return await asyncio.wait_for(self._hitl_queue.get(), timeout=60.0)
        except asyncio.TimeoutError:
            return False   # auto-deny on timeout

    async def _send(self, data: dict):
        await self.ws.send_text(json.dumps(data))

    async def _send_token_error(self):
        reset_in = self.bucket.reset_at_seconds()
        await self._send({
            "type"   : "token_exhausted",
            "taskId" : self.task.taskId,
            "payload": {
                "status" : "error",
                "message": f"Token limit reached. Resets in {reset_in:.0f}s"
            }
        })

    async def _log_action(self, step, node, package, status,
                           hitl_req=False, hitl_approved=None):
        entry = ActionLogEntry(
            task_id       = uuid.UUID(self.task.taskId),
            step_id       = step.stepId,
            action_type   = step.actionType.value,
            node_id       = node.nodeId if node else None,
            app_package   = package,
            status        = status,
            hitl_required = hitl_req,
            hitl_approved = hitl_approved
        )
        self.db.add(entry)
        await self.db.commit()

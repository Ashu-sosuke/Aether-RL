import asyncio
import logging
from typing import Dict, Optional
from models import TaskPlan, NodeData, ActionCommand, OutboundMessage, CommandPayload, StatusPayload
from intent_parser import IntentParser
from semantic_mapper import SemanticMapper
from memory_store import MemoryStore
from config import settings

logger = logging.getLogger("AetherOrchestrator")

class Orchestrator:
    def __init__(self):
        self.intent_parser = IntentParser()
        self.semantic_mapper = SemanticMapper()
        self.memory = MemoryStore()
        
        # Track active asyncio Tasks for cancellation
        self._running_tasks: Dict[str, asyncio.Task] = {}
        
        # Coroutine-safe event store for ACKs
        self._ack_events: Dict[str, asyncio.Event] = {}
        self._last_ack_status: Dict[str, str] = {}
        
        # Current observation for each task
        self._current_nodes: Dict[str, list[NodeData]] = {}
        self._current_app: Dict[str, str] = {}
        self._current_screenshot: Dict[str, Optional[str]] = {}
        self._observation_events: Dict[str, asyncio.Event] = {}
        self._blind_events: Dict[str, str] = {}
        self._retry_counts: Dict[str, int] = {}

    async def run_task(self, task: TaskPlan, websocket, user_id: str):
        task_id = task.task_id
        self._running_tasks[task_id] = asyncio.current_task()
        logger.info(f"Starting task loop: {task_id} - Goal: {task.goal}")
        
        try:
            while task.status not in ["completed", "failed"]:
                # Create a fresh event for this observation wait (Bug 5)
                obs_event = asyncio.Event()
                self._observation_events[task_id] = obs_event
                if self._current_nodes.get(task_id) or self._current_screenshot.get(task_id):
                    obs_event.set()
                
                try:
                    logger.info(f"Task {task_id}: Waiting for observation...")
                    await asyncio.wait_for(
                        obs_event.wait(), 
                        timeout=settings.observation_timeout_seconds
                    )
                except asyncio.TimeoutError:
                    logger.error(f"Task {task_id}: Observation timeout")
                    await self._send_failed(websocket, task_id, 
                        "Observation timeout — is the accessibility service running?")
                    break

                # 2. Check for blind state
                if task_id in self._blind_events:
                    reason = self._blind_events.get(task_id, "unknown")
                    logger.warning(f"Task {task_id}: Agent is blind: {reason}")
                    self._retry_counts[task_id] = self._retry_counts.get(task_id, 0) + 1
                    wait = min(2 ** self._retry_counts[task_id], 30)
                    await asyncio.sleep(wait)
                    
                    # Try navigating home before re-attempting
                    logger.info(f"Task {task_id}: Attempting global HOME to recover from blind state")
                    cmd = ActionCommand(type="home")
                    await websocket.send_json(OutboundMessage(
                        type="command",
                        task_id=task_id,
                        payload=CommandPayload(action=cmd, thought=f"Device observation blocked ({reason}). Navigating Home to recover.")
                    ).model_dump(by_alias=True))
                    
                    # Clear blind state to wait for next observation
                    self._blind_events.pop(task_id, None)
                    continue
                
                self._retry_counts[task_id] = 0

                # 3. Get LLM recommendation
                nodes = self._current_nodes.get(task_id, [])
                active_app = self._current_app.get(task_id, "unknown")
                screenshot = self._current_screenshot.get(task_id)
                
                # Retrieve history from memory store (placeholder for now)
                history = [] 
                
                logger.info(f"Task {task_id}: Consulting Aether Brain...")
                decision_data = await self.intent_parser.get_next_action(
                    task.goal, task.steps, nodes, active_app, 
                    history=history, screenshot=screenshot
                )
                
                from models import BrainDecision
                decision = BrainDecision(**decision_data)
                
                # Check for completion
                if decision.is_complete or decision.action == "COMPLETE":
                    task.status = "completed"
                    await self.memory.log_completed_task(task, "completed", user_id)
                    await websocket.send_json(OutboundMessage(
                        type="task_complete",
                        task_id=task_id,
                        payload=StatusPayload(message=decision.status_message, status="completed")
                    ).model_dump(by_alias=True))
                    break

                # Translate Brain action to ActionCommand type
                action_map = {
                    "CLICK": "tap",
                    "TYPE": "type",
                    "SCROLL_DOWN": "scroll_down",
                    "SCROLL_UP": "scroll_up",
                    "BACK": "back",
                    "OPEN_APP": "open_app"
                }
                
                action_type = action_map.get(decision.action, "tap")
                params = decision.params or {}
                
                cmd = ActionCommand(
                    type=action_type,
                    node_id=params.get("target_id"),
                    text=params.get("text"),
                    x=params.get("coords", {}).get("x") if params.get("coords") else None,
                    y=params.get("coords", {}).get("y") if params.get("coords") else None
                )

                # 4. Map semantic node_id to physical nodeId if needed
                if cmd.node_id and not cmd.node_id.startswith("physical_"):
                    best_node = self.semantic_mapper.find_best_node(cmd.node_id, nodes)
                    if best_node:
                        logger.info(f"Mapped semantic ID '{cmd.node_id}' -> '{best_node.node_id}'")
                        cmd.node_id = best_node.node_id

                # 5. Send command to Android
                logger.info(f"Task {task_id}: Sending {cmd.type} - {decision.status_message}")
                await websocket.send_json(OutboundMessage(
                    type="command",
                    task_id=task_id,
                    payload=CommandPayload(action=cmd, thought=decision.thought)
                ).model_dump(by_alias=True))

                # Update status message for the user
                await websocket.send_json(OutboundMessage(
                    type="status",
                    task_id=task_id,
                    payload=StatusPayload(message=decision.status_message, status="running")
                ).model_dump(by_alias=True))

                # 6. Wait for ACK
                ack_status = await self._wait_for_ack(task_id, cmd.action_id)
                if ack_status == "failed":
                    logger.warning(f"Task {task_id}: Action failed on device")
                
                # Small delay to let UI settle
                await asyncio.sleep(1.0)

        except asyncio.CancelledError:
            logger.info(f"Task {task_id} was cancelled")
            await self._send_failed(websocket, task_id, "Task stopped by user")
        except Exception as e:
            logger.error(f"Task {task_id} crashed: {e}", exc_info=True)
            await self._send_failed(websocket, task_id, f"Internal Orchestrator Error: {str(e)}")
        finally:
            self._running_tasks.pop(task_id, None)
            self.cleanup_task(task_id)

    async def _wait_for_ack(self, task_id: str, action_id: str) -> str:
        event = asyncio.Event()
        self._ack_events[action_id] = event
        try:
            await asyncio.wait_for(event.wait(), timeout=15.0)
            return self._last_ack_status.get(action_id, "unknown")
        except asyncio.TimeoutError:
            return "timeout"
        finally:
            self._ack_events.pop(action_id, None)
            self._last_ack_status.pop(action_id, None)

    def update_observation(self, task_id: str, nodes: list[NodeData], active_package: str, screenshot: Optional[str] = None, type: str = "observation", reason: Optional[str] = None):
        if type == "blind":
            self._blind_events[task_id] = reason or "unknown"
        else:
            self._blind_events.pop(task_id, None)
            
        self._current_nodes[task_id] = nodes
        self._current_app[task_id] = active_package
        if screenshot:
            self._current_screenshot[task_id] = screenshot
            
        if task_id in self._observation_events:
            self._observation_events[task_id].set()

    def handle_ack(self, action_id: str, status: str):
        self._last_ack_status[action_id] = status
        if action_id in self._ack_events:
            self._ack_events[action_id].set()

    async def _send_failed(self, websocket, task_id: str, message: str):
        try:
            await websocket.send_json(OutboundMessage(
                type="task_failed",
                task_id=task_id,
                payload=StatusPayload(message=message, status="failed")
            ).model_dump(by_alias=True))
        except: pass

    def cleanup_task(self, task_id: str):
        self._current_nodes.pop(task_id, None)
        self._current_app.pop(task_id, None)
        self._current_screenshot.pop(task_id, None)
        self._observation_events.pop(task_id, None)
        self._blind_events.pop(task_id, None)
        self._retry_counts.pop(task_id, None)

    def stop_task(self, task_id: str):
        if task_id in self._running_tasks:
            self._running_tasks[task_id].cancel()
            return True
        return False

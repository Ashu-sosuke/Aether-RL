import asyncio
import logging
import json
from fastapi import WebSocket, WebSocketDisconnect
from sqlalchemy.ext.asyncio import AsyncSession
from models import InboundMessage, StartTaskPayload, ObservationPayload, AckPayload, HitlResponsePayload
from orchestrator import Orchestrator
from memory_store import MemoryStore

logger = logging.getLogger("AetherWS")

# Global orchestrator instance
orchestrator = Orchestrator()
memory = MemoryStore()

async def handle_websocket(websocket: WebSocket, db: AsyncSession):
    await websocket.accept()
    session_id = str(id(websocket))
    logger.info(f"WebSocket connected: {session_id}")
    
    active_tasks = set()

    try:
        while True:
            data = await websocket.receive_text()
            try:
                # 1. Parse Inbound Message
                msg_dict = json.loads(data)
                msg = InboundMessage(**msg_dict)
                
                # 2. Route by type
                if msg.type == "start_task":
                    payload = StartTaskPayload(**msg.payload)
                    task_id = msg.task_id
                    
                    logger.info(f"Received start_task: {task_id} for goal: {payload.goal}")
                    
                    # Initialize task state via IntentParser
                    task_plan = await orchestrator.intent_parser.parse_goal(payload.goal, task_id)
                    
                    # Log to database
                    await memory.log_task(task_plan, "started", payload.user_id)
                    
                    # Start orchestration loop in background
                    active_tasks.add(task_id)
                    asyncio.create_task(orchestrator.run_task(task_plan, websocket, payload.user_id))
                
                elif msg.type == "observation":
                    payload = ObservationPayload(**msg.payload)
                    orchestrator.update_observation(msg.task_id, payload.nodes, payload.active_package)
                
                elif msg.type == "ack":
                    payload = AckPayload(**msg.payload)
                    orchestrator.handle_ack(payload.action_id, payload.status)
                
                elif msg.type == "hitl_response":
                    # Future implementation for Human-In-The-Loop
                    pass
                
                elif msg.type == "stop_task":
                    logger.info(f"Received stop_task for: {msg.task_id}")
                    orchestrator.stop_task(msg.task_id)

            except Exception as e:
                logger.error(f"Error processing message: {e}", exc_info=True)
                # Notify client if it was a task-related error
                try:
                    if 'msg' in locals() and msg.task_id:
                        from models import OutboundMessage, StatusPayload
                        await websocket.send_json(OutboundMessage(
                            type="task_failed",
                            task_id=msg.task_id,
                            payload=StatusPayload(message=f"Server Error: {str(e)}", status="failed")
                        ).model_dump(by_alias=True))
                except:
                    pass

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket session error: {e}", exc_info=True)
    finally:
        # Cleanup any tasks associated with this connection if possible
        for tid in active_tasks:
            orchestrator.cleanup_task(tid)

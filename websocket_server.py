import asyncio
import json
import traceback
from uuid import uuid4
from fastapi import WebSocket, WebSocketDisconnect
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import ValidationError
from models import (
    InboundMessage, StartTaskPayload, ObservationPayload, 
    AckPayload, HitlResponsePayload, TaskPlan
)
from token_bucket import TokenBucket
from orchestrator import TaskOrchestrator, _ack_store
from intent_parser import IntentParser
from semantic_mapper import SemanticMapper
from memory_store import MemoryStore
from config import settings

# Singletons shared between REST and WebSocket
parser = IntentParser()
mapper = SemanticMapper()
memory = MemoryStore()

class SessionContext:
    def __init__(self, session_id: str, ws: WebSocket, bucket: TokenBucket, orchestrator=None, task_plan=None, user_id="user_default"):
        self.session_id    = session_id
        self.ws            = ws
        self.user_id       = user_id
        self.bucket        = bucket
        self.orchestrator: TaskOrchestrator | None = orchestrator
        self.task_plan: TaskPlan | None = task_plan

# Registry of active sessions
sessions: dict[str, SessionContext] = {}

async def handle_websocket(ws: WebSocket, db: AsyncSession, session_id: str):
    # Register session FIRST before any await
    sessions[session_id] = SessionContext(
        session_id=session_id,
        ws=ws,
        bucket=TokenBucket(
            capacity=settings.token_bucket_capacity,
            refill_rate=settings.token_refill_rate
        ),
        orchestrator=None,
        task_plan=None,
        user_id="user_default"
    )
    print(f"[WS] Session registered: {session_id}")

    try:
        async for raw in ws.iter_text():
            print(f"[WS RECV] session={session_id} raw={raw[:200]}")
            try:
                msg = InboundMessage.model_validate_json(raw)
                await _handle_message(session_id, msg, db)
            except ValidationError as e:
                print(f"[WS] Pydantic validation failed: {e}")
            except Exception as e:
                import traceback
                print(f"[WS] Message handler error: {type(e).__name__}: {e}")
                traceback.print_exc()
    except WebSocketDisconnect:
        raise
    except Exception as e:
        print(f"[WS] Receive loop crashed: {type(e).__name__}: {e}")
        raise
    finally:
        sessions.pop(session_id, None)

async def _handle_message(
    session_id: str, 
    msg: InboundMessage, 
    db: AsyncSession
):
    session = sessions.get(session_id)
    if not session:
        print(f"[WS] Error: Message received for unknown session {session_id}")
        return

    if msg.type == "start_task":
        payload = StartTaskPayload(**msg.payload)
        session.user_id = payload.userId
        memory_ctx = await memory.get_context(payload.userId)
        
        # Parse goal into a task plan
        task_plan = await parser.parse_goal(payload.goal, memory_ctx)
        session.task_plan = task_plan
        
        # Initialize and start orchestrator
        orch = TaskOrchestrator(
            ws         = session.ws,
            task       = task_plan,
            session_id = session.session_id,
            user_id    = payload.userId,
            mapper     = mapper,
            memory     = memory,
            bucket     = session.bucket,
            db         = db
        )
        session.orchestrator = orch
        asyncio.create_task(orch.run())

    elif msg.type == "observation":
        payload = ObservationPayload(**msg.payload)
        if session.orchestrator:
            # Clear queue if full, then put new observation
            try:
                session.orchestrator._obs_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            await session.orchestrator._obs_queue.put(payload)

    elif msg.type == "ack":
        payload = AckPayload(**msg.payload)
        _ack_store[payload.actionId] = payload.status

    elif msg.type == "hitl_response":
        payload = HitlResponsePayload(**msg.payload)
        if session.orchestrator:
            await session.orchestrator._hitl_queue.put(payload.approved)

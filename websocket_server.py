import asyncio
import json
import traceback
from uuid import uuid4
from fastapi import WebSocket, WebSocketDisconnect
from sqlalchemy.ext.asyncio import AsyncSession
from models import (
    InboundMessage, StartTaskPayload, ObservationPayload, 
    AckPayload, HitlResponsePayload, TaskPlan
)
from token_bucket import TokenBucket
from orchestrator import TaskOrchestrator, _ack_store
from intent_parser import IntentParser
from semantic_mapper import SemanticMapper
from memory_store import MemoryStore

class SessionContext:
    def __init__(self, session_id: str, ws: WebSocket):
        self.session_id    = session_id
        self.ws            = ws
        self.user_id: str | None = None
        self.bucket        = TokenBucket()
        self.orchestrator: TaskOrchestrator | None = None
        self.task_plan: TaskPlan | None = None

# Registry of active sessions
sessions: dict[str, SessionContext] = {}

async def handle_websocket(
    ws: WebSocket, 
    db: AsyncSession, 
    parser: IntentParser, 
    mapper: SemanticMapper, 
    memory: MemoryStore
):
    session_id = str(uuid4())
    session = SessionContext(session_id, ws)
    print(f"DEBUG: New connection attempt. Assigning session_id: {session_id}")
    
    await ws.accept()
    print(f"DEBUG: Session {session_id} accepted.")
    sessions[session_id] = session
    
    try:
        async for raw in ws.iter_text():
            print(f"DEBUG: Session {session_id} received raw message: {raw[:100]}...")
            try:
                data = json.loads(raw)
                msg = InboundMessage(**data)
                print(f"DEBUG: Session {session_id} processing message type: {msg.type}")
                await _handle_message(session, msg, db, parser, mapper, memory)
            except Exception as e:
                print(f"ERROR handling message in session {session_id}: {e}")
                traceback.print_exc()
                
    except WebSocketDisconnect:
        print(f"INFO: Session {session_id} disconnected by client (WebSocketDisconnect)")
    except Exception as e:
        print(f"CRITICAL error in session {session_id} loop: {e}")
        traceback.print_exc()
    finally:
        print(f"DEBUG: Cleaning up session {session_id}")
        sessions.pop(session_id, None)

async def _handle_message(
    session: SessionContext, 
    msg: InboundMessage, 
    db: AsyncSession,
    parser: IntentParser, 
    mapper: SemanticMapper, 
    memory: MemoryStore
):
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

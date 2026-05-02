from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import uuid

import traceback
import sys

try:
    print("INFO: Initializing settings...")
    from config import settings
    print(f"INFO: Environment: {settings.environment}")
    
    print("INFO: Importing modules...")
    from db import engine, Base, get_db, ActionLogEntry
    from intent_parser import IntentParser
    from semantic_mapper import SemanticMapper
    from memory_store import MemoryStore
    from websocket_server import handle_websocket, sessions
    print("INFO: Modules imported successfully.")
except Exception as e:
    print("CRITICAL: Error during module-level imports/init")
    # If it's a Pydantic ValidationError, it's likely missing env vars
    if "ValidationError" in str(type(e)):
        print("ERROR: Missing or invalid Environment Variables!")
    traceback.print_exc()
    sys.exit(1)

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        # Startup: create DB tables
        print("Starting up: Creating database tables...")
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        print("Database tables verified/created.")
        
        # Init wandb if configured
        if settings.wandb_api_key and settings.wandb_mode == "online":
            import wandb
            print("Initializing wandb...")
            wandb.init(project="aether-brain", mode="online")
            
        yield
    except Exception as e:
        print("CRITICAL: Error during lifespan startup")
        traceback.print_exc()
        raise e
    finally:
        # Shutdown
        await engine.dispose()

app = FastAPI(title="Project Aether — Neural Brain", lifespan=lifespan)

# Singletons
parser  = IntentParser()
mapper  = SemanticMapper()
memory  = MemoryStore()

@app.get("/ping")
async def ping():
    """Simple ping for health checks."""
    return {"status": "ok"}

@app.get("/health")
async def health():
    """Detailed health check."""
    return {"status": "healthy", "environment": settings.environment}

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket, db: AsyncSession = Depends(get_db)):
    """Main WebSocket entry point for Android clients."""
    await handle_websocket(ws, db, parser, mapper, memory)

@app.post("/task")
async def create_task(body: dict, db: AsyncSession = Depends(get_db)):
    """Create a task plan from a goal (REST API)."""
    user_id = body.get("user_id", "user_default")
    goal    = body.get("goal", "")
    ctx     = await memory.get_context(user_id)
    plan    = await parser.parse_goal(goal, ctx)
    return {"task_id": plan.taskId, "steps": len(plan.steps)}

@app.get("/task/{task_id}/log")
async def get_task_log(task_id: str, db: AsyncSession = Depends(get_db)):
    """Retrieve action logs for a specific task."""
    try:
        task_uuid = uuid.UUID(task_id)
    except ValueError:
        raise HTTPException(400, "Invalid task_id format")
        
    result = await db.execute(
        select(ActionLogEntry).where(
            ActionLogEntry.task_id == task_uuid
        ).order_by(ActionLogEntry.timestamp)
    )
    rows = result.scalars().all()
    return [
        {
            "step_id": r.step_id, 
            "action_type": r.action_type,
            "status": r.status, 
            "timestamp": str(r.timestamp)
        }
        for r in rows
    ]

@app.get("/session/{session_id}/tokens")
async def get_tokens(session_id: str):
    """Check remaining tokens for a session."""
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    return {"balance": session.bucket.balance()}

@app.post("/memory/{user_id}")
async def set_memory(user_id: str, body: dict):
    """Manually update user memory."""
    if "key" not in body or "value" not in body:
        raise HTTPException(400, "Missing key or value")
    await memory.set_memory(user_id, body["key"], body["value"])
    return {"status": "ok"}

import sys
print("DEBUG: main.py reached", flush=True)

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from uuid import uuid4
import uuid

import traceback
import sys

try:
    print("INFO: Initializing settings...", flush=True)
    from config import settings
    print(f"INFO: Environment: {settings.environment}")
    
    print("INFO: Importing modules...")
    from db import engine, Base, get_db, ActionLogEntry, AsyncSessionLocal
    # Import singletons from websocket_server to avoid duplication
    from websocket_server import handle_websocket, sessions, parser, mapper, memory
    print("INFO: Modules imported successfully.")
except Exception as e:
    print("CRITICAL: Error during module-level imports/init")
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

@app.api_route("/", methods=["GET", "HEAD"])
async def root():
    """Health-check compatible root endpoint with premium UI."""
    # If it's a HEAD request, just return status ok
    # (FastAPI handles the body removal automatically, but let's be explicit if needed)
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Aether Neural Brain | Operational</title>
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;700&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
        <style>
            :root {{
                --bg: #05070a;
                --card-bg: rgba(13, 17, 23, 0.7);
                --accent: #00f2ff;
                --accent-dim: rgba(0, 242, 255, 0.2);
                --text: #e6edf3;
                --text-dim: #8b949e;
                --success: #238636;
            }}
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{
                font-family: 'Outfit', sans-serif;
                background-color: var(--bg);
                color: var(--text);
                height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                overflow: hidden;
                background-image: 
                    radial-gradient(circle at 50% 50%, rgba(0, 242, 255, 0.05) 0%, transparent 50%),
                    linear-gradient(rgba(255,255,255,0.02) 1px, transparent 1px),
                    linear-gradient(90deg, rgba(255,255,255,0.02) 1px, transparent 1px);
                background-size: 100% 100%, 40px 40px, 40px 40px;
            }}
            .container {{
                width: 90%;
                max-width: 800px;
                background: var(--card-bg);
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 24px;
                padding: 40px;
                box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
                position: relative;
                animation: fadeIn 1s ease-out;
            }}
            @keyframes fadeIn {{
                from {{ opacity: 0; transform: translateY(20px); }}
                to {{ opacity: 1; transform: translateY(0); }}
            }}
            .header {{
                display: flex;
                justify-content: space-between;
                align-items: flex-start;
                margin-bottom: 40px;
            }}
            .logo-area h1 {{
                font-size: 2.5rem;
                font-weight: 700;
                background: linear-gradient(90deg, #fff, var(--accent));
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                letter-spacing: -1px;
            }}
            .logo-area p {{
                color: var(--text-dim);
                font-size: 1.1rem;
                margin-top: 4px;
            }}
            .status-badge {{
                background: var(--accent-dim);
                border: 1px solid var(--accent);
                color: var(--accent);
                padding: 6px 12px;
                border-radius: 20px;
                font-size: 0.85rem;
                font-weight: 700;
                display: flex;
                align-items: center;
                gap: 8px;
            }}
            .status-dot {{
                width: 8px;
                height: 8px;
                background: var(--accent);
                border-radius: 50%;
                box-shadow: 0 0 10px var(--accent);
                animation: pulse 2s infinite;
            }}
            @keyframes pulse {{
                0% {{ opacity: 1; }}
                50% {{ opacity: 0.4; }}
                100% {{ opacity: 1; }}
            }}
            .terminal {{
                background: #000;
                border-radius: 12px;
                padding: 20px;
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.9rem;
                margin-bottom: 30px;
                border-left: 4px solid var(--accent);
            }}
            .terminal-line {{ margin-bottom: 8px; display: flex; gap: 10px; }}
            .prompt {{ color: var(--accent); }}
            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                gap: 20px;
                margin-bottom: 40px;
            }}
            .stat-card {{
                background: rgba(255,255,255,0.03);
                padding: 20px;
                border-radius: 16px;
                border: 1px solid rgba(255,255,255,0.05);
            }}
            .stat-card h4 {{ color: var(--text-dim); font-size: 0.8rem; text-transform: uppercase; margin-bottom: 8px; }}
            .stat-card p {{ font-size: 1.2rem; font-weight: 700; }}
            .actions {{
                display: flex;
                gap: 16px;
            }}
            .btn {{
                flex: 1;
                padding: 14px;
                border-radius: 12px;
                text-decoration: none;
                font-weight: 700;
                text-align: center;
                transition: all 0.2s;
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 10px;
            }}
            .btn-primary {{
                background: var(--accent);
                color: #000;
            }}
            .btn-primary:hover {{
                transform: translateY(-2px);
                box-shadow: 0 10px 20px rgba(0, 242, 255, 0.3);
            }}
            .btn-secondary {{
                background: rgba(255,255,255,0.05);
                color: #fff;
                border: 1px solid rgba(255,255,255,0.1);
            }}
            .btn-secondary:hover {{
                background: rgba(255,255,255,0.1);
                transform: translateY(-2px);
            }}
            .brain-bg {{
                position: absolute;
                top: -100px;
                right: -100px;
                width: 300px;
                height: 300px;
                background: var(--accent);
                filter: blur(120px);
                opacity: 0.1;
                pointer-events: none;
            }}
        </style>
    </head>
    <body>
        <div class="brain-bg"></div>
        <div class="container">
            <div class="header">
                <div class="logo-area">
                    <h1>AETHER BRAIN</h1>
                    <p>Neural Orchestration Engine</p>
                </div>
                <div class="status-badge">
                    <div class="status-dot"></div>
                    NEURAL LINK ACTIVE
                </div>
            </div>

            <div class="terminal">
                <div class="terminal-line">
                    <span class="prompt">$</span>
                    <span>initializing_kernel... <span style="color:var(--success)">DONE</span></span>
                </div>
                <div class="terminal-line">
                    <span class="prompt">$</span>
                    <span>checking_subsystems... <span style="color:var(--success)">OPTIMAL</span></span>
                </div>
                <div class="terminal-line">
                    <span class="prompt">$</span>
                    <span>environment: <span style="color:var(--accent)">{settings.environment}</span></span>
                </div>
            </div>

            <div class="stats-grid">
                <div class="stat-card">
                    <h4>Core Model</h4>
                    <p>Gemini 1.5 Pro</p>
                </div>
                <div class="stat-card">
                    <h4>Memory</h4>
                    <p>Active SQL Store</p>
                </div>
                <div class="stat-card">
                    <h4>Capacity</h4>
                    <p>{settings.token_bucket_capacity} Units</p>
                </div>
                <div class="stat-card">
                    <h4>Refill Rate</h4>
                    <p>{settings.token_refill_rate}/s</p>
                </div>
            </div>

            <div class="actions">
                <a href="/docs" class="btn btn-primary">
                    View API Documentation
                </a>
                <a href="/health" class="btn btn-secondary">
                    System Health
                </a>
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.api_route("/ping", methods=["GET", "HEAD"])
async def ping():
    """Simple ping for health checks."""
    return JSONResponse({"status": "ok"})

@app.get("/health")
async def health():
    """Detailed health check."""
    return {
        "status"     : "healthy",
        "environment": settings.environment,
        "version"    : "1.0.0"
    }

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """Main WebSocket entry point with crash protection."""
    await ws.accept()
    session_id = str(uuid4())
    print(f"[WS] New connection — session {session_id}")
    try:
        async with AsyncSessionLocal() as db:
            await handle_websocket(ws, db, session_id)
    except WebSocketDisconnect:
        print(f"[WS] Client disconnected — session {session_id}")
    except Exception as e:
        import traceback
        print(f"[WS CRASH] session {session_id}: {type(e).__name__}: {e}")
        traceback.print_exc()
        try:
            await ws.close(code=1011)
        except Exception:
            pass
    finally:
        sessions.pop(session_id, None)
        print(f"[WS] Session cleaned up — {session_id}")

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

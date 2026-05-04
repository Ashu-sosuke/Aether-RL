import logging
from fastapi import FastAPI, Depends, WebSocket, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
from database import get_db, init_db
from websocket_server import handle_websocket
from config import settings
import time

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("AetherMain")

app = FastAPI(title="Aether Brain API")

# CORS for UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    logger.info("Initializing Aether Brain...")
    await init_db()
    logger.info("Initialization complete.")

# --- Health Checks (Bug 10) ---

@app.get("/")
async def root():
    return {
        "status": "online",
        "service": "Aether Neural Brain",
        "version": "2.0.0-gemini-flash",
        "timestamp": time.time()
    }

@app.get("/ping")
async def ping():
    return "pong"

# --- WebSocket ---

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, db: AsyncSession = Depends(get_db)):
    await handle_websocket(websocket, db)

# --- Admin UI (Minimal) ---

@app.get("/ui", response_class=HTMLResponse)
async def admin_ui():
    return """
    <html>
        <head>
            <title>Aether Control</title>
            <style>
                body { font-family: sans-serif; background: #0a0a0a; color: #00ff00; padding: 20px; }
                .card { border: 1px solid #333; padding: 20px; border-radius: 8px; background: #111; }
                h1 { color: #fff; border-bottom: 2px solid #00ff00; padding-bottom: 10px; }
                .status-on { color: #00ff00; }
            </style>
        </head>
        <body>
            <div class="card">
                <h1>Aether Brain v2.0</h1>
                <p>Status: <span class="status-on">OPERATIONAL</span></p>
                <p>LLM: <b>gemini-2.0-flash</b></p>
                <p>Environment: <b>production</b></p>
            </div>
        </body>
    </html>
    """

# --- Global Exception Handler ---

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global Exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"message": "Internal Server Error", "detail": str(exc)},
    )

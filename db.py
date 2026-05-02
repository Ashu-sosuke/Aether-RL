from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from sqlalchemy import Column, String, Boolean, DateTime, Integer, func
from sqlalchemy.dialects.postgresql import UUID
from uuid import uuid4
from typing import AsyncGenerator
from config import settings

try:
    print("INFO: Initializing database engine...", flush=True)
    engine = create_async_engine(
        settings.async_database_url,
        echo=False,
        connect_args=settings.async_database_connect_args,
    )
    AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession,
                                      expire_on_commit=False)
    print("INFO: Database engine initialized.", flush=True)
except Exception as e:
    print("CRITICAL: Failed to initialize database engine!", flush=True)
    import traceback
    traceback.print_exc()
    raise e

class Base(DeclarativeBase): pass

class ActionLogEntry(Base):
    __tablename__ = "action_log"
    id             = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    task_id        = Column(UUID(as_uuid=True), nullable=False, index=True)
    step_id        = Column(String(64))
    action_type    = Column(String(32))
    node_id        = Column(String(128))
    app_package    = Column(String(128))
    status         = Column(String(16))
    hitl_required  = Column(Boolean, default=False)
    hitl_approved  = Column(Boolean, nullable=True)
    timestamp      = Column(DateTime(timezone=True), server_default=func.now())

class SessionRecord(Base):
    __tablename__ = "sessions"
    session_id      = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    connected_at    = Column(DateTime(timezone=True), server_default=func.now())
    disconnected_at = Column(DateTime(timezone=True), nullable=True)
    token_balance   = Column(Integer, default=100)

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise

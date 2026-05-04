from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from sqlalchemy import Column, String, Boolean, DateTime, Integer, func, TypeDecorator
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
import uuid
from typing import AsyncGenerator
from config import settings

# Custom UUID type for SQLite compatibility
class GUID(TypeDecorator):
    """Platform-independent GUID type.
    Uses PostgreSQL's UUID type, otherwise uses CHAR(32), storing as string without dashes.
    """
    impl = String(36)
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == 'postgresql':
            return dialect.type_descriptor(PG_UUID(as_uuid=True))
        else:
            return dialect.type_descriptor(String(36))

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        elif dialect.name == 'postgresql':
            return value
        else:
            return str(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        else:
            if not isinstance(value, uuid.UUID):
                return uuid.UUID(value)
            return value

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
    print("CRITICAL: Failed to initialize database engine!")
    import traceback
    traceback.print_exc()
    raise e

class Base(DeclarativeBase): pass

class ActionLogEntry(Base):
    __tablename__ = "action_log"
    id             = Column(GUID, primary_key=True, default=uuid.uuid4)
    task_id        = Column(GUID, nullable=False, index=True)
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
    session_id      = Column(GUID, primary_key=True, default=uuid.uuid4)
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

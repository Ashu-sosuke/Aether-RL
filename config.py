import os
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from dotenv import load_dotenv

# Load .env if it exists (for local development)
load_dotenv()

class Settings:
    def __init__(self):
        print("INFO: Loading environment variables...", flush=True)
        # LLM
        self.gemini_api_key = self._get_required("GEMINI_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")

        # Database
        self.database_url   = self._get_required("DATABASE_URL")
        self.supabase_url   = self._get_required("SUPABASE_URL")
        self.supabase_key   = self._get_required("SUPABASE_KEY")

        # Redis
        self.redis_url      = os.getenv("REDIS_URL", "")

        # Wandb
        self.wandb_api_key  = os.getenv("WANDB_API_KEY", "")
        self.wandb_mode     = os.getenv("WANDB_MODE", "disabled")

        # App
        self.environment    = os.getenv("ENVIRONMENT", "production")
        self.server_url     = os.getenv("SERVER_URL", "http://localhost:8000")
        self.token_bucket_capacity = int(os.getenv("TOKEN_BUCKET_CAPACITY", "100"))
        self.token_refill_rate     = float(os.getenv("TOKEN_REFILL_RATE", "10.0"))

        print(f"INFO: Config loaded. Environment: {self.environment}", flush=True)

    def _get_required(self, key: str) -> str:
        val = os.getenv(key)
        if not val:
            print(f"CRITICAL: Missing required environment variable: {key}", flush=True)
            # Don't raise here, let the full Traceback happen in the try/except block
            raise ValueError(f"Missing required environment variable: {key}")
        return val

    @property
    def async_database_url(self) -> str:
        url = self.database_url.strip()
        
        # 1. Handle both postgres:// and postgresql:// schemes
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql+asyncpg://", 1)
        elif url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
        
        return self._without_asyncpg_ssl_query_params(url)

    @property
    def async_database_connect_args(self) -> dict[str, Any]:
        url = self.async_database_url
        parsed = urlsplit(url)

        if parsed.scheme != "postgresql+asyncpg":
            return {}

        host = (parsed.hostname or "").lower()
        if host in {"localhost", "127.0.0.1", "::1"}:
            return {"timeout": 10}

        ssl_enabled = os.getenv("DB_SSL", "true").lower() == "true"
        return {"ssl": ssl_enabled, "timeout": 10, "statement_cache_size": 0}

    @property
    def redacted_database_target(self) -> str:
        parsed = urlsplit(self.async_database_url)
        host = parsed.hostname or "unknown-host"
        port = parsed.port or 5432
        database = parsed.path.lstrip("/") or "unknown-db"
        return f"{parsed.scheme}://{host}:{port}/{database}"

    def _without_asyncpg_ssl_query_params(self, url: str) -> str:
        parsed = urlsplit(url)
        if parsed.scheme != "postgresql+asyncpg":
            return url

        query_items = [
            (key, value)
            for key, value in parse_qsl(parsed.query, keep_blank_values=True)
            if key.lower() not in {"sslmode", "ssl"}
        ]
        return urlunsplit(parsed._replace(query=urlencode(query_items)))

try:
    settings = Settings()
except Exception as e:
    print(f"CRITICAL: Settings initialization failed: {e}", flush=True)
    import sys
    sys.exit(1)

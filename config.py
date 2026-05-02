import os
import re
from typing import Optional
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
        url = self.database_url
        
        # 1. Handle both postgres:// and postgresql:// schemes
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql+asyncpg://", 1)
        elif url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
        
        # 2. Aggressively strip sslmode (case-insensitive)
        # This regex removes ?sslmode=... or &sslmode=...
        url = re.sub(r'([?&])sslmode=[^&]*', '', url, flags=re.IGNORECASE)
        
        # 3. Clean up potential artifacts like ?& or trailing ? or &
        url = url.replace("?&", "?").replace("&&", "&").rstrip("?").rstrip("&")

        # 4. Enforce ssl=true for Supabase/Cloud hosts if not already there
        if ("supabase.co" in url or "render.com" in url) and "ssl=" not in url.lower():
            separator = "&" if "?" in url else "?"
            url += f"{separator}ssl=true"
            
        return url

try:
    settings = Settings()
except Exception as e:
    print(f"CRITICAL: Settings initialization failed: {e}", flush=True)
    import sys
    sys.exit(1)

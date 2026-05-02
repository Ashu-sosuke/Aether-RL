from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # LLM
    gemini_api_key       : str
    openai_api_key       : str = ""

    # Database
    database_url         : str   # asyncpg postgresql+asyncpg://...
    supabase_url         : str
    supabase_key         : str

    # Redis
    redis_url            : str = ""   # empty = disable Redis

    # Wandb
    wandb_api_key        : str = ""
    wandb_mode           : str = "disabled"   # "online" | "disabled"

    # App
    environment          : str = "production"
    server_url           : str = "http://localhost:8000"
    token_bucket_capacity: int = 100
    token_refill_rate    : float = 10.0   # per minute

    @property
    def async_database_url(self) -> str:
        url = self.database_url
        if url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
        
        # Ensure SSL for Supabase/Production
        # asyncpg expects 'ssl=require' rather than 'sslmode=require'
        if "supabase.co" in url and "ssl" not in url:
            separator = "&" if "?" in url else "?"
            url += f"{separator}ssl=require"
            
        return url

    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()

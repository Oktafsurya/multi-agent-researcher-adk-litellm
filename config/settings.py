"""
Application settings loaded from environment variables / .env file.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # AWS Bedrock
    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "")
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    AWS_REGION_NAME: str = os.getenv("AWS_REGION_NAME", "us-east-1")

    # LiteLLM Proxy
    LITELLM_PROXY_URL: str = os.getenv("LITELLM_PROXY_URL", "http://localhost:4000")
    LITELLM_MASTER_KEY: str = os.getenv("LITELLM_MASTER_KEY", "sk-local-master-key-change-me")

    # Redis
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")

    # PostgreSQL — memory store (litellm-db, host port 5433)
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://litellm:litellm_secret@localhost:5433/litellm")

    # Langfuse
    LANGFUSE_SECRET_KEY: str = os.getenv("LANGFUSE_SECRET_KEY", "")
    LANGFUSE_PUBLIC_KEY: str = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    LANGFUSE_HOST: str = os.getenv("LANGFUSE_HOST", "http://localhost:3000")

    # Model aliases (must match model_name keys in litellm_config.yaml)
    ORCHESTRATOR_MODEL: str = os.getenv("ORCHESTRATOR_MODEL", "orchestrator-model")
    SUBAGENT_MODEL: str = os.getenv("SUBAGENT_MODEL", "subagent-model")

    # App
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")


settings = Settings()

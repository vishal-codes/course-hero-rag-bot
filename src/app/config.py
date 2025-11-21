from dataclasses import dataclass
from typing import Any
from .utils import env_get

@dataclass
class Config:
    """Application configuration with defaults"""
    embed_model: str = "@cf/baai/bge-base-en-v1.5"
    chat_model: str = "@cf/meta/llama-3.1-8b-instruct-fast"
    topk: int = 5
    temperature: float = 0.2
    max_tokens: int = 350
    app_version: str = "dev"

    @classmethod
    def from_env(cls, env: Any) -> "Config":
        return cls(
            embed_model=env_get(env, "CF_EMBED_MODEL", cls.embed_model),
            chat_model=env_get(env, "CF_CHAT_MODEL", cls.chat_model),
            topk=int(env_get(env, "CF_TOPK", cls.topk)),
            temperature=float(env_get(env, "GEN_TEMPERATURE", cls.temperature)),
            max_tokens=int(env_get(env, "GEN_MAX_TOKENS", cls.max_tokens)),
            app_version=env_get(env, "APP_VERSION", cls.app_version),
        )

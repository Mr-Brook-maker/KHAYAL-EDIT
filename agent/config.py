"""
Agent Configuration Module
Manages all API keys, model selections, and global constants.
Production-grade: uses pydantic-settings for env validation.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from enum import Enum
from pathlib import Path


class LLMProvider(str, Enum):
    GEMINI = "gemini"
    GROQ = "groq"


class Settings(BaseSettings):
    # ── LLM Provider ──────────────────────────────────────────
    llm_provider: LLMProvider = Field(
        default=LLMProvider.GEMINI,
        description="Primary LLM provider for agent reasoning"
    )

    # ── Gemini (Google AI Studio — Free Tier) ─────────────────
    google_api_key: str = Field(default="", env="GOOGLE_API_KEY")
    gemini_model: str = "gemini-1.5-flash"          # Free tier, high RPM
    gemini_temperature: float = 0.1                  # Low for deterministic tool calls
    gemini_max_tokens: int = 2048

    # ── Groq (Fallback — Free Tier) ───────────────────────────
    groq_api_key: str = Field(default="", env="GROQ_API_KEY")
    groq_model: str = "llama3-70b-8192"             # Fast inference, free
    groq_temperature: float = 0.1

    # ── HuggingFace Inference API ─────────────────────────────
    hf_api_token: str = Field(default="", env="HF_API_TOKEN")

    # ── Fal.ai (Free credits available) ───────────────────────
    fal_api_key: str = Field(default="", env="FAL_API_KEY")

    # ── Storage ───────────────────────────────────────────────
    output_dir: Path = Path("./outputs")
    temp_dir: Path = Path("./temp")
    max_image_size_mb: int = 10

    # ── Agent Behavior ────────────────────────────────────────
    agent_max_iterations: int = 10
    agent_verbose: bool = True
    enable_memory: bool = True
    memory_window_size: int = 5          # Last N exchanges kept in context

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


settings = Settings()
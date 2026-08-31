"""
LLM Factory
Provides a unified interface for Gemini Flash and Groq backends.
Implements automatic fallback: if primary fails, switch to secondary.
"""

import logging
from functools import lru_cache
from langchain_core.language_models import BaseChatModel
from agent.config import settings, LLMProvider

logger = logging.getLogger(__name__)


def _build_gemini() -> BaseChatModel:
    """Google Gemini 1.5 Flash — Free tier: 15 RPM, 1M TPM."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.google_api_key,
        temperature=settings.gemini_temperature,
        max_output_tokens=settings.gemini_max_tokens,
        convert_system_message_to_human=True,  # Gemini requirement
    )


def _build_groq() -> BaseChatModel:
    """Groq LPU — Free tier: 30 RPM, extremely fast inference."""
    from langchain_groq import ChatGroq
    return ChatGroq(
        model=settings.groq_model,
        groq_api_key=settings.groq_api_key,
        temperature=settings.groq_temperature,
        max_tokens=settings.gemini_max_tokens,
    )


@lru_cache(maxsize=1)
def get_llm(provider: LLMProvider = None) -> BaseChatModel:
    """
    Cached LLM factory. Returns configured model instance.
    Falls back to Groq if Gemini key not set, and vice versa.
    """
    target = provider or settings.llm_provider

    builders = {
        LLMProvider.GEMINI: (_build_gemini, LLMProvider.GROQ, _build_groq),
        LLMProvider.GROQ:   (_build_groq, LLMProvider.GEMINI, _build_gemini),
    }

    primary_builder, fallback_name, fallback_builder = builders[target]

    try:
        llm = primary_builder()
        logger.info(f"LLM initialized: {target.value}")
        return llm
    except Exception as e:
        logger.warning(f"Primary LLM '{target.value}' failed ({e}). Falling back to {fallback_name.value}.")
        return fallback_builder()
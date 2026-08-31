"""
Rate Limiter & API Key Rotation System

Architecture:
- Per-provider token bucket (tracks remaining quota)
- Round-robin key rotation across multiple keys of same provider
- Automatic fallback chain: Primary API → Secondary API → Local
- Exponential backoff on 429 responses
- Quota state persisted to disk (survives restarts)
"""

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Any

logger = logging.getLogger(__name__)

QUOTA_STATE_PATH = Path(os.getenv("QUOTA_STATE_PATH", "./temp/quota_state.json"))


class Provider(str, Enum):
    GEMINI       = "gemini"
    GROQ         = "groq"
    HUGGINGFACE  = "huggingface"
    REPLICATE    = "replicate"
    FAL          = "fal"


@dataclass
class KeyState:
    key:            str
    requests_made:  int   = 0
    tokens_used:    int   = 0
    last_used:      float = 0.0
    backoff_until:  float = 0.0    # Epoch timestamp — don't use before this
    failures:       int   = 0

    @property
    def is_available(self) -> bool:
        return time.time() >= self.backoff_until

    def record_failure(self, backoff_s: float = 60.0):
        self.failures    += 1
        # Exponential backoff: 60s, 120s, 240s … cap at 1 hour
        delay             = min(backoff_s * (2 ** (self.failures - 1)), 3600)
        self.backoff_until = time.time() + delay
        logger.warning(f"Key ...{self.key[-6:]} backed off for {delay:.0f}s (failure #{self.failures})")

    def record_success(self):
        self.failures    = 0
        self.backoff_until = 0.0
        self.last_used   = time.time()
        self.requests_made += 1


@dataclass
class ProviderPool:
    provider:   Provider
    keys:       List[KeyState] = field(default_factory=list)
    _current_idx: int = field(default=0, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add_key(self, key: str):
        if key and key not in {k.key for k in self.keys}:
            self.keys.append(KeyState(key=key))

    def next_available_key(self) -> Optional[KeyState]:
        """Round-robin selection of first available (non-backed-off) key."""
        with self._lock:
            if not self.keys:
                return None
            for _ in range(len(self.keys)):
                idx = self._current_idx % len(self.keys)
                self._current_idx += 1
                key_state = self.keys[idx]
                if key_state.is_available:
                    return key_state
            return None  # All keys backed off


class RateLimitedExecutor:
    """
    Central coordinator for all API calls.
    
    Usage:
        result = executor.call(
            provider=Provider.GEMINI,
            fn=my_api_function,
            args=(arg1,),
            fallback_fn=local_fallback,
            fallback_args=(arg1,),
        )
    """

    def __init__(self):
        self._pools: Dict[Provider, ProviderPool] = {
            p: ProviderPool(provider=p) for p in Provider
        }
        self._lock = threading.Lock()
        self._load_state()
        self._register_keys_from_env()

    def _register_keys_from_env(self):
        """Load all API keys from env — supports comma-separated multiple keys."""

        key_map = {
            Provider.GEMINI:      "GOOGLE_API_KEY",
            Provider.GROQ:        "GROQ_API_KEY",
            Provider.HUGGINGFACE: "HF_API_TOKEN",
            Provider.REPLICATE:   "REPLICATE_API_TOKEN",
            Provider.FAL:         "FAL_API_KEY",
        }
        multi_key_map = {
            Provider.GEMINI:      "GOOGLE_API_KEYS",     # Comma-separated fallback keys
            Provider.GROQ:        "GROQ_API_KEYS",
            Provider.HUGGINGFACE: "HF_API_TOKENS",
        }

        for provider, env_var in key_map.items():
            key = os.getenv(env_var, "")
            if key:
                self._pools[provider].add_key(key)

        for provider, env_var in multi_key_map.items():
            keys_str = os.getenv(env_var, "")
            for k in keys_str.split(","):
                k = k.strip()
                if k:
                    self._pools[provider].add_key(k)

        for p, pool in self._pools.items():
            logger.info(f"Provider {p.value}: {len(pool.keys)} key(s) registered")

    def call(
        self,
        provider: Provider,
        fn: Callable,
        args: tuple = (),
        kwargs: dict = None,
        fallback_fn: Optional[Callable] = None,
        fallback_args: tuple = (),
        fallback_kwargs: dict = None,
        retries: int = 3,
    ) -> Any:
        """
        Execute fn with rate-limit awareness.
        On quota error (429/503) → backoff current key, try next key.
        If all keys exhausted → run fallback_fn.
        """
        kwargs          = kwargs or {}
        fallback_kwargs = fallback_kwargs or {}
        pool            = self._pools[provider]

        for attempt in range(retries):
            key_state = pool.next_available_key()

            if key_state is None:
                logger.warning(f"All {provider.value} keys backed off — using fallback")
                if fallback_fn:
                    return fallback_fn(*fallback_args, **fallback_kwargs)
                raise RuntimeError(f"No available {provider.value} keys and no fallback defined")

            try:
                # Inject the active key into kwargs if function accepts 'api_key'
                call_kwargs = kwargs.copy()
                if "api_key" in fn.__code__.co_varnames:
                    call_kwargs["api_key"] = key_state.key

                result = fn(*args, **call_kwargs)
                key_state.record_success()
                self._save_state()
                return result

            except Exception as e:
                err_str = str(e).lower()
                is_rate_limit = any(
                    x in err_str for x in ["429", "rate limit", "quota", "too many requests"]
                )
                is_server_err = any(
                    x in err_str for x in ["503", "502", "server error", "overloaded"]
                )

                if is_rate_limit:
                    key_state.record_failure(backoff_s=60)
                    logger.warning(f"Rate limit hit on {provider.value}, rotating key...")
                    continue

                if is_server_err and attempt < retries - 1:
                    wait = 5 * (attempt + 1)
                    logger.warning(f"Server error on {provider.value}, retrying in {wait}s...")
                    time.sleep(wait)
                    continue

                # Non-retryable error
                key_state.record_failure(backoff_s=30)
                if fallback_fn:
                    logger.warning(f"Non-retryable error on {provider.value}: {e}. Using fallback.")
                    return fallback_fn(*fallback_args, **fallback_kwargs)
                raise

        # All retries exhausted
        if fallback_fn:
            return fallback_fn(*fallback_args, **fallback_kwargs)
        raise RuntimeError(f"All {retries} attempts failed for {provider.value}")

    # ── State Persistence ─────────────────────────────────────────────────────

    def _save_state(self):
        """Persist quota/backoff state to JSON (survives restarts)."""
        try:
            QUOTA_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            state = {
                p.value: [asdict(k) for k in pool.keys]
                for p, pool in self._pools.items()
            }
            with open(QUOTA_STATE_PATH, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.debug(f"State save failed (non-critical): {e}")

    def _load_state(self):
        """Restore backoff state from previous run."""
        if not QUOTA_STATE_PATH.exists():
            return
        try:
            with open(QUOTA_STATE_PATH) as f:
                state = json.load(f)
            for p_str, key_states in state.items():
                try:
                    provider = Provider(p_str)
                except ValueError:
                    continue
                for ks_data in key_states:
                    existing_keys = {k.key for k in self._pools[provider].keys}
                    if ks_data["key"] not in existing_keys:
                        continue
                    # Only restore active backoffs (not resolved ones)
                    if ks_data["backoff_until"] > time.time():
                        ks = next(k for k in self._pools[provider].keys if k.key == ks_data["key"])
                        ks.backoff_until = ks_data["backoff_until"]
                        ks.failures      = ks_data["failures"]
            logger.info("Quota state restored from disk")
        except Exception as e:
            logger.warning(f"Could not restore quota state: {e}")

    def get_status(self) -> dict:
        """Return current health of all provider key pools."""
        status = {}
        for provider, pool in self._pools.items():
            status[provider.value] = {
                "total_keys": len(pool.keys),
                "available":  sum(1 for k in pool.keys if k.is_available),
                "keys": [
                    {
                        "suffix":     f"...{k.key[-6:]}",
                        "available":  k.is_available,
                        "failures":   k.failures,
                        "requests":   k.requests_made,
                        "backoff_remaining_s": max(0, k.backoff_until - time.time()),
                    }
                    for k in pool.keys
                ],
            }
        return status


# Singleton
_executor_instance: Optional[RateLimitedExecutor] = None

def get_executor() -> RateLimitedExecutor:
    global _executor_instance
    if _executor_instance is None:
        _executor_instance = RateLimitedExecutor()
    return _executor_instance
"""
File Cache & Cleanup System

Strategies:
1. Content-addressed cache: SHA256(input_path + params) → output_path
   Identical operations return cached result instantly.
2. LRU eviction when disk usage exceeds threshold.
3. TTL-based expiry (configurable, default 24h).
4. Memory-mapped index for fast lookup without loading files.
"""

import hashlib
import json
import logging
import os
import shutil
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)

CACHE_DIR       = Path(os.getenv("CACHE_DIR", "./cache"))
CACHE_INDEX     = CACHE_DIR / "index.json"
MAX_CACHE_GB    = float(os.getenv("MAX_CACHE_GB", "2.0"))
CACHE_TTL_HOURS = int(os.getenv("CACHE_TTL_HOURS", "24"))


@dataclass
class CacheEntry:
    key:         str
    output_path: str
    input_hash:  str
    params_hash: str
    created_at:  float
    last_used:   float
    size_bytes:  int
    operation:   str

    @property
    def is_expired(self) -> bool:
        ttl = CACHE_TTL_HOURS * 3600
        return (time.time() - self.created_at) > ttl

    @property
    def is_valid(self) -> bool:
        return Path(self.output_path).exists() and not self.is_expired


class FileCache:
    """
    Thread-safe content-addressed file cache.
    
    Cache key = SHA256(file_content_hash + sorted_params_json)
    This ensures:
    - Same file + same params = cache hit regardless of filename
    - File rename doesn't break cache
    - Param order doesn't matter
    """

    def __init__(self):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self._index: Dict[str, CacheEntry] = {}
        self._lock  = threading.RLock()
        self._load_index()

    # ── Public API ────────────────────────────────────────────────────────────

    def get(self, input_path: str, operation: str, params: dict) -> Optional[str]:
        """
        Look up cached result.
        Returns output_path if cache hit and file still exists, else None.
        """
        key = self._make_key(input_path, operation, params)
        with self._lock:
            entry = self._index.get(key)
            if entry and entry.is_valid:
                entry.last_used = time.time()
                self._save_index()
                logger.debug(f"Cache HIT: {operation} ({key[:12]}...)")
                return entry.output_path
            if entry:
                # Stale entry
                del self._index[key]
        logger.debug(f"Cache MISS: {operation}")
        return None

    def put(
        self,
        input_path: str,
        operation: str,
        params: dict,
        output_path: str,
    ) -> str:
        """Register a completed operation result in the cache."""
        key     = self._make_key(input_path, operation, params)
        out     = Path(output_path)
        size    = out.stat().st_size if out.exists() else 0

        entry = CacheEntry(
            key=key,
            output_path=output_path,
            input_hash=self._hash_file(input_path),
            params_hash=self._hash_params(params),
            created_at=time.time(),
            last_used=time.time(),
            size_bytes=size,
            operation=operation,
        )

        with self._lock:
            self._index[key] = entry
            self._save_index()

        logger.debug(f"Cache stored: {operation} → {output_path}")
        self._maybe_evict()
        return output_path

    def invalidate(self, input_path: str):
        """Remove all cached results derived from a specific input file."""
        file_hash = self._hash_file(input_path)
        with self._lock:
            to_remove = [
                k for k, e in self._index.items()
                if e.input_hash == file_hash
            ]
            for k in to_remove:
                del self._index[k]
            if to_remove:
                self._save_index()
                logger.info(f"Invalidated {len(to_remove)} cache entries for {input_path}")

    def get_stats(self) -> dict:
        """Return cache statistics for monitoring."""
        with self._lock:
            total_size = sum(e.size_bytes for e in self._index.values())
            valid      = sum(1 for e in self._index.values() if e.is_valid)
            return {
                "total_entries": len(self._index),
                "valid_entries": valid,
                "total_size_mb": round(total_size / 1_048_576, 2),
                "max_size_gb":   MAX_CACHE_GB,
                "ttl_hours":     CACHE_TTL_HOURS,
            }

    def cleanup(self, force: bool = False):
        """
        Remove expired and missing-file entries.
        If force=True, clear entire cache regardless of TTL.
        """
        with self._lock:
            if force:
                for entry in self._index.values():
                    try:
                        Path(entry.output_path).unlink(missing_ok=True)
                    except Exception:
                        pass
                self._index.clear()
                logger.info("Cache force-cleared")
            else:
                stale = [k for k, e in self._index.items() if not e.is_valid]
                for k in stale:
                    del self._index[k]
                logger.info(f"Cache: removed {len(stale)} stale entries")
            self._save_index()

    # ── Eviction ──────────────────────────────────────────────────────────────

    def _maybe_evict(self):
        """LRU eviction when total cache size exceeds MAX_CACHE_GB."""
        with self._lock:
            total_bytes = sum(e.size_bytes for e in self._index.values())
            max_bytes   = MAX_CACHE_GB * 1_073_741_824

            if total_bytes <= max_bytes:
                return

            # Sort by last_used (LRU first)
            sorted_entries = sorted(
                self._index.items(), key=lambda x: x[1].last_used
            )

            freed = 0
            for key, entry in sorted_entries:
                if total_bytes - freed <= max_bytes * 0.8:  # Evict to 80% threshold
                    break
                try:
                    p = Path(entry.output_path)
                    freed += entry.size_bytes
                    p.unlink(missing_ok=True)
                    del self._index[key]
                    logger.info(f"LRU evicted: {entry.output_path} ({entry.size_bytes/1024:.0f}KB)")
                except Exception as e:
                    logger.debug(f"Eviction failed for {entry.output_path}: {e}")

            self._save_index()
            logger.info(f"Cache eviction freed {freed/1_048_576:.1f}MB")

    # ── Hashing ───────────────────────────────────────────────────────────────

    @staticmethod
    def _hash_file(path: str) -> str:
        """SHA256 of file content (first 1MB for speed on large videos)."""
        h = hashlib.sha256()
        try:
            with open(path, "rb") as f:
                h.update(f.read(1_048_576))   # Sample first 1MB
                h.update(str(Path(path).stat().st_size).encode())
        except FileNotFoundError:
            h.update(path.encode())
        return h.hexdigest()[:16]

    @staticmethod
    def _hash_params(params: dict) -> str:
        """Deterministic hash of operation parameters."""
        canonical = json.dumps(params, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    @staticmethod
    def _make_key(input_path: str, operation: str, params: dict) -> str:
        content = f"{FileCache._hash_file(input_path)}:{operation}:{FileCache._hash_params(params)}"
        return hashlib.sha256(content.encode()).hexdigest()[:32]

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save_index(self):
        try:
            data = {k: asdict(v) for k, v in self._index.items()}
            with open(CACHE_INDEX, "w") as f:
                json.dump(data, f)
        except Exception as e:
            logger.debug(f"Cache index save failed: {e}")

    def _load_index(self):
        if not CACHE_INDEX.exists():
            return
        try:
            with open(CACHE_INDEX) as f:
                data = json.load(f)
            self._index = {k: CacheEntry(**v) for k, v in data.items()}
            # Remove entries whose files no longer exist
            self._index = {k: v for k, v in self._index.items() if Path(v.output_path).exists()}
            logger.info(f"Cache loaded: {len(self._index)} entries")
        except Exception as e:
            logger.warning(f"Cache index load failed: {e}")
            self._index = {}


# ── Disk Cleanup Scheduler ────────────────────────────────────────────────────

class DiskCleanupScheduler:
    """
    Background thread that periodically cleans temp files and old outputs.
    """

    def __init__(
        self,
        temp_dir:      Path = Path("./temp"),
        output_dir:    Path = Path("./outputs"),
        temp_ttl_min:  int  = 30,       # Temp files older than 30 minutes
        output_ttl_h:  int  = 48,       # Output files older than 48 hours
        interval_min:  int  = 15,       # Run cleanup every 15 minutes
        max_output_gb: float = 5.0,
    ):
        self.temp_dir      = temp_dir
        self.output_dir    = output_dir
        self.temp_ttl      = temp_ttl_min * 60
        self.output_ttl    = output_ttl_h * 3600
        self.interval      = interval_min * 60
        self.max_output_gb = max_output_gb
        self._thread: Optional[threading.Thread] = None
        self._stop_event   = threading.Event()

    def start(self):
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="DiskCleanup"
        )
        self._thread.start()
        logger.info("Disk cleanup scheduler started")

    def stop(self):
        self._stop_event.set()

    def _loop(self):
        while not self._stop_event.wait(timeout=self.interval):
            self._clean_temp()
            self._clean_outputs()

    def _clean_temp(self):
        if not self.temp_dir.exists():
            return
        now     = time.time()
        removed = 0
        freed   = 0
        for f in self.temp_dir.iterdir():
            if f.is_file() and (now - f.stat().st_mtime) > self.temp_ttl:
                size = f.stat().st_size
                f.unlink(missing_ok=True)
                removed += 1
                freed   += size
        if removed:
            logger.info(f"Temp cleanup: {removed} files, {freed/1024:.0f}KB freed")

    def _clean_outputs(self):
        if not self.output_dir.exists():
            return
        now         = time.time()
        total_size  = sum(f.stat().st_size for f in self.output_dir.rglob("*") if f.is_file())
        max_bytes   = self.max_output_gb * 1_073_741_824

        if total_size < max_bytes:
            return

        # Delete oldest files first
        files = sorted(
            [f for f in self.output_dir.rglob("*") if f.is_file()],
            key=lambda f: f.stat().st_mtime,
        )
        freed = 0
        for f in files:
            if total_size - freed < max_bytes * 0.75:
                break
            freed += f.stat().st_size
            f.unlink(missing_ok=True)
            logger.info(f"Output evicted: {f.name}")
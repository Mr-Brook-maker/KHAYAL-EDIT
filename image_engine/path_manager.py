"""
Path Manager
Handles auto-generation of output paths, temp files, and cleanup.
"""

import uuid
from pathlib import Path
from typing import Optional
from agent.config import settings


class PathManager:
    """Generates unique, organized output paths for all operations."""

    @staticmethod
    def resolve_output(
        input_path: str,
        output_path: Optional[str],
        suffix: str,
        extension: str
    ) -> Path:
        """
        Resolve final output path.
        
        Strategy:
        - If output_path provided: use it directly
        - Otherwise: outputs/<input_stem>_<suffix>_<uuid4[:8]>.<ext>
        """
        if output_path:
            p = Path(output_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            return p

        stem = Path(input_path).stem
        uid = uuid.uuid4().hex[:8]
        filename = f"{stem}_{suffix}_{uid}.{extension}"
        out = settings.output_dir / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        return out

    @staticmethod
    def temp_path(extension: str) -> Path:
        """Create a guaranteed-unique temp file path."""
        uid = uuid.uuid4().hex
        p = settings.temp_dir / f"tmp_{uid}.{extension}"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
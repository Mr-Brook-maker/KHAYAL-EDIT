"""
Video Path Manager
Handles output path resolution, temp dir lifecycle, and frame sequence dirs.
"""

import uuid
from pathlib import Path
from typing import Optional
import os

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./outputs"))
TEMP_DIR   = Path(os.getenv("TEMP_DIR",   "./temp"))


class VideoPathManager:

    @staticmethod
    def resolve(
        input_path: str,
        output_path: Optional[str],
        suffix: str,
        extension: str,
    ) -> Path:
        if output_path:
            p = Path(output_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            return p
        uid  = uuid.uuid4().hex[:8]
        stem = Path(input_path).stem
        p    = OUTPUT_DIR / f"{stem}_{suffix}_{uid}.{extension}"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    @staticmethod
    def frames_dir(input_path: str, suffix: str = "frames") -> Path:
        uid = uuid.uuid4().hex[:8]
        stem = Path(input_path).stem
        d = OUTPUT_DIR / f"{stem}_{suffix}_{uid}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @staticmethod
    def temp(extension: str) -> Path:
        TEMP_DIR.mkdir(parents=True, exist_ok=True)
        return TEMP_DIR / f"tmp_{uuid.uuid4().hex}.{extension}"
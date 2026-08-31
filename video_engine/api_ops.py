"""
Generative Video Operations via Replicate API
Free tier: limited predictions per month.

Models used:
- Stable Video Diffusion: stability-ai/stable-video-diffusion
- SDXL img2img for frame styling (style transfer pipeline)
"""

import io
import logging
import os
import time
import urllib.request
from pathlib import Path
from typing import Generator

import replicate
import requests
from PIL import Image

from video_engine.schemas import AnimateImageInput, VideoStyleTransferInput
from video_engine.path_manager import VideoPathManager
from video_engine.local_ops import extract_frames, merge_videos
from video_engine.schemas import ExtractFramesInput, MergeVideosInput

logger = logging.getLogger(__name__)

REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN", "")


def _get_replicate_client():
    if not REPLICATE_API_TOKEN:
        raise RuntimeError(
            "REPLICATE_API_TOKEN not set. "
            "Get a free token at https://replicate.com/account"
        )
    os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN
    return replicate


# ─────────────────────────────────────────────────────────────────────────────
# 1. Image → Video (Stable Video Diffusion)
# ─────────────────────────────────────────────────────────────────────────────

def animate_image(params: AnimateImageInput) -> dict:
    """
    Animate a still image using Stable Video Diffusion via Replicate.
    
    Pipeline:
    1. Upload image (base64 data URI)
    2. Submit prediction to SVD model
    3. Poll until complete
    4. Download MP4 output
    """
    client = _get_replicate_client()

    # Resize image to SVD's required 1024x576 or 576x1024
    img = Image.open(params.image_path).convert("RGB")
    img = _fit_to_svd(img)

    buf = io.BytesIO()
    img.save(buf, format="WEBP", quality=90)
    img_bytes = buf.getvalue()

    out = VideoPathManager.resolve(
        params.image_path, params.output_path,
        "animated", "mp4",
    )

    logger.info("Submitting to Stable Video Diffusion (Replicate)...")

    output = client.run(
        "stability-ai/stable-video-diffusion:3f0457e4619daac51203dedb472816fd4af51f3149fa7a9e0b5ffcf1b8172438",
        input={
            "input_image":         f"data:image/webp;base64,{_b64(img_bytes)}",
            "video_length":        "14_frames_with_svd" if params.num_frames <= 14 else "25_frames_with_svd_xt",
            "sizing_strategy":     "crop_to_16_9",
            "frames_per_second":   params.fps,
            "motion_bucket_id":    params.motion_bucket_id,
            "cond_aug":            params.noise_aug_strength,
            "decoding_t":          1,        # Decode 1 frame at a time (memory efficient)
        },
    )

    # Replicate returns a file URL or FileOutput object
    video_url = output[0] if isinstance(output, (list, tuple)) else str(output)
    resp = requests.get(str(video_url), timeout=120)
    resp.raise_for_status()

    with open(str(out), "wb") as f:
        f.write(resp.content)

    logger.info(f"SVD animation saved: {out}")

    return {
        "output_path":     str(out),
        "model":           "stable-video-diffusion",
        "motion_bucket_id": params.motion_bucket_id,
        "fps":             params.fps,
        "num_frames":      params.num_frames,
    }


def _fit_to_svd(img: Image.Image) -> Image.Image:
    """Resize image to nearest SVD-compatible resolution (multiples of 64)."""
    target_w, target_h = 1024, 576
    ratio = img.width / img.height
    if ratio > 1:
        img = img.resize((target_w, target_h), Image.LANCZOS)
    else:
        img = img.resize((target_h, target_w), Image.LANCZOS)
    return img


def _b64(data: bytes) -> str:
    import base64
    return base64.b64encode(data).decode()


# ─────────────────────────────────────────────────────────────────────────────
# 2. Video Style Transfer
# ─────────────────────────────────────────────────────────────────────────────

def video_style_transfer(params: VideoStyleTransferInput) -> dict:
    """
    Frame-by-frame style transfer using SDXL img2img (Replicate).
    
    Pipeline:
    1. Extract N frames from video
    2. Style-transfer each frame via Replicate SDXL
    3. Re-encode frames back to video at original fps
    
    Note: Temporal consistency is approximate. For production use,
    AnimateDiff or ControlNet+temporal would be superior but are heavier.
    """
    client = _get_replicate_client()

    # Step 1: Extract frames
    from video_engine.local_ops import _probe
    meta = _probe(params.video_path)
    fps  = meta.get("fps", 24)

    frames_result = extract_frames(ExtractFramesInput(
        video_path=params.video_path,
        fps=params.max_frames / max(meta.get("duration", 1), 1),
        max_frames=params.max_frames,
    ))

    frame_paths = sorted(
        Path(frames_result["output_dir"]).glob("*.png")
    )[:params.max_frames]

    logger.info(f"Style transfer: processing {len(frame_paths)} frames...")
    styled_paths = []

    for i, frame_path in enumerate(frame_paths):
        logger.info(f"  Frame {i+1}/{len(frame_paths)}: {params.style_prompt}")

        img = Image.open(frame_path).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="WEBP", quality=85)

        output = client.run(
            "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
            input={
                "prompt":            params.style_prompt,
                "negative_prompt":   "blurry, artifacts, distorted, worst quality",
                "image":             f"data:image/webp;base64,{_b64(buf.getvalue())}",
                "strength":          params.strength,
                "num_inference_steps": 20,
                "guidance_scale":    7.5,
            },
        )

        styled_url = output[0] if isinstance(output, (list, tuple)) else str(output)
        resp = requests.get(str(styled_url), timeout=60)
        styled_frame = VideoPathManager.temp("png")
        styled_frame.write_bytes(resp.content)
        styled_paths.append(str(styled_frame))

        # Rate limit respect (Replicate free tier)
        if i < len(frame_paths) - 1:
            time.sleep(1.5)

    # Step 3: Re-encode frames → video
    out = VideoPathManager.resolve(
        params.video_path, params.output_path,
        "styled", "mp4",
    )

    # Use ffmpeg to encode frame sequence
    import ffmpeg as ffmpeg_module
    pattern = str(Path(styled_paths[0]).parent / "tmp_*.png")

    (
        ffmpeg_module
        .input(pattern, pattern_type="glob", framerate=fps)
        .output(str(out), vcodec="libx264", pix_fmt="yuv420p", crf=18, preset="fast")
        .run(quiet=True, overwrite_output=True)
    )

    # Cleanup temp styled frames
    for p in styled_paths:
        Path(p).unlink(missing_ok=True)

    return {
        "output_path":   str(out),
        "style_prompt":  params.style_prompt,
        "frames_styled": len(frame_paths),
        "strength":      params.strength,
    }
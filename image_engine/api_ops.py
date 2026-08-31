"""
Generative API Operations
Handles: HuggingFace Inference API, Fal.ai
For: Inpainting, Outpainting, Object Removal (AI generation tasks)

Design decisions:
- HuggingFace: Free tier, no credit card, rate-limited
- Fal.ai: Free credits on signup, faster inference
- Both backends implement the same interface for easy swapping
"""

import io
import time
import logging
import base64
import requests
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image
import numpy as np

from agent.config import settings
from agent.tool_schemas import InpaintImageInput, OutpaintImageInput, ObjectRemovalInput
from image_engine.result import ProcessingResult
from image_engine.path_manager import PathManager
from image_engine.local_ops import _load_pil, _save_pil, _pil_to_cv2

logger = logging.getLogger(__name__)


# ── HuggingFace Client ────────────────────────────────────────────────────────

class HuggingFaceClient:
    """
    HuggingFace Inference API client.
    Free tier: limited RPM, suitable for low-volume usage.
    Models auto-loaded on first request (cold start ~30s).
    """

    BASE_URL = "https://api-inference.huggingface.co/models"

    # Best free inpainting model on HF Inference API
    INPAINT_MODEL = "runwayml/stable-diffusion-inpainting"

    def __init__(self):
        self.headers = {"Authorization": f"Bearer {settings.hf_api_token}"}
        self.timeout = 120   # Models may need cold start time

    def _pil_to_b64(self, img: Image.Image, format: str = "PNG") -> str:
        buf = io.BytesIO()
        img.save(buf, format=format)
        return base64.b64encode(buf.getvalue()).decode()

    def _b64_to_pil(self, data: str) -> Image.Image:
        return Image.open(io.BytesIO(base64.b64decode(data)))

    def inpaint(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        negative_prompt: str,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        retries: int = 3,
    ) -> Image.Image:
        """
        Call SD inpainting model.
        Handles model loading wait (503 → retry with backoff).
        """
        # Resize to model-supported size (SD requires multiples of 64)
        size = self._snap_to_64(image.size)
        image = image.resize(size, Image.LANCZOS)
        mask = mask.resize(size, Image.LANCZOS).convert("L")

        # Convert mask: SD inpainting expects white=inpaint
        mask_arr = np.array(mask)
        mask_arr = (mask_arr > 127).astype(np.uint8) * 255
        mask_clean = Image.fromarray(mask_arr)

        payload = {
            "inputs": {
                "prompt": prompt or "clean seamless background",
                "negative_prompt": negative_prompt,
                "image": self._pil_to_b64(image),
                "mask_image": self._pil_to_b64(mask_clean),
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
            }
        }

        url = f"{self.BASE_URL}/{self.INPAINT_MODEL}"

        for attempt in range(retries):
            try:
                resp = requests.post(url, headers=self.headers, json=payload, timeout=self.timeout)

                if resp.status_code == 503:
                    # Model loading — wait and retry
                    wait = resp.json().get("estimated_time", 20)
                    logger.info(f"HF model loading, waiting {wait:.0f}s... (attempt {attempt+1}/{retries})")
                    time.sleep(min(wait + 5, 60))
                    continue

                resp.raise_for_status()

                # Response is raw image bytes for image-to-image models
                result_img = Image.open(io.BytesIO(resp.content))
                return result_img.resize(image.size, Image.LANCZOS)

            except requests.RequestException as e:
                if attempt == retries - 1:
                    raise RuntimeError(f"HuggingFace API failed after {retries} attempts: {e}")
                time.sleep(5 * (attempt + 1))

    @staticmethod
    def _snap_to_64(size: Tuple[int, int]) -> Tuple[int, int]:
        """Round dimensions to nearest multiple of 64 (SD requirement)."""
        w, h = size
        max_dim = 512    # Keep within HF free tier limits
        # Scale down if too large
        if w > max_dim or h > max_dim:
            ratio = min(max_dim / w, max_dim / h)
            w, h = int(w * ratio), int(h * ratio)
        return (max(64, (w // 64) * 64), max(64, (h // 64) * 64))


# ── Fal.ai Client ─────────────────────────────────────────────────────────────

class FalClient:
    """
    Fal.ai inference client.
    Faster than HF, better quality models available.
    Requires free account + API key from fal.ai.
    """

    def __init__(self):
        try:
            import fal_client
            self._client = fal_client
            self._client.api_key = settings.fal_api_key
        except ImportError:
            raise ImportError("fal-client not installed. Run: pip install fal-client")

    def inpaint(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        negative_prompt: str,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
    ) -> Image.Image:
        """
        Use Fal.ai SDXL Inpainting — higher quality, faster.
        Model: fal-ai/stable-diffusion-xl-inpainting
        """
        import tempfile, os

        def pil_to_data_url(img: Image.Image, fmt="PNG") -> str:
            buf = io.BytesIO()
            img.save(buf, format=fmt)
            b64 = base64.b64encode(buf.getvalue()).decode()
            mime = "image/png" if fmt == "PNG" else "image/jpeg"
            return f"data:{mime};base64,{b64}"

        result = self._client.subscribe(
            "fal-ai/stable-diffusion-xl-inpainting",
            arguments={
                "prompt": prompt or "clean seamless background, photorealistic",
                "negative_prompt": negative_prompt,
                "image_url": pil_to_data_url(image),
                "mask_url": pil_to_data_url(mask.convert("L")),
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "strength": 0.99,
            },
            with_logs=False,
        )

        image_url = result["images"][0]["url"]
        resp = requests.get(image_url, timeout=30)
        return Image.open(io.BytesIO(resp.content))


# ── Outpaint Helper ───────────────────────────────────────────────────────────

def _prepare_outpaint_canvas(
    image: Image.Image,
    expand_left: int,
    expand_right: int,
    expand_top: int,
    expand_bottom: int,
) -> Tuple[Image.Image, Image.Image, Tuple[int, int]]:
    """
    Create expanded canvas and corresponding inpaint mask.
    
    Returns:
        (padded_image, mask, original_offset)
        mask: white = region to generate, black = original image kept
    """
    orig_w, orig_h = image.size
    new_w = orig_w + expand_left + expand_right
    new_h = orig_h + expand_top + expand_bottom

    # Canvas with original image centered
    canvas = Image.new("RGB", (new_w, new_h), (128, 128, 128))
    offset = (expand_left, expand_top)
    canvas.paste(image, offset)

    # Mask: white everywhere EXCEPT the original image area
    mask = Image.new("L", (new_w, new_h), 255)  # All white (generate)
    original_region = Image.new("L", (orig_w, orig_h), 0)   # Black (keep)
    mask.paste(original_region, offset)

    return canvas, mask, offset


# ── Object Detection for Object Removal ──────────────────────────────────────

def _detect_object_mask(
    image: Image.Image,
    object_description: str,
    confidence_threshold: float = 0.5,
) -> Optional[Image.Image]:
    """
    Use HuggingFace zero-shot object detection (OWL-ViT) to find an object
    and generate a binary mask.
    
    Returns mask image or None if object not found.
    """
    try:
        from transformers import pipeline

        detector = pipeline(
            "zero-shot-object-detection",
            model="google/owlvit-base-patch32",  # Free, runs locally
        )

        results = detector(image, candidate_labels=[object_description])

        if not results or results[0]["score"] < confidence_threshold:
            logger.warning(f"Object '{object_description}' not found (confidence too low)")
            return None

        # Take highest-confidence detection
        best = max(results, key=lambda x: x["score"])
        box = best["box"]   # {xmin, ymin, xmax, ymax}

        logger.info(
            f"Detected '{object_description}' at {box} "
            f"(confidence: {best['score']:.2%})"
        )

        # Generate mask from bounding box + dilation
        mask = Image.new("L", image.size, 0)
        import PIL.ImageDraw as ImageDraw
        draw = ImageDraw.Draw(mask)

        # Expand box slightly for cleaner inpainting
        padding = 15
        draw.rectangle([
            max(0, box["xmin"] - padding),
            max(0, box["ymin"] - padding),
            min(image.width, box["xmax"] + padding),
            min(image.height, box["ymax"] + padding),
        ], fill=255)

        # Optional: dilate mask for smoother edges
        import cv2
        mask_arr = np.array(mask)
        kernel = np.ones((20, 20), np.uint8)
        mask_arr = cv2.dilate(mask_arr, kernel, iterations=1)

        return Image.fromarray(mask_arr)

    except Exception as e:
        logger.error(f"Object detection failed: {e}")
        raise


# ── Main API Operation Functions ──────────────────────────────────────────────

def inpaint_image_api(params: InpaintImageInput) -> ProcessingResult:
    """
    AI inpainting via HuggingFace or Fal.ai.
    Handles image/mask loading, client selection, and result saving.
    """
    image = _load_pil(params.image_path).convert("RGB")
    mask = _load_pil(params.mask_path).convert("L")

    out_path = PathManager.resolve_output(
        params.image_path, params.output_path, "inpainted", "png"
    )

    if params.backend == "huggingface":
        client = HuggingFaceClient()
    else:
        client = FalClient()

    result_img = client.inpaint(
        image=image,
        mask=mask,
        prompt=params.prompt,
        negative_prompt=params.negative_prompt,
        num_inference_steps=params.num_inference_steps,
        guidance_scale=params.guidance_scale,
    )

    _save_pil(result_img, out_path, "png")

    return ProcessingResult(output_path=str(out_path)).with_meta(
        backend=params.backend,
        prompt=params.prompt,
        model=(
            HuggingFaceClient.INPAINT_MODEL
            if params.backend == "huggingface"
            else "fal-ai/sdxl-inpainting"
        ),
    )


def outpaint_image_api(params: OutpaintImageInput) -> ProcessingResult:
    """
    AI-powered canvas expansion using inpainting on padded canvas.
    """
    image = _load_pil(params.image_path).convert("RGB")

    canvas, mask, offset = _prepare_outpaint_canvas(
        image,
        expand_left=params.expand_left,
        expand_right=params.expand_right,
        expand_top=params.expand_top,
        expand_bottom=params.expand_bottom,
    )

    out_path = PathManager.resolve_output(
        params.image_path, params.output_path, "outpainted", "png"
    )

    inpaint_params = InpaintImageInput(
        image_path=str(PathManager.temp_path("png")),   # Temp (not used directly)
        mask_path="",    # Not used directly
        prompt=params.prompt or "seamless natural extension of the scene",
        negative_prompt="artifacts, seams, distortion, blurry edges",
        backend=params.backend,
        output_path=str(out_path),
    )

    if params.backend == "huggingface":
        client = HuggingFaceClient()
    else:
        client = FalClient()

    result_img = client.inpaint(
        image=canvas,
        mask=mask,
        prompt=inpaint_params.prompt,
        negative_prompt=inpaint_params.negative_prompt,
    )

    _save_pil(result_img, out_path, "png")

    return ProcessingResult(output_path=str(out_path)).with_meta(
        expansion={
            "left": params.expand_left, "right": params.expand_right,
            "top": params.expand_top, "bottom": params.expand_bottom,
        },
        new_size=f"{canvas.width}x{canvas.height}",
        backend=params.backend,
    )


def remove_object_api(params: ObjectRemovalInput) -> ProcessingResult:
    """
    Full object removal pipeline:
    1. OWL-ViT zero-shot detection → bounding box
    2. Box → dilated binary mask
    3. Mask → SD inpainting → clean result
    """
    image = _load_pil(params.image_path).convert("RGB")

    # Step 1: Detect object
    mask = _detect_object_mask(image, params.object_description, params.detection_confidence)
    if mask is None:
        raise ValueError(
            f"Could not detect '{params.object_description}' in the image. "
            f"Try a more specific description or lower detection_confidence."
        )

    # Save mask for debugging/transparency
    mask_path = PathManager.temp_path("png")
    mask.save(str(mask_path))

    out_path = PathManager.resolve_output(
        params.image_path, params.output_path, "obj_removed", "png"
    )

    # Step 2: Inpaint the detected region
    client = HuggingFaceClient()
    result_img = client.inpaint(
        image=image,
        mask=mask,
        prompt=params.inpaint_prompt,
        negative_prompt="artifacts, distortion, blurry, inconsistent lighting",
    )

    _save_pil(result_img, out_path, "png")

    return ProcessingResult(output_path=str(out_path)).with_meta(
        object_removed=params.object_description,
        mask_path=str(mask_path),
        inpaint_prompt=params.inpaint_prompt,
    )
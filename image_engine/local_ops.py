"""
Local Image Operations
Zero API cost. Uses: rembg, OpenCV, Pillow, numpy.
These run entirely on local hardware — no network required.
"""

import logging
import numpy as np
from pathlib import Path
from typing import Tuple

import cv2
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageChops
from PIL.ImageColor import getrgb
import colorsys

from agent.tool_schemas import (
    RemoveBackgroundInput, ResizeImageInput, AdjustColorsInput,
    ApplyFilterInput, ConvertFormatInput, ResizeMode, ImageFormat
)
from image_engine.result import ProcessingResult
from image_engine.path_manager import PathManager

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_pil(path: str) -> Image.Image:
    """Load image, auto-rotate based on EXIF, return PIL Image."""
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)   # Respect camera orientation
    return img


def _save_pil(img: Image.Image, path: Path, file_format: str, quality: int = 95):
    """Save PIL image with format-appropriate options."""
    file_format = file_format.upper()
    if file_format == "JPG":
        file_format = "JPEG"

    save_kwargs = {}
    if file_format == "JPEG":
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGB")          # JPEG has no alpha
        save_kwargs = {"quality": quality, "optimize": True, "progressive": True}
    elif file_format == "WEBP":
        save_kwargs = {"quality": quality, "method": 6}
    elif file_format == "PNG":
        save_kwargs = {"optimize": True, "compress_level": 6}

    img.save(str(path), format=file_format, **save_kwargs)
    logger.debug(f"Saved {file_format} → {path} ({path.stat().st_size / 1024:.1f} KB)")


def _pil_to_cv2(img: Image.Image) -> np.ndarray:
    """PIL RGB/RGBA → OpenCV BGR/BGRA."""
    arr = np.array(img)
    if arr.ndim == 3 and arr.shape[2] == 4:
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
    elif arr.ndim == 3:
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return arr


def _cv2_to_pil(arr: np.ndarray) -> Image.Image:
    """OpenCV BGR/BGRA → PIL RGB/RGBA."""
    if arr.ndim == 3 and arr.shape[2] == 4:
        return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGRA2RGBA))
    elif arr.ndim == 3:
        return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
    return Image.fromarray(arr)


# ── 1. Background Removal ─────────────────────────────────────────────────────

def remove_background_local(params: RemoveBackgroundInput) -> ProcessingResult:
    """
    Background removal using rembg (ONNX U2Net model).
    Free, local, no API calls. Handles alpha matting for fine edges.
    
    Model selection:
    - u2net: General purpose (default)
    - u2net_human_seg: Optimized for portraits
    - silueta: Faster, smaller model
    """
    from rembg import remove, new_session

    out_path = PathManager.resolve_output(
        params.image_path, params.output_path, "nobg", "png"
    )

    # Choose model based on alpha matting flag
    # u2net_human_seg gives better results for portraits
    session = new_session("u2net")

    with open(params.image_path, "rb") as f:
        input_bytes = f.read()

    result_bytes = remove(
        input_bytes,
        session=session,
        alpha_matting=params.alpha_matting,
        alpha_matting_foreground_threshold=240,
        alpha_matting_background_threshold=10,
        alpha_matting_erode_size=10,
        post_process_mask=params.post_process,
    )

    # Write result directly (rembg returns PNG bytes)
    with open(str(out_path), "wb") as f:
        f.write(result_bytes)

    # Gather metadata
    result_img = Image.open(str(out_path))
    orig_img = _load_pil(params.image_path)

    return ProcessingResult(output_path=str(out_path)).with_meta(
        original_size=f"{orig_img.width}x{orig_img.height}",
        output_size=f"{result_img.width}x{result_img.height}",
        has_alpha=result_img.mode == "RGBA",
        model="u2net",
    )


# ── 2. Resize ─────────────────────────────────────────────────────────────────

def resize_image_local(params: ResizeImageInput) -> ProcessingResult:
    """
    High-quality image resize with Lanczos resampling.
    
    Modes:
    - FIT: Scale to fit within target, pad with background color
    - FILL: Scale to fill target completely, center-crop excess
    - STRETCH: Force to exact dimensions (may distort)
    - THUMBNAIL: Fit within target, no padding (standard thumbnail)
    """
    img = _load_pil(params.image_path)
    target = (params.width, params.height)
    ext = params.output_format.value

    out_path = PathManager.resolve_output(
        params.image_path, params.output_path,
        f"resize_{params.width}x{params.height}", ext
    )

    if params.mode == ResizeMode.STRETCH:
        resized = img.resize(target, Image.LANCZOS)

    elif params.mode == ResizeMode.THUMBNAIL:
        resized = img.copy()
        resized.thumbnail(target, Image.LANCZOS)

    elif params.mode == ResizeMode.FIT:
        # Scale to fit, then center on canvas with background
        img.thumbnail(target, Image.LANCZOS)

        if params.background_color.lower() == "transparent":
            canvas = Image.new("RGBA", target, (0, 0, 0, 0))
        else:
            rgb = getrgb(params.background_color)
            canvas = Image.new("RGBA", target, rgb + (255,))

        offset = (
            (target[0] - img.width) // 2,
            (target[1] - img.height) // 2,
        )
        if img.mode == "RGBA":
            canvas.paste(img, offset, img)
        else:
            canvas.paste(img, offset)
        resized = canvas

    elif params.mode == ResizeMode.FILL:
        # Scale to fill, then center-crop
        src_ratio = img.width / img.height
        tgt_ratio = params.width / params.height

        if src_ratio > tgt_ratio:
            # Source is wider — match height, crop width
            new_h = params.height
            new_w = int(new_h * src_ratio)
        else:
            # Source is taller — match width, crop height
            new_w = params.width
            new_h = int(new_w / src_ratio)

        img = img.resize((new_w, new_h), Image.LANCZOS)
        left = (new_w - params.width) // 2
        top = (new_h - params.height) // 2
        resized = img.crop((left, top, left + params.width, top + params.height))

    else:
        resized = img.resize(target, Image.LANCZOS)

    _save_pil(resized, out_path, ext, params.quality)

    return ProcessingResult(output_path=str(out_path)).with_meta(
        original_size=f"{img.width}x{img.height}",
        output_size=f"{resized.width}x{resized.height}",
        mode=params.mode.value,
        format=ext,
    )


# ── 3. Color Adjustment ───────────────────────────────────────────────────────

def adjust_colors_local(params: AdjustColorsInput) -> ProcessingResult:
    """
    Full color grading pipeline using Pillow enhancers + numpy/OpenCV for
    hue shift and gamma correction (not natively supported by Pillow).
    """
    img = _load_pil(params.image_path)
    original_mode = img.mode

    # Keep alpha separate during processing
    alpha = None
    if img.mode == "RGBA":
        r, g, b, alpha = img.split()
        img = Image.merge("RGB", (r, g, b))
    elif img.mode != "RGB":
        img = img.convert("RGB")

    # 1. Brightness
    if params.brightness != 1.0:
        img = ImageEnhance.Brightness(img).enhance(params.brightness)

    # 2. Contrast
    if params.contrast != 1.0:
        img = ImageEnhance.Contrast(img).enhance(params.contrast)

    # 3. Saturation
    if params.saturation != 1.0:
        img = ImageEnhance.Color(img).enhance(params.saturation)

    # 4. Sharpness
    if params.sharpness != 1.0:
        img = ImageEnhance.Sharpness(img).enhance(params.sharpness)

    # 5. Hue Shift (via HSV in OpenCV — faster than pure numpy)
    if params.hue_shift != 0:
        arr = _pil_to_cv2(img)
        hsv = cv2.cvtColor(arr, cv2.COLOR_BGR2HSV).astype(np.int32)
        hsv[:, :, 0] = (hsv[:, :, 0] + params.hue_shift // 2) % 180
        hsv = hsv.astype(np.uint8)
        arr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        img = _cv2_to_pil(arr)

    # 6. Gamma Correction (via lookup table — very fast O(1) per pixel)
    if params.gamma != 1.0:
        inv_gamma = 1.0 / params.gamma
        table = np.array([
            ((i / 255.0) ** inv_gamma) * 255
            for i in range(256)
        ], dtype=np.uint8)
        arr = np.array(img)
        img = Image.fromarray(cv2.LUT(arr, table))

    # Restore alpha channel
    if alpha is not None:
        img = img.convert("RGBA")
        img.putalpha(alpha)

    out_path = PathManager.resolve_output(
        params.image_path, params.output_path, "adjusted", "png"
    )
    _save_pil(img, out_path, "png")

    return ProcessingResult(output_path=str(out_path)).with_meta(
        adjustments={
            "brightness": params.brightness,
            "contrast": params.contrast,
            "saturation": params.saturation,
            "sharpness": params.sharpness,
            "hue_shift": params.hue_shift,
            "gamma": params.gamma,
        }
    )


# ── 4. Artistic Filters ───────────────────────────────────────────────────────

class FilterLibrary:
    """
    Collection of LUT-based and pixel-manipulation filters.
    Each filter is a pure function: (PIL.Image, float) → PIL.Image
    """

    @staticmethod
    def _blend(original: Image.Image, filtered: Image.Image, intensity: float) -> Image.Image:
        """Blend original and filtered based on intensity."""
        return Image.blend(original, filtered, intensity)

    @staticmethod
    def vintage(img: Image.Image, intensity: float) -> Image.Image:
        """Warm tones, reduced saturation, slight vignette."""
        # Warm color cast via channel curves
        r, g, b = img.split()
        r = r.point(lambda x: min(255, int(x * 1.1 + 15)))   # Boost red
        g = g.point(lambda x: min(255, int(x * 0.95 + 5)))   # Slight green boost
        b = b.point(lambda x: max(0, int(x * 0.85 - 10)))    # Reduce blue

        filtered = Image.merge("RGB", (r, g, b))

        # Add slight fade (lift blacks)
        filtered = ImageEnhance.Contrast(filtered).enhance(0.85)
        filtered = filtered.point(lambda x: x * 0.88 + 30)   # Lift shadows

        return FilterLibrary._blend(img, filtered, intensity)

    @staticmethod
    def noir(img: Image.Image, intensity: float) -> Image.Image:
        """High-contrast black and white."""
        gray = ImageOps.grayscale(img).convert("RGB")
        gray = ImageEnhance.Contrast(gray).enhance(1.4)
        gray = ImageEnhance.Brightness(gray).enhance(0.9)
        return FilterLibrary._blend(img, gray, intensity)

    @staticmethod
    def cinematic(img: Image.Image, intensity: float) -> Image.Image:
        """Teal shadows, orange highlights (Hollywood color grade)."""
        arr = np.array(img).astype(np.float32)

        # Teal in shadows: push dark pixels toward teal (0, 128, 128)
        shadows_mask = (arr.mean(axis=2, keepdims=True) / 255.0) ** 2
        teal = np.array([0, 50, 60], dtype=np.float32)
        arr += teal * (1 - shadows_mask) * 0.3

        # Orange in highlights: push bright pixels toward orange (255, 140, 0)
        highlights_mask = (arr.mean(axis=2, keepdims=True) / 255.0) ** 2
        orange = np.array([20, -10, -30], dtype=np.float32)
        arr += orange * highlights_mask * 0.25

        arr = np.clip(arr, 0, 255).astype(np.uint8)
        filtered = Image.fromarray(arr)
        filtered = ImageEnhance.Contrast(filtered).enhance(1.1)

        return FilterLibrary._blend(img, filtered, intensity)

    @staticmethod
    def matte(img: Image.Image, intensity: float) -> Image.Image:
        """Faded, desaturated matte look — lifts blacks."""
        filtered = ImageEnhance.Color(img).enhance(0.7)
        filtered = filtered.point(lambda x: x * 0.8 + 40)
        return FilterLibrary._blend(img, filtered, intensity)

    @staticmethod
    def vivid(img: Image.Image, intensity: float) -> Image.Image:
        """Boosted saturation and contrast."""
        filtered = ImageEnhance.Color(img).enhance(1.6)
        filtered = ImageEnhance.Contrast(filtered).enhance(1.2)
        return FilterLibrary._blend(img, filtered, intensity)

    @staticmethod
    def cool(img: Image.Image, intensity: float) -> Image.Image:
        """Blue-tinted cool temperature."""
        r, g, b = img.split()
        r = r.point(lambda x: max(0, int(x * 0.9)))
        b = b.point(lambda x: min(255, int(x * 1.15 + 10)))
        filtered = Image.merge("RGB", (r, g, b))
        return FilterLibrary._blend(img, filtered, intensity)

    @staticmethod
    def warm(img: Image.Image, intensity: float) -> Image.Image:
        """Orange-tinted warm temperature."""
        r, g, b = img.split()
        r = r.point(lambda x: min(255, int(x * 1.15 + 10)))
        b = b.point(lambda x: max(0, int(x * 0.88)))
        filtered = Image.merge("RGB", (r, g, b))
        return FilterLibrary._blend(img, filtered, intensity)

    @staticmethod
    def faded(img: Image.Image, intensity: float) -> Image.Image:
        """Washed-out, low-contrast faded look."""
        filtered = img.point(lambda x: x * 0.75 + 50)
        filtered = ImageEnhance.Color(filtered).enhance(0.65)
        return FilterLibrary._blend(img, filtered, intensity)

    @staticmethod
    def cross_process(img: Image.Image, intensity: float) -> Image.Image:
        """Harsh cross-processing effect: inverted channel curves."""
        r, g, b = img.split()
        r = r.point(lambda x: min(255, int(x * 1.3) if x < 128 else max(0, int(x * 0.7 + 80))))
        g = g.point(lambda x: max(0, int(x * 0.8)) if x < 64 else min(255, int(x * 1.1)))
        b = b.point(lambda x: min(255, int(x * 1.4) if x > 100 else max(0, int(x * 0.6))))
        filtered = Image.merge("RGB", (r, g, b))
        filtered = ImageEnhance.Saturation(filtered).enhance(1.3)
        return FilterLibrary._blend(img, filtered, intensity)

    @staticmethod
    def duotone(img: Image.Image, intensity: float) -> Image.Image:
        """Two-color tone mapping: shadows to deep blue, highlights to gold."""
        gray = ImageOps.grayscale(img)
        arr = np.array(gray, dtype=np.float32) / 255.0

        shadow_color = np.array([20, 30, 80], dtype=np.float32)    # Deep blue
        highlight_color = np.array([255, 200, 50], dtype=np.float32)  # Gold

        result = np.zeros((*arr.shape, 3), dtype=np.float32)
        for c in range(3):
            result[:, :, c] = (
                shadow_color[c] * (1 - arr) + highlight_color[c] * arr
            )

        filtered = Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))
        return FilterLibrary._blend(img.convert("RGB"), filtered, intensity)


_FILTER_MAP = {
    "vintage": FilterLibrary.vintage,
    "noir": FilterLibrary.noir,
    "cinematic": FilterLibrary.cinematic,
    "matte": FilterLibrary.matte,
    "vivid": FilterLibrary.vivid,
    "cool": FilterLibrary.cool,
    "warm": FilterLibrary.warm,
    "faded": FilterLibrary.faded,
    "cross_process": FilterLibrary.cross_process,
    "duotone": FilterLibrary.duotone,
}


def apply_filter_local(params: ApplyFilterInput) -> ProcessingResult:
    img = _load_pil(params.image_path).convert("RGB")

    filter_fn = _FILTER_MAP.get(params.filter_type)
    if not filter_fn:
        raise ValueError(f"Unknown filter: {params.filter_type}")

    filtered = filter_fn(img, params.intensity)

    out_path = PathManager.resolve_output(
        params.image_path, params.output_path,
        f"filter_{params.filter_type}", "jpg"
    )
    _save_pil(filtered, out_path, "jpeg", quality=92)

    return ProcessingResult(output_path=str(out_path)).with_meta(
        filter=params.filter_type,
        intensity=params.intensity,
    )


# ── 5. Format Conversion ──────────────────────────────────────────────────────

def convert_format_local(params: ConvertFormatInput) -> ProcessingResult:
    img = _load_pil(params.image_path)

    if params.strip_metadata:
        # Re-create image from raw pixel data to strip EXIF
        data = list(img.getdata())
        clean = Image.new(img.mode, img.size)
        clean.putdata(data)
        img = clean

    ext = params.target_format.value
    out_path = PathManager.resolve_output(
        params.image_path, params.output_path, "converted", ext
    )
    _save_pil(img, out_path, ext, params.quality)

    return ProcessingResult(output_path=str(out_path)).with_meta(
        source_format=Path(params.image_path).suffix.lstrip(".").upper(),
        target_format=ext.upper(),
        metadata_stripped=params.strip_metadata,
    )
"""
Tool Schema Definitions
All tools follow OpenAI Function Calling spec (LangChain compatible).
Each schema is typed with Pydantic for runtime validation.
"""

from pydantic import BaseModel, Field
from typing import Optional, Literal
from enum import Enum


# ── Shared Types ──────────────────────────────────────────────────────────────

class ImageFormat(str, Enum):
    PNG = "png"
    JPEG = "jpeg"
    WEBP = "webp"


class ResizeMode(str, Enum):
    STRETCH = "stretch"
    FIT = "fit"           # Maintain aspect ratio, pad if needed
    FILL = "fill"         # Crop to fill target dimensions
    THUMBNAIL = "thumbnail"


# ── Image Tool Input Schemas ──────────────────────────────────────────────────

class RemoveBackgroundInput(BaseModel):
    """Remove background from an image, returning subject on transparent PNG."""
    image_path: str = Field(description="Absolute or relative path to the source image")
    output_path: Optional[str] = Field(
        default=None,
        description="Destination path. Auto-generated if not provided."
    )
    alpha_matting: bool = Field(
        default=True,
        description="Use alpha matting for fine edge detail (hair, fur). Slower but higher quality."
    )
    post_process: bool = Field(
        default=True,
        description="Apply morphological cleanup to remove alpha artifacts."
    )


class ResizeImageInput(BaseModel):
    """Resize an image to target dimensions with configurable behavior."""
    image_path: str = Field(description="Path to source image")
    width: int = Field(ge=1, le=8192, description="Target width in pixels")
    height: int = Field(ge=1, le=8192, description="Target height in pixels")
    mode: ResizeMode = Field(default=ResizeMode.FIT)
    background_color: str = Field(
        default="#FFFFFF",
        description="Hex color for padding when mode=FIT. Use 'transparent' for PNG."
    )
    output_path: Optional[str] = None
    output_format: ImageFormat = ImageFormat.PNG
    quality: int = Field(default=95, ge=1, le=100, description="JPEG/WEBP quality")


class AdjustColorsInput(BaseModel):
    """Apply color grading adjustments to an image."""
    image_path: str
    brightness: float = Field(default=1.0, ge=0.1, le=3.0, description="1.0 = no change")
    contrast: float = Field(default=1.0, ge=0.1, le=3.0)
    saturation: float = Field(default=1.0, ge=0.0, le=3.0, description="0.0 = grayscale")
    sharpness: float = Field(default=1.0, ge=0.0, le=5.0)
    hue_shift: int = Field(default=0, ge=-180, le=180, description="Hue rotation in degrees")
    gamma: float = Field(default=1.0, ge=0.1, le=3.0)
    output_path: Optional[str] = None


class InpaintImageInput(BaseModel):
    """
    Remove or replace objects using AI inpainting.
    Requires a mask image where white = area to inpaint.
    """
    image_path: str = Field(description="Source image path")
    mask_path: str = Field(description="Binary mask: white=inpaint, black=keep")
    prompt: str = Field(
        default="",
        description="Text guidance for inpainted content. Empty = erase/remove."
    )
    negative_prompt: str = Field(
        default="artifacts, blurry, distorted",
        description="What to avoid in generated content"
    )
    backend: Literal["huggingface", "fal"] = Field(
        default="huggingface",
        description="Which API backend to use for generation"
    )
    num_inference_steps: int = Field(default=30, ge=10, le=100)
    guidance_scale: float = Field(default=7.5, ge=1.0, le=20.0)
    output_path: Optional[str] = None


class OutpaintImageInput(BaseModel):
    """Expand image canvas in any direction using AI generation."""
    image_path: str
    expand_left: int = Field(default=0, ge=0, description="Pixels to add on left")
    expand_right: int = Field(default=0, ge=0)
    expand_top: int = Field(default=0, ge=0)
    expand_bottom: int = Field(default=0, ge=0)
    prompt: str = Field(default="", description="Guidance for expanded content")
    backend: Literal["huggingface", "fal"] = "huggingface"
    output_path: Optional[str] = None


class ObjectRemovalInput(BaseModel):
    """Detect and remove a named object from an image automatically."""
    image_path: str
    object_description: str = Field(
        description="Natural language description of object to remove (e.g., 'the red car')"
    )
    detection_confidence: float = Field(default=0.5, ge=0.1, le=1.0)
    inpaint_prompt: str = Field(
        default="clean background, seamless",
        description="What to fill the removed area with"
    )
    output_path: Optional[str] = None


class ApplyFilterInput(BaseModel):
    """Apply artistic filters and effects to an image."""
    image_path: str
    filter_type: Literal[
        "vintage", "noir", "cinematic", "matte", "vivid",
        "cool", "warm", "faded", "cross_process", "duotone"
    ]
    intensity: float = Field(default=0.8, ge=0.0, le=1.0)
    output_path: Optional[str] = None


class ConvertFormatInput(BaseModel):
    """Convert image between formats with quality control."""
    image_path: str
    target_format: ImageFormat
    quality: int = Field(default=95, ge=1, le=100)
    strip_metadata: bool = Field(default=False)
    output_path: Optional[str] = None
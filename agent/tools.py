"""
LangChain Tool Wrappers
Bridges tool schemas → image engine functions with full error handling,
structured logging, and output normalization.
"""

import json
import logging
import traceback
from pathlib import Path
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import ValidationError

from agent.tool_schemas import (
    RemoveBackgroundInput, ResizeImageInput, AdjustColorsInput,
    InpaintImageInput, OutpaintImageInput, ObjectRemovalInput,
    ApplyFilterInput, ConvertFormatInput
)
# Image engine imported in Part 2
from image_engine.processor import ImageProcessor

logger = logging.getLogger(__name__)


# ── Tool Response Builder ──────────────────────────────────────────────────────

def _success(output_path: str, message: str, metadata: dict = None) -> str:
    """Standardized success response consumed by the agent for chaining."""
    response = {
        "status": "success",
        "output_path": output_path,
        "message": message,
        "metadata": metadata or {}
    }
    return json.dumps(response)


def _error(tool_name: str, error: Exception, input_data: Any = None) -> str:
    response = {
        "status": "error",
        "tool": tool_name,
        "error_type": type(error).__name__,
        "message": str(error),
        "input": str(input_data)[:500]  # Truncate for token safety
    }
    logger.error(f"Tool '{tool_name}' failed: {error}\n{traceback.format_exc()}")
    return json.dumps(response)


# ── Tool Implementation Functions ─────────────────────────────────────────────

processor = ImageProcessor()   # Singleton — initialized once


def remove_background(
    image_path: str,
    output_path: str = None,
    alpha_matting: bool = True,
    post_process: bool = True
) -> str:
    """
    Remove background from image. Best for portraits, products, objects.
    Returns path to PNG with transparent background.
    """
    try:
        params = RemoveBackgroundInput(
            image_path=image_path,
            output_path=output_path,
            alpha_matting=alpha_matting,
            post_process=post_process
        )
        result = processor.remove_background(params)
        return _success(result.output_path, f"Background removed. Output: {result.output_path}", result.metadata)
    except (ValidationError, Exception) as e:
        return _error("remove_background", e, image_path)


def resize_image(
    image_path: str,
    width: int,
    height: int,
    mode: str = "fit",
    background_color: str = "#FFFFFF",
    output_path: str = None,
    output_format: str = "png",
    quality: int = 95
) -> str:
    """Resize image to exact pixel dimensions. Supports fit, fill, stretch modes."""
    try:
        params = ResizeImageInput(
            image_path=image_path, width=width, height=height,
            mode=mode, background_color=background_color,
            output_path=output_path, output_format=output_format, quality=quality
        )
        result = processor.resize_image(params)
        return _success(result.output_path, f"Resized to {width}x{height}", result.metadata)
    except Exception as e:
        return _error("resize_image", e, {"path": image_path, "w": width, "h": height})


def adjust_colors(
    image_path: str,
    brightness: float = 1.0,
    contrast: float = 1.0,
    saturation: float = 1.0,
    sharpness: float = 1.0,
    hue_shift: int = 0,
    gamma: float = 1.0,
    output_path: str = None
) -> str:
    """Apply color grading: brightness, contrast, saturation, hue, gamma."""
    try:
        params = AdjustColorsInput(
            image_path=image_path, brightness=brightness, contrast=contrast,
            saturation=saturation, sharpness=sharpness, hue_shift=hue_shift,
            gamma=gamma, output_path=output_path
        )
        result = processor.adjust_colors(params)
        return _success(result.output_path, "Color adjustments applied", result.metadata)
    except Exception as e:
        return _error("adjust_colors", e, image_path)


def inpaint_image(
    image_path: str,
    mask_path: str,
    prompt: str = "",
    negative_prompt: str = "artifacts, blurry",
    backend: str = "huggingface",
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    output_path: str = None
) -> str:
    """
    AI-powered inpainting. Fill/replace masked regions using diffusion models.
    Use for object removal, content replacement, damage repair.
    """
    try:
        params = InpaintImageInput(
            image_path=image_path, mask_path=mask_path,
            prompt=prompt, negative_prompt=negative_prompt,
            backend=backend, num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale, output_path=output_path
        )
        result = processor.inpaint_image(params)
        return _success(result.output_path, f"Inpainting complete via {backend}", result.metadata)
    except Exception as e:
        return _error("inpaint_image", e, image_path)


def outpaint_image(
    image_path: str,
    expand_left: int = 0,
    expand_right: int = 0,
    expand_top: int = 0,
    expand_bottom: int = 0,
    prompt: str = "",
    backend: str = "huggingface",
    output_path: str = None
) -> str:
    """
    AI-powered canvas expansion. Generates new content outside original bounds.
    Perfect for extending backgrounds, creating wider compositions.
    """
    try:
        params = OutpaintImageInput(
            image_path=image_path, expand_left=expand_left,
            expand_right=expand_right, expand_top=expand_top,
            expand_bottom=expand_bottom, prompt=prompt,
            backend=backend, output_path=output_path
        )
        result = processor.outpaint_image(params)
        return _success(result.output_path, "Outpainting complete", result.metadata)
    except Exception as e:
        return _error("outpaint_image", e, image_path)


def remove_object(
    image_path: str,
    object_description: str,
    detection_confidence: float = 0.5,
    inpaint_prompt: str = "clean background, seamless",
    output_path: str = None
) -> str:
    """
    Automatically detect and remove a named object from the image.
    Combines YOLO/DINO detection → mask generation → inpainting pipeline.
    """
    try:
        params = ObjectRemovalInput(
            image_path=image_path, object_description=object_description,
            detection_confidence=detection_confidence,
            inpaint_prompt=inpaint_prompt, output_path=output_path
        )
        result = processor.remove_object(params)
        return _success(result.output_path, f"Object '{object_description}' removed", result.metadata)
    except Exception as e:
        return _error("remove_object", e, {"path": image_path, "obj": object_description})


def apply_filter(
    image_path: str,
    filter_type: str,
    intensity: float = 0.8,
    output_path: str = None
) -> str:
    """Apply artistic filters: vintage, noir, cinematic, matte, vivid, etc."""
    try:
        params = ApplyFilterInput(
            image_path=image_path, filter_type=filter_type,
            intensity=intensity, output_path=output_path
        )
        result = processor.apply_filter(params)
        return _success(result.output_path, f"Filter '{filter_type}' applied", result.metadata)
    except Exception as e:
        return _error("apply_filter", e, {"path": image_path, "filter": filter_type})


def convert_format(
    image_path: str,
    target_format: str,
    quality: int = 95,
    strip_metadata: bool = False,
    output_path: str = None
) -> str:
    """Convert image format: PNG↔JPEG↔WEBP with quality control."""
    try:
        params = ConvertFormatInput(
            image_path=image_path, target_format=target_format,
            quality=quality, strip_metadata=strip_metadata, output_path=output_path
        )
        result = processor.convert_format(params)
        return _success(result.output_path, f"Converted to {target_format.upper()}", result.metadata)
    except Exception as e:
        return _error("convert_format", e, image_path)


# ── LangChain Tool Registry ───────────────────────────────────────────────────

def get_all_tools() -> list:
    """
    Returns all tools bound to LangChain's StructuredTool interface.
    Agent uses this list for tool selection and argument extraction.
    """
    return [
        StructuredTool.from_function(
            func=remove_background,
            name="remove_background",
            description=(
                "Remove the background from any image, leaving the subject on a "
                "transparent background. Best for portraits, product photos, logos. "
                "Output is always PNG with alpha channel."
            ),
            args_schema=RemoveBackgroundInput,
            return_direct=False,
        ),
        StructuredTool.from_function(
            func=resize_image,
            name="resize_image",
            description=(
                "Resize an image to specific pixel dimensions. Choose mode: "
                "'fit' (preserve ratio, pad), 'fill' (crop to fill), 'stretch' (distort to fit). "
                "Supports PNG, JPEG, WEBP output."
            ),
            args_schema=ResizeImageInput,
            return_direct=False,
        ),
        StructuredTool.from_function(
            func=adjust_colors,
            name="adjust_colors",
            description=(
                "Adjust image color properties: brightness, contrast, saturation, "
                "sharpness, hue rotation, and gamma. Values above 1.0 increase, below 1.0 decrease. "
                "Use for color correction, mood adjustment, style matching."
            ),
            args_schema=AdjustColorsInput,
            return_direct=False,
        ),
        StructuredTool.from_function(
            func=inpaint_image,
            name="inpaint_image",
            description=(
                "AI-powered inpainting: fill or replace masked image regions. "
                "Requires a mask image (white=modify, black=keep). "
                "Use for removing objects, fixing damage, replacing content with AI generation."
            ),
            args_schema=InpaintImageInput,
            return_direct=False,
        ),
        StructuredTool.from_function(
            func=outpaint_image,
            name="outpaint_image",
            description=(
                "AI-powered outpainting: extend the image canvas beyond its original bounds. "
                "Generates realistic content in expanded areas. "
                "Specify pixels to add on each side (left, right, top, bottom)."
            ),
            args_schema=OutpaintImageInput,
            return_direct=False,
        ),
        StructuredTool.from_function(
            func=remove_object,
            name="remove_object",
            description=(
                "Automatically detect and remove a specific object from an image "
                "using natural language description. Handles detection, masking, and inpainting. "
                "Example: 'the red car in the background', 'the person on the left'."
            ),
            args_schema=ObjectRemovalInput,
            return_direct=False,
        ),
        StructuredTool.from_function(
            func=apply_filter,
            name="apply_filter",
            description=(
                "Apply artistic photo filters: vintage, noir, cinematic, matte, vivid, "
                "cool, warm, faded, cross_process, duotone. Control intensity 0.0–1.0."
            ),
            args_schema=ApplyFilterInput,
            return_direct=False,
        ),
        StructuredTool.from_function(
            func=convert_format,
            name="convert_format",
            description=(
                "Convert image between formats: PNG, JPEG, WEBP. "
                "Control output quality (1-100) and optionally strip EXIF metadata."
            ),
            args_schema=ConvertFormatInput,
            return_direct=False,
        ),
    ]
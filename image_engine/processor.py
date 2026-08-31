"""
ImageProcessor — Unified Engine Interface
Routes every operation to the correct implementation (local or API).
This is the only class imported by agent/tools.py.
"""

import logging
from agent.tool_schemas import (
    RemoveBackgroundInput, ResizeImageInput, AdjustColorsInput,
    InpaintImageInput, OutpaintImageInput, ObjectRemovalInput,
    ApplyFilterInput, ConvertFormatInput
)
from image_engine.result import ProcessingResult
from image_engine import local_ops, api_ops

logger = logging.getLogger(__name__)


class ImageProcessor:
    """
    Single entry point for all image operations.
    
    Routing logic:
    - Local ops (rembg, OpenCV, Pillow): free, instant, no network
    - API ops (HF, Fal.ai): required for generative tasks
    """

    # ── Local Operations ───────────────────────────────────────────────────────

    def remove_background(self, params: RemoveBackgroundInput) -> ProcessingResult:
        logger.info(f"remove_background: {params.image_path}")
        return local_ops.remove_background_local(params)

    def resize_image(self, params: ResizeImageInput) -> ProcessingResult:
        logger.info(f"resize_image: {params.image_path} → {params.width}x{params.height}")
        return local_ops.resize_image_local(params)

    def adjust_colors(self, params: AdjustColorsInput) -> ProcessingResult:
        logger.info(f"adjust_colors: {params.image_path}")
        return local_ops.adjust_colors_local(params)

    def apply_filter(self, params: ApplyFilterInput) -> ProcessingResult:
        logger.info(f"apply_filter: {params.filter_type} on {params.image_path}")
        return local_ops.apply_filter_local(params)

    def convert_format(self, params: ConvertFormatInput) -> ProcessingResult:
        logger.info(f"convert_format: {params.image_path} → {params.target_format}")
        return local_ops.convert_format_local(params)

    # ── API Operations ────────────────────────────────────────────────────────

    def inpaint_image(self, params: InpaintImageInput) -> ProcessingResult:
        logger.info(f"inpaint_image: {params.image_path} via {params.backend}")
        return api_ops.inpaint_image_api(params)

    def outpaint_image(self, params: OutpaintImageInput) -> ProcessingResult:
        logger.info(f"outpaint_image: {params.image_path}")
        return api_ops.outpaint_image_api(params)

    def remove_object(self, params: ObjectRemovalInput) -> ProcessingResult:
        logger.info(f"remove_object: '{params.object_description}' from {params.image_path}")
        return api_ops.remove_object_api(params)
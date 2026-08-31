"""
VideoProcessor — Unified Entry Point
Single class imported by agent tools and Gradio UI.
Routes to local ops or API ops based on task type and availability.
"""

import logging
from video_engine import local_ops, api_ops
from video_engine.schemas import (
    TrimVideoInput, MergeVideosInput, ExtractAudioInput,
    AdjustSpeedInput, AddWatermarkInput, ExtractFramesInput,
    VideoToGifInput, ReplaceAudioInput,
    AnimateImageInput, VideoStyleTransferInput,
)

logger = logging.getLogger(__name__)


class VideoProcessor:

    # ── Local ops ─────────────────────────────────────────────────────────────

    def trim(self, p: TrimVideoInput)           -> dict: return local_ops.trim_video(p)
    def merge(self, p: MergeVideosInput)        -> dict: return local_ops.merge_videos(p)
    def extract_audio(self, p: ExtractAudioInput) -> dict: return local_ops.extract_audio(p)
    def adjust_speed(self, p: AdjustSpeedInput) -> dict: return local_ops.adjust_speed(p)
    def watermark(self, p: AddWatermarkInput)   -> dict: return local_ops.add_watermark(p)
    def extract_frames(self, p: ExtractFramesInput) -> dict: return local_ops.extract_frames(p)
    def to_gif(self, p: VideoToGifInput)        -> dict: return local_ops.video_to_gif(p)
    def replace_audio(self, p: ReplaceAudioInput) -> dict: return local_ops.replace_audio(p)

    # ── API ops ───────────────────────────────────────────────────────────────

    def animate_image(self, p: AnimateImageInput) -> dict:
        return api_ops.animate_image(p)

    def style_transfer(self, p: VideoStyleTransferInput) -> dict:
        return api_ops.video_style_transfer(p)
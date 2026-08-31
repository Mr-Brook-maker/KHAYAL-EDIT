"""
Video Processing Schemas
Typed Pydantic models for every video operation.
All schemas validated before execution — prevents runtime surprises.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Literal
from enum import Enum


class VideoFormat(str, Enum):
    MP4  = "mp4"
    WEBM = "webm"
    AVI  = "avi"
    MOV  = "mov"
    GIF  = "gif"


class AudioFormat(str, Enum):
    MP3  = "mp3"
    WAV  = "wav"
    AAC  = "aac"
    OGG  = "ogg"
    FLAC = "flac"


class VideoCodec(str, Enum):
    H264   = "libx264"
    H265   = "libx265"
    VP9    = "libvpx-vp9"
    AV1    = "libaom-av1"


# ─────────────────────────────────────────────────────────────────────────────
# Local Operation Schemas
# ─────────────────────────────────────────────────────────────────────────────

class TrimVideoInput(BaseModel):
    """Cut a video to a specific time range."""
    video_path: str         = Field(description="Source video file path")
    start_time: float       = Field(ge=0,   description="Start time in seconds")
    end_time:   float       = Field(gt=0,   description="End time in seconds")
    output_path: Optional[str] = None
    output_format: VideoFormat = VideoFormat.MP4
    codec: VideoCodec          = VideoCodec.H264
    audio: bool = Field(default=True, description="Include audio in output")


class MergeVideosInput(BaseModel):
    """Concatenate multiple video files into one."""
    video_paths:   List[str]   = Field(min_length=2, description="Ordered list of video paths")
    output_path:   Optional[str] = None
    output_format: VideoFormat   = VideoFormat.MP4
    codec:         VideoCodec    = VideoCodec.H264
    normalize_resolution: bool   = Field(
        default=True,
        description="Scale all clips to match the first clip's resolution"
    )
    crossfade_duration: float = Field(
        default=0.0, ge=0.0, le=3.0,
        description="Crossfade transition in seconds between clips (0 = cut)"
    )


class ExtractAudioInput(BaseModel):
    """Extract audio track from a video file."""
    video_path:    str
    output_path:   Optional[str]  = None
    output_format: AudioFormat    = AudioFormat.MP3
    bitrate:       str            = Field(default="192k")
    normalize:     bool           = Field(
        default=False,
        description="Apply loudnorm filter for broadcast-standard audio levels"
    )


class AdjustSpeedInput(BaseModel):
    """Change video playback speed (time-stretch / time-lapse)."""
    video_path:   str
    speed_factor: float = Field(
        gt=0.1, le=10.0,
        description="1.0=original, 2.0=double speed, 0.5=half speed"
    )
    output_path:  Optional[str] = None
    preserve_audio: bool = Field(
        default=True,
        description="Pitch-correct audio when changing speed"
    )


class AddWatermarkInput(BaseModel):
    """Overlay image or text watermark on video."""
    video_path:      str
    watermark_path:  Optional[str] = Field(
        default=None, description="Path to overlay image (PNG with alpha recommended)"
    )
    watermark_text:  Optional[str] = Field(default=None, description="Text watermark content")
    position: Literal["top_left","top_right","bottom_left","bottom_right","center"] = "bottom_right"
    opacity:  float = Field(default=0.7, ge=0.0, le=1.0)
    font_size: int  = Field(default=36, ge=8, le=200)
    output_path: Optional[str] = None


class ExtractFramesInput(BaseModel):
    """Extract frames from video as image sequence."""
    video_path:  str
    fps:         float  = Field(default=1.0, gt=0, description="Frames to extract per second")
    start_time:  float  = Field(default=0.0, ge=0)
    end_time:    Optional[float] = None
    output_dir:  Optional[str]  = None
    format:      Literal["png","jpg"] = "png"
    max_frames:  int    = Field(default=100, description="Safety cap on frame count")


class VideoToGifInput(BaseModel):
    """Convert video segment to optimized GIF."""
    video_path:  str
    start_time:  float = Field(default=0.0, ge=0)
    duration:    float = Field(default=5.0, gt=0, le=30.0)
    width:       int   = Field(default=480, ge=64, le=1280)
    fps:         int   = Field(default=12, ge=1, le=30)
    output_path: Optional[str] = None
    optimize:    bool  = True


class ReplaceAudioInput(BaseModel):
    """Replace video's audio track with a different audio file."""
    video_path:  str
    audio_path:  str
    output_path: Optional[str] = None
    loop_audio:  bool  = Field(default=False, description="Loop audio if shorter than video")
    fade_out:    float = Field(default=0.0, ge=0, description="Audio fade-out duration in seconds")


# ─────────────────────────────────────────────────────────────────────────────
# AI / Generative Video Schemas
# ─────────────────────────────────────────────────────────────────────────────

class AnimateImageInput(BaseModel):
    """
    Animate a static image using Stable Video Diffusion (via Replicate free tier).
    Produces a 2–4 second video from a single frame.
    """
    image_path:         str
    motion_bucket_id:   int   = Field(default=127, ge=1, le=255,
        description="Controls motion intensity: 1=minimal, 255=maximum")
    fps:                int   = Field(default=6, ge=1, le=30)
    num_frames:         int   = Field(default=14, ge=2, le=25)
    noise_aug_strength: float = Field(default=0.02, ge=0.0, le=1.0,
        description="Variation seed noise — higher = more creative, less stable")
    output_path:        Optional[str] = None
    backend: Literal["replicate"] = "replicate"


class VideoStyleTransferInput(BaseModel):
    """
    Apply text-guided style transfer to video frames using diffusion models.
    Processes frame-by-frame with temporal consistency hints.
    """
    video_path:       str
    style_prompt:     str   = Field(description="Style description e.g. 'oil painting, Van Gogh'")
    strength:         float = Field(default=0.6, ge=0.1, le=1.0,
        description="Denoising strength: 0.1=subtle, 1.0=total transformation")
    max_frames:       int   = Field(default=30, ge=1, le=120)
    output_path:      Optional[str] = None
    backend: Literal["replicate"] = "replicate"
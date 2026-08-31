"""
Local Video Operations
Zero-cost processing via MoviePy (FFmpeg wrapper) + ffmpeg-python.

Architecture decisions:
- MoviePy for high-level clip composition (merge, crossfade, watermark)
- ffmpeg-python for low-level stream operations (speed, audio, GIF)
  FFmpeg avoids MoviePy's full-decode overhead for single-stream tasks.
- All heavy operations offloaded to thread pool (see async layer)
"""

import logging
import os
import subprocess
from pathlib import Path
from typing import List, Tuple

import ffmpeg
import numpy as np
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# MoviePy Compatible Import Block (Supports MoviePy 1.x & 2.x on Python 3.13)
# ─────────────────────────────────────────────────────────────────────────────
try:
    from moviepy import (
        VideoFileClip, AudioFileClip, ImageClip, TextClip, ColorClip,
        CompositeVideoClip, concatenate_videoclips
    )
except (ImportError, ModuleNotFoundError):
    try:
        from moviepy.editor import (
            VideoFileClip, AudioFileClip, ImageClip, TextClip, ColorClip,
            CompositeVideoClip, concatenate_videoclips
        )
    except (ImportError, ModuleNotFoundError):
        from moviepy.video.io.VideoFileClip import VideoFileClip
        from moviepy.audio.io.AudioFileClip import AudioFileClip
        from moviepy.video.VideoClip import ImageClip, TextClip, ColorClip
        from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
        from moviepy.video.compositing.concatenate import concatenate_videoclips

try:
    from moviepy.video.fx.all import speedx, fadein, fadeout
except (ImportError, ModuleNotFoundError):
    try:
        from moviepy.video.fx import speedx, fadein, fadeout
    except (ImportError, ModuleNotFoundError):
        speedx = None
        fadein = None
        fadeout = None

from video_engine.schemas import (
    TrimVideoInput, MergeVideosInput, ExtractAudioInput,
    AdjustSpeedInput, AddWatermarkInput, ExtractFramesInput,
    VideoToGifInput, ReplaceAudioInput,
)
from video_engine.path_manager import VideoPathManager

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Shared Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_fps(rate_str: str) -> float:
    """Safely parse frame rate string like '30000/1001' to float."""
    if "/" in rate_str:
        num, den = rate_str.split("/")
        return float(num) / float(den)
    return float(rate_str)


def _probe(path: str) -> dict:
    """Get video metadata via ffprobe."""
    try:
        probe = ffmpeg.probe(path)
        vs = next(s for s in probe["streams"] if s["codec_type"] == "video")
        return {
            "width":    int(vs["width"]),
            "height":   int(vs["height"]),
            "fps":      _parse_fps(vs["r_frame_rate"]),
            "duration": float(probe["format"]["duration"]),
            "size_mb":  float(probe["format"]["size"]) / 1_048_576,
        }
    except Exception as e:
        logger.warning(f"ffprobe failed for {path}: {e}")
        return {}


def _ffmpeg_run(stream, desc: str):
    """Execute ffmpeg stream with logging."""
    logger.info(f"FFmpeg: {desc}")
    try:
        stream.run(quiet=True, overwrite_output=True)
    except ffmpeg.Error as e:
        stderr = e.stderr.decode() if e.stderr else "unknown"
        raise RuntimeError(f"FFmpeg error [{desc}]: {stderr}") from e


# ─────────────────────────────────────────────────────────────────────────────
# 1. Trim
# ─────────────────────────────────────────────────────────────────────────────

def trim_video(params: TrimVideoInput) -> dict:
    """
    Precise frame-accurate trim using FFmpeg stream copy (no re-encode)
    when possible, or libx264 when format conversion is needed.
    """
    out = VideoPathManager.resolve(
        params.video_path, params.output_path,
        f"trim_{int(params.start_time)}_{int(params.end_time)}",
        params.output_format.value,
    )

    meta     = _probe(params.video_path)
    duration = params.end_time - params.start_time

    if duration <= 0:
        raise ValueError(f"end_time ({params.end_time}) must be > start_time ({params.start_time})")

    stream = (
        ffmpeg
        .input(params.video_path, ss=params.start_time, t=duration)
        .output(
            str(out),
            vcodec=params.codec.value,
            acodec="aac" if params.audio else None,
            audio_bitrate="192k" if params.audio else None,
            preset="fast",
            crf=18,
        )
    )
    _ffmpeg_run(stream, f"trim {params.start_time}s–{params.end_time}s")

    return {
        "output_path": str(out),
        "duration_s":  duration,
        "source_meta": meta,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. Merge / Concatenate
# ─────────────────────────────────────────────────────────────────────────────

def merge_videos(params: MergeVideosInput) -> dict:
    """
    Concatenate clips using MoviePy.
    Handles resolution normalization and crossfade compositing.
    """
    out = VideoPathManager.resolve(
        params.video_paths[0], params.output_path,
        "merged", params.output_format.value,
    )

    clips: List[VideoFileClip] = []
    first_meta = _probe(params.video_paths[0])
    target_res  = (first_meta.get("width", 1280), first_meta.get("height", 720))

    for i, path in enumerate(params.video_paths):
        clip = VideoFileClip(path)
        if params.normalize_resolution and (clip.w, clip.h) != target_res:
            clip = clip.resize(target_res)
        clips.append(clip)

    if params.crossfade_duration > 0:
        faded_clips = []
        for i, clip in enumerate(clips):
            if i > 0 and hasattr(clip, "crossfadein"):
                clip = clip.crossfadein(params.crossfade_duration)
            if i < len(clips) - 1 and hasattr(clip, "crossfadeout"):
                clip = clip.crossfadeout(params.crossfade_duration)
            faded_clips.append(clip)

        final = concatenate_videoclips(
            faded_clips,
            method="compose",
            padding=-params.crossfade_duration,
        )
    else:
        final = concatenate_videoclips(clips, method="compose")

    final.write_videofile(
        str(out),
        codec=params.codec.value,
        audio_codec="aac",
        fps=first_meta.get("fps", 30),
        logger=None,
        threads=os.cpu_count(),
    )

    for c in clips:
        c.close()
    final.close()

    return {
        "output_path":      str(out),
        "total_clips":      len(params.video_paths),
        "total_duration_s": final.duration,
        "resolution":       f"{target_res[0]}x{target_res[1]}",
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Extract Audio
# ─────────────────────────────────────────────────────────────────────────────

def extract_audio(params: ExtractAudioInput) -> dict:
    """
    Extract audio track via FFmpeg — stream copy when format matches,
    re-encode with chosen codec otherwise. Optionally normalize loudness.
    """
    out = VideoPathManager.resolve(
        params.video_path, params.output_path,
        "audio", params.output_format.value,
    )

    codec_map = {
        "mp3": "libmp3lame", "wav": "pcm_s16le",
        "aac": "aac", "ogg": "libvorbis", "flac": "flac",
    }
    acodec = codec_map[params.output_format.value]

    audio_stream = ffmpeg.input(params.video_path).audio

    if params.normalize:
        audio_stream = audio_stream.filter(
            "loudnorm",
            I=-16,       # Integrated loudness target (EBU R128)
            TP=-1.5,     # True peak
            LRA=11,      # Loudness range
        )

    _ffmpeg_run(
        audio_stream.output(str(out), acodec=acodec, audio_bitrate=params.bitrate),
        f"extract audio → {params.output_format.value}",
    )

    return {
        "output_path": str(out),
        "format":      params.output_format.value,
        "normalized":  params.normalize,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Speed Adjustment
# ─────────────────────────────────────────────────────────────────────────────

def adjust_speed(params: AdjustSpeedInput) -> dict:
    """
    Frame-accurate speed change via FFmpeg setpts/atempo filters.
    """
    out = VideoPathManager.resolve(
        params.video_path, params.output_path,
        f"speed_{params.speed_factor}x", "mp4",
    )

    video = (
        ffmpeg.input(params.video_path)
        .video
        .filter("setpts", f"PTS/{params.speed_factor}")
    )

    if params.preserve_audio:
        audio = ffmpeg.input(params.video_path).audio
        factor = params.speed_factor
        tempos = []
        while factor > 2.0:
            tempos.append(2.0)
            factor /= 2.0
        while factor < 0.5:
            tempos.append(0.5)
            factor /= 0.5
        tempos.append(round(factor, 6))

        for t in tempos:
            audio = audio.filter("atempo", t)

        output = ffmpeg.output(
            video, audio, str(out),
            vcodec="libx264", acodec="aac",
            preset="fast", crf=18,
        )
    else:
        output = ffmpeg.output(
            video, str(out),
            vcodec="libx264", an=None,
            preset="fast", crf=18,
        )

    _ffmpeg_run(output, f"speed {params.speed_factor}x")

    return {
        "output_path":    str(out),
        "speed_factor":   params.speed_factor,
        "audio_adjusted": params.preserve_audio,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. Watermark
# ─────────────────────────────────────────────────────────────────────────────

_POSITION_MAP = {
    "top_left":     lambda cw, ch, iw, ih, m: (m, m),
    "top_right":    lambda cw, ch, iw, ih, m: (cw - iw - m, m),
    "bottom_left":  lambda cw, ch, iw, ih, m: (m, ch - ih - m),
    "bottom_right": lambda cw, ch, iw, ih, m: (cw - iw - m, ch - ih - m),
    "center":       lambda cw, ch, iw, ih, m: ((cw - iw) // 2, (ch - ih) // 2),
}

def add_watermark(params: AddWatermarkInput) -> dict:
    """
    Overlay image or text watermark using MoviePy compositing.
    """
    if not params.watermark_path and not params.watermark_text:
        raise ValueError("Provide either watermark_path or watermark_text")

    out = VideoPathManager.resolve(
        params.video_path, params.output_path,
        "watermarked", "mp4",
    )

    base   = VideoFileClip(params.video_path)
    margin = 20

    if params.watermark_path:
        wm = (
            ImageClip(params.watermark_path)
            .set_duration(base.duration)
            .set_opacity(params.opacity)
        )
        pos_fn = _POSITION_MAP[params.position]
        pos    = pos_fn(base.w, base.h, wm.w, wm.h, margin)
        wm     = wm.set_position(pos)

    else:
        wm = (
            TextClip(
                params.watermark_text,
                fontsize=params.font_size,
                color="white",
                font="DejaVu-Sans-Bold",
                stroke_color="black",
                stroke_width=1,
            )
            .set_duration(base.duration)
            .set_opacity(params.opacity)
        )
        pos_fn = _POSITION_MAP[params.position]
        pos    = pos_fn(base.w, base.h, wm.w, wm.h, margin)
        wm     = wm.set_position(pos)

    composite = CompositeVideoClip([base, wm])
    composite.write_videofile(
        str(out), codec="libx264", audio_codec="aac",
        fps=base.fps, logger=None,
    )
    base.close()

    return {"output_path": str(out), "watermark_applied": True}


# ─────────────────────────────────────────────────────────────────────────────
# 6. Extract Frames
# ─────────────────────────────────────────────────────────────────────────────

def extract_frames(params: ExtractFramesInput) -> dict:
    """
    Extract video frames as image sequence via FFmpeg vf=fps filter.
    """
    out_dir = (
        Path(params.output_dir)
        if params.output_dir
        else VideoPathManager.frames_dir(params.video_path)
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    pattern   = str(out_dir / f"frame_%05d.{params.format}")
    input_kw  = {"ss": params.start_time}
    if params.end_time:
        input_kw["to"] = params.end_time

    stream = (
        ffmpeg
        .input(params.video_path, **input_kw)
        .filter("fps", fps=params.fps)
        .output(pattern, frames=params.max_frames, vsync="vfr")
    )
    _ffmpeg_run(stream, f"extract frames @ {params.fps}fps")

    extracted = sorted(out_dir.glob(f"*.{params.format}"))

    return {
        "output_dir":    str(out_dir),
        "frame_count":   len(extracted),
        "fps_extracted": params.fps,
        "paths":         [str(p) for p in extracted[:5]] + (["..."] if len(extracted) > 5 else []),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7. Video → GIF
# ─────────────────────────────────────────────────────────────────────────────

def video_to_gif(params: VideoToGifInput) -> dict:
    """
    High-quality GIF using FFmpeg palettegen → paletteuse pipeline.
    """
    out = VideoPathManager.resolve(
        params.video_path, params.output_path,
        f"gif_{params.width}w", "gif",
    )

    palette_path = VideoPathManager.temp("png")

    palettegen = (
        ffmpeg
        .input(params.video_path, ss=params.start_time, t=params.duration)
        .filter("fps", fps=params.fps)
        .filter("scale", params.width, -1, flags="lanczos")
        .filter("palettegen", stats_mode="diff")
        .output(str(palette_path))
    )
    _ffmpeg_run(palettegen, "GIF: palette generation")

    video_in   = ffmpeg.input(params.video_path, ss=params.start_time, t=params.duration)
    palette_in = ffmpeg.input(str(palette_path))

    paletteuse = (
        ffmpeg
        .filter(
            [
                video_in.filter("fps", fps=params.fps).filter("scale", params.width, -1, flags="lanczos"),
                palette_in,
            ],
            "paletteuse",
            dither="bayer",
            bayer_scale=5,
            diff_mode="rectangle",
        )
        .output(str(out))
    )
    _ffmpeg_run(paletteuse, "GIF: palette render")

    size_kb = out.stat().st_size / 1024
    palette_path.unlink(missing_ok=True)

    return {
        "output_path": str(out),
        "size_kb":     round(size_kb, 1),
        "fps":         params.fps,
        "width_px":    params.width,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 8. Replace Audio
# ─────────────────────────────────────────────────────────────────────────────

def replace_audio(params: ReplaceAudioInput) -> dict:
    """Replace video audio track — mux video stream with new audio."""
    out = VideoPathManager.resolve(
        params.video_path, params.output_path,
        "replaced_audio", "mp4",
    )

    video_meta = _probe(params.video_path)
    duration   = video_meta.get("duration", 0)

    video = ffmpeg.input(params.video_path).video
    audio = ffmpeg.input(params.audio_path)

    if params.loop_audio:
        audio = ffmpeg.input(params.audio_path, stream_loop=-1).audio
        audio = audio.filter("atrim", duration=duration)
    else:
        audio = audio.audio

    if params.fade_out > 0:
        audio = audio.filter(
            "afade", type="out",
            start_time=max(0, duration - params.fade_out),
            duration=params.fade_out,
        )

    _ffmpeg_run(
        ffmpeg.output(video, audio, str(out), vcodec="copy", acodec="aac", shortest=None),
        "replace audio",
    )

    return {"output_path": str(out), "audio_replaced": True}

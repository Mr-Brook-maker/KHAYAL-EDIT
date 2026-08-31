"""
Gradio UI — Multimodal AI Image & Video Editor
Full-featured web frontend wired directly to the Agent orchestrator.

Features:
- Tabbed layout: Image Editor | Video Editor | Agent Chat | System Status
- Real-time progress via Gradio's built-in streaming
- Side-by-side before/after preview
- Job queue status panel
- Direct tool controls (no LLM needed for simple ops)
"""

import asyncio
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Optional, Generator

import gradio as gr

from agent.orchestrator import ImageEditingAgent
from video_engine.processor import VideoProcessor
from video_engine.async_queue import VideoProcessingQueue, submit_video_job
from video_engine.schemas import (
    TrimVideoInput, MergeVideosInput, ExtractAudioInput,
    AdjustSpeedInput, VideoToGifInput, AnimateImageInput,
)
from image_engine.processor import ImageProcessor
from agent.tool_schemas import (
    RemoveBackgroundInput, ResizeImageInput, AdjustColorsInput, ApplyFilterInput,
)
from devops.cache import FileCache, DiskCleanupScheduler
from devops.rate_limiter import get_executor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Global Singletons ─────────────────────────────────────────────────────────
agent_instance:   Optional[ImageEditingAgent] = None
img_processor     = ImageProcessor()
vid_processor     = VideoProcessor()
cache             = FileCache()
cleanup_scheduler = DiskCleanupScheduler()

def get_agent() -> ImageEditingAgent:
    global agent_instance
    if agent_instance is None:
        agent_instance = ImageEditingAgent()
    return agent_instance


# ── Startup ───────────────────────────────────────────────────────────────────
cleanup_scheduler.start()


# ═════════════════════════════════════════════════════════════════════════════
# EVENT HANDLERS — Image Tab
# ═════════════════════════════════════════════════════════════════════════════

def handle_remove_bg(image_path: str, alpha_matting: bool) -> tuple:
    if not image_path:
        return None, "⚠️ Please upload an image first."
    cached = cache.get(image_path, "remove_bg", {"alpha_matting": alpha_matting})
    if cached:
        return cached, f"✅ Background removed (cached) → `{cached}`"
    try:
        result = img_processor.remove_background(RemoveBackgroundInput(
            image_path=image_path, alpha_matting=alpha_matting
        ))
        cache.put(image_path, "remove_bg", {"alpha_matting": alpha_matting}, result.output_path)
        return result.output_path, f"✅ Background removed → `{result.output_path}`"
    except Exception as e:
        return None, f"❌ Error: {e}"


def handle_resize(
    image_path: str, width: int, height: int, mode: str
) -> tuple:
    if not image_path:
        return None, "⚠️ Please upload an image."
    params = {"width": width, "height": height, "mode": mode}
    cached = cache.get(image_path, "resize", params)
    if cached:
        return cached, f"✅ Resized (cached) → `{cached}`"
    try:
        result = img_processor.resize_image(ResizeImageInput(
            image_path=image_path, width=width, height=height, mode=mode
        ))
        cache.put(image_path, "resize", params, result.output_path)
        return result.output_path, f"✅ Resized to {width}×{height} → `{result.output_path}`"
    except Exception as e:
        return None, f"❌ Error: {e}"


def handle_color_adjust(
    image_path: str,
    brightness: float, contrast: float, saturation: float,
    sharpness: float, hue_shift: int, gamma: float,
) -> tuple:
    if not image_path:
        return None, "⚠️ Please upload an image."
    params = dict(
        brightness=brightness, contrast=contrast, saturation=saturation,
        sharpness=sharpness, hue_shift=hue_shift, gamma=gamma
    )
    cached = cache.get(image_path, "color_adjust", params)
    if cached:
        return cached, "✅ Color adjusted (cached)"
    try:
        result = img_processor.adjust_colors(AdjustColorsInput(image_path=image_path, **params))
        cache.put(image_path, "color_adjust", params, result.output_path)
        return result.output_path, f"✅ Color adjusted → `{result.output_path}`"
    except Exception as e:
        return None, f"❌ Error: {e}"


def handle_apply_filter(image_path: str, filter_type: str, intensity: float) -> tuple:
    if not image_path:
        return None, "⚠️ Please upload an image."
    params = {"filter_type": filter_type, "intensity": intensity}
    cached = cache.get(image_path, "apply_filter", params)
    if cached:
        return cached, f"✅ Filter applied (cached)"
    try:
        result = img_processor.apply_filter(ApplyFilterInput(
            image_path=image_path, filter_type=filter_type, intensity=intensity
        ))
        cache.put(image_path, "apply_filter", params, result.output_path)
        return result.output_path, f"✅ {filter_type.title()} filter applied → `{result.output_path}`"
    except Exception as e:
        return None, f"❌ Error: {e}"


# ═════════════════════════════════════════════════════════════════════════════
# EVENT HANDLERS — Video Tab
# ═════════════════════════════════════════════════════════════════════════════

def handle_trim_video(
    video_path: str,
    start_time: float,
    end_time: float,
    progress: gr.Progress = gr.Progress()
) -> tuple:
    if not video_path:
        return None, "⚠️ Please upload a video."
    progress(0, desc="Submitting job...")
    try:
        result = asyncio.run(submit_video_job(
            vid_processor.trim,
            TrimVideoInput(video_path=video_path, start_time=start_time, end_time=end_time)
        ))
        if result.get("result"):
            out = result["result"]["output_path"]
            return out, f"✅ Trimmed {start_time}s–{end_time}s → `{out}`"
        return None, f"❌ {result.get('error', 'Unknown error')}"
    except Exception as e:
        return None, f"❌ Error: {e}"


def handle_extract_audio(
    video_path: str,
    audio_format: str,
    normalize: bool,
    progress: gr.Progress = gr.Progress()
) -> tuple:
    if not video_path:
        return None, "⚠️ Please upload a video."
    progress(0.1, desc="Extracting audio...")
    try:
        result = asyncio.run(submit_video_job(
            vid_processor.extract_audio,
            ExtractAudioInput(video_path=video_path, output_format=audio_format, normalize=normalize)
        ))
        if result.get("result"):
            out = result["result"]["output_path"]
            return out, f"✅ Audio extracted → `{out}`"
        return None, f"❌ {result.get('error')}"
    except Exception as e:
        return None, f"❌ Error: {e}"


def handle_speed(
    video_path: str, speed: float,
    progress: gr.Progress = gr.Progress()
) -> tuple:
    if not video_path:
        return None, "⚠️ Please upload a video."
    progress(0.1, desc=f"Applying {speed}x speed...")
    try:
        result = asyncio.run(submit_video_job(
            vid_processor.adjust_speed,
            AdjustSpeedInput(video_path=video_path, speed_factor=speed)
        ))
        if result.get("result"):
            out = result["result"]["output_path"]
            return out, f"✅ Speed set to {speed}x → `{out}`"
        return None, f"❌ {result.get('error')}"
    except Exception as e:
        return None, f"❌ Error: {e}"


def handle_to_gif(
    video_path: str, start: float, duration: float, width: int, fps: int,
    progress: gr.Progress = gr.Progress()
) -> tuple:
    if not video_path:
        return None, "⚠️ Please upload a video."
    progress(0.1, desc="Generating GIF...")
    try:
        result = asyncio.run(submit_video_job(
            vid_processor.to_gif,
            VideoToGifInput(
                video_path=video_path, start_time=start,
                duration=duration, width=width, fps=fps
            )
        ))
        if result.get("result"):
            out = result["result"]["output_path"]
            size = result["result"].get("size_kb", "?")
            return out, f"✅ GIF created ({size}KB) → `{out}`"
        return None, f"❌ {result.get('error')}"
    except Exception as e:
        return None, f"❌ Error: {e}"


def handle_animate_image(
    image_path: str,
    motion: int,
    fps: int,
    progress: gr.Progress = gr.Progress()
) -> tuple:
    if not image_path:
        return None, "⚠️ Please upload an image."
    progress(0.1, desc="Submitting to Stable Video Diffusion...")
    try:
        result = asyncio.run(submit_video_job(
            vid_processor.animate_image,
            AnimateImageInput(image_path=image_path, motion_bucket_id=motion, fps=fps),
            timeout=300,
        ))
        if result.get("result"):
            out = result["result"]["output_path"]
            return out, f"✅ Animation complete → `{out}`"
        return None, f"❌ {result.get('error')}"
    except Exception as e:
        return None, f"❌ Error: {e}"


# ═════════════════════════════════════════════════════════════════════════════
# EVENT HANDLERS — Agent Chat Tab
# ═════════════════════════════════════════════════════════════════════════════

def handle_agent_chat(
    message: str,
    image_path: Optional[str],
    video_path: Optional[str],
    history: list,
) -> Generator:
    if not message.strip():
        yield history, "⚠️ Please enter a prompt.", None
        return

    media_path = image_path or video_path

    history = history + [[message, None]]
    yield history, "🤔 Agent is thinking...", None

    try:
        agent = get_agent()
        result = agent.run(message, image_path=media_path)

        response_parts = [result["output"]]

        if result["steps"]:
            response_parts.append(f"\n\n---\n**🔧 Steps taken ({len(result['steps'])}):**")
            for i, step in enumerate(result["steps"], 1):
                try:
                    obs = json.loads(step["output"])
                    status_icon = "✅" if obs.get("status") == "success" else "❌"
                    out_path    = obs.get("output_path", "")
                    response_parts.append(f"\n{i}. {status_icon} `{step['tool']}` → `{out_path}`")
                except Exception:
                    response_parts.append(f"\n{i}. `{step['tool']}`")

        full_response = "".join(response_parts)
        history[-1][1] = full_response

        final_output = result.get("final_output_path")
        yield history, "✅ Done!", final_output

    except Exception as e:
        history[-1][1] = f"❌ Agent error: {e}"
        yield history, f"❌ Error: {e}", None


def clear_agent_memory():
    try:
        get_agent().reset_memory()
        return "✅ Agent memory cleared."
    except Exception as e:
        return f"❌ Error: {e}"


# ═════════════════════════════════════════════════════════════════════════════
# EVENT HANDLERS — System Tab
# ═════════════════════════════════════════════════════════════════════════════

def get_system_status() -> str:
    executor  = get_executor()
    api_status = executor.get_status()
    cache_stats = cache.get_stats()

    lines = ["## 🔑 API Key Status\n"]
    for provider, info in api_status.items():
        avail = info["available"]
        total = info["total_keys"]
        icon  = "🟢" if avail > 0 else "🔴"
        lines.append(f"- {icon} **{provider}**: {avail}/{total} keys available")
        for k in info["keys"]:
            bo = k.get("backoff_remaining_s", 0)
            suffix = f" (backoff: {bo:.0f}s)" if bo > 0 else ""
            lines.append(f"  - `...{k['suffix']}` | {k['requests']} requests{suffix}")

    lines.append("\n## 💾 Cache Stats\n")
    lines.append(f"- Entries: {cache_stats['valid_entries']} valid / {cache_stats['total_entries']} total")
    lines.append(f"- Size: {cache_stats['total_size_mb']} MB / {cache_stats['max_size_gb']*1024:.0f} MB")
    lines.append(f"- TTL: {cache_stats['ttl_hours']}h")

    return "\n".join(lines)


def handle_clear_cache() -> str:
    cache.cleanup(force=True)
    return "✅ Cache cleared."


# ═════════════════════════════════════════════════════════════════════════════
# GRADIO UI LAYOUT
# ═════════════════════════════════════════════════════════════════════════════

CSS = """
:root {
    --primary:    #7C3AED;
    --secondary:  #06B6D4;
    --success:    #10B981;
    --warning:    #F59E0B;
    --danger:     #EF4444;
    --bg:         #0F0F1A;
    --surface:    #1A1A2E;
    --surface2:   #16213E;
    --text:       #E2E8F0;
    --muted:      #64748B;
    --radius:     12px;
    --shadow:     0 4px 24px rgba(124,58,237,0.15);
}

body, .gradio-container {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Inter', system-ui, sans-serif !important;
}

.app-header {
    background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
    padding: 28px 32px;
    border-radius: var(--radius);
    margin-bottom: 20px;
    text-align: center;
    box-shadow: var(--shadow);
}
.app-header h1 { font-size: 2rem; font-weight: 800; color: white; margin: 0; }
.app-header p  { color: rgba(255,255,255,0.85); margin: 6px 0 0; }

.panel-card {
    background: var(--surface);
    border: 1px solid rgba(124,58,237,0.2);
    border-radius: var(--radius);
    padding: 20px;
    box-shadow: var(--shadow);
}

.tab-nav { background: var(--surface2) !important; border-radius: 10px; }
.tab-nav button {
    color: var(--muted) !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    transition: all 0.2s !important;
}
.tab-nav button.selected {
    background: var(--primary) !important;
    color: white !important;
}

.btn-primary {
    background: linear-gradient(135deg, var(--primary), #5B21B6) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    padding: 10px 24px !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
}
.btn-primary:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(124,58,237,0.4) !important;
}
.btn-secondary {
    background: var(--surface2) !important;
    color: var(--secondary) !important;
    border: 1px solid var(--secondary) !important;
    border-radius: 8px !important;
}

.status-box {
    background: var(--surface2);
    border-left: 4px solid var(--primary);
    border-radius: 0 var(--radius) var(--radius) 0;
    padding: 10px 16px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.875rem;
    color: var(--text);
}

.preview-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
}
.preview-label {
    text-align: center;
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--muted);
    margin-bottom: 8px;
}

input[type=range]::-webkit-slider-thumb { background: var(--primary) !important; }

.chatbot { background: var(--surface) !important; border-radius: var(--radius) !important; }
.message.user { background: rgba(124,58,237,0.15) !important; }
.message.bot  { background: rgba(6,182,212,0.08) !important; }
"""

# ── Build UI ──────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    with gr.Blocks(
        css=CSS,
        title="🎨 Multimodal AI Editor",
        theme=gr.themes.Base(
            primary_hue="purple",
            secondary_hue="cyan",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Inter"),
        ),
    ) as demo:

        # ── Header ────────────────────────────────────────────────────────────
        gr.HTML("""
        <div class="app-header">
            <h1>🎨 Multimodal AI Editor</h1>
            <p>AI-powered image & video editing · LangChain + Gemini · 100% free APIs</p>
        </div>
        """)

        with gr.Tabs() as tabs:

            # ══════════════════════════════════════════════════════════════════
            # TAB 1: AI AGENT CHAT
            # ══════════════════════════════════════════════════════════════════
            with gr.Tab("🤖 AI Agent", id="agent"):
                gr.Markdown(
                    "**Describe any editing task in plain English.** "
                    "The agent will plan and execute the right tools automatically."
                )
                with gr.Row():
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(
                            label="Agent Conversation",
                            elem_classes=["chatbot"],
                            height=480,
                            avatar_images=(None, "https://img.icons8.com/fluency/48/robot.png"),
                        )
                        with gr.Row():
                            agent_prompt = gr.Textbox(
                                placeholder="e.g. Remove background from my photo, apply cinematic filter, resize to 1920×1080",
                                label="Your editing prompt",
                                lines=2,
                                scale=5,
                            )
                            agent_send_btn = gr.Button("Send ▶", elem_classes=["btn-primary"], scale=1)

                        agent_status = gr.Textbox(
                            label="Status", value="Ready.", interactive=False,
                            elem_classes=["status-box"],
                        )

                    with gr.Column(scale=2):
                        gr.Markdown("**📎 Attach Media (optional)**")
                        agent_image_in = gr.Image(
                            type="filepath", label="Input Image",
                            elem_classes=["panel-card"]
                        )
                        agent_video_in = gr.Video(
                            label="Input Video",
                            elem_classes=["panel-card"]
                        )
                        agent_output_preview = gr.Image(
                            label="🖼️ Last Output Preview",
                            elem_classes=["panel-card"],
                            interactive=False,
                        )
                        with gr.Row():
                            clear_memory_btn = gr.Button(
                                "🧹 Clear Memory", elem_classes=["btn-secondary"]
                            )
                        memory_status = gr.Textbox(
                            label="", interactive=False, visible=True, lines=1
                        )

                # ── Wiring ────────────────────────────────────────────────────
                chat_state = gr.State([])

                def send_message(msg, img, vid, history):
                    yield from handle_agent_chat(msg, img, vid, history)

                agent_send_btn.click(
                    fn=send_message,
                    inputs=[agent_prompt, agent_image_in, agent_video_in, chat_state],
                    outputs=[chatbot, agent_status, agent_output_preview],
                    show_progress="full",
                ).then(lambda: gr.update(value=""), outputs=agent_prompt)

                agent_prompt.submit(
                    fn=send_message,
                    inputs=[agent_prompt, agent_image_in, agent_video_in, chat_state],
                    outputs=[chatbot, agent_status, agent_output_preview],
                ).then(lambda: gr.update(value=""), outputs=agent_prompt)

                clear_memory_btn.click(
                    fn=clear_agent_memory, outputs=memory_status
                )


            # ══════════════════════════════════════════════════════════════════
            # TAB 2: IMAGE EDITOR
            # ══════════════════════════════════════════════════════════════════
            with gr.Tab("🖼️ Image Editor", id="image"):
                with gr.Row():
                    # ── Input Panel ───────────────────────────────────────────
                    with gr.Column(scale=1, elem_classes=["panel-card"]):
                        gr.Markdown("### 📂 Input Image")
                        img_input = gr.Image(
                            type="filepath", label="Upload Image",
                            height=280,
                        )

                        with gr.Accordion("✂️ Remove Background", open=True):
                            alpha_matting_cb = gr.Checkbox(
                                label="Alpha matting (better edges)", value=True
                            )
                            remove_bg_btn = gr.Button(
                                "Remove Background", elem_classes=["btn-primary"]
                            )

                        with gr.Accordion("📐 Resize", open=False):
                            with gr.Row():
                                resize_w = gr.Number(label="Width px",  value=800, precision=0)
                                resize_h = gr.Number(label="Height px", value=600, precision=0)
                            resize_mode = gr.Dropdown(
                                ["fit", "fill", "stretch", "thumbnail"],
                                label="Mode", value="fit",
                            )
                            resize_btn = gr.Button("Resize", elem_classes=["btn-primary"])

                        with gr.Accordion("🎨 Color Adjustment", open=False):
                            brightness_sl = gr.Slider(0.1, 3.0, value=1.0, step=0.05, label="Brightness")
                            contrast_sl   = gr.Slider(0.1, 3.0, value=1.0, step=0.05, label="Contrast")
                            saturation_sl = gr.Slider(0.0, 3.0, value=1.0, step=0.05, label="Saturation")
                            sharpness_sl  = gr.Slider(0.0, 5.0, value=1.0, step=0.1,  label="Sharpness")
                            hue_sl        = gr.Slider(-180, 180, value=0, step=1, label="Hue Shift (°)")
                            gamma_sl      = gr.Slider(0.1, 3.0, value=1.0, step=0.05, label="Gamma")
                            color_btn     = gr.Button("Apply Colors", elem_classes=["btn-primary"])

                        with gr.Accordion("✨ Filters", open=False):
                            filter_dd = gr.Dropdown(
                                choices=["vintage","noir","cinematic","matte","vivid",
                                         "cool","warm","faded","cross_process","duotone"],
                                label="Filter", value="cinematic",
                            )
                            intensity_sl = gr.Slider(0.0, 1.0, value=0.8, step=0.05, label="Intensity")
                            filter_btn   = gr.Button("Apply Filter", elem_classes=["btn-primary"])

                    # ── Output Panel ──────────────────────────────────────────
                    with gr.Column(scale=1):
                        gr.Markdown("### 🔍 Before / After")
                        with gr.Row(elem_classes=["preview-row"]):
                            with gr.Column():
                                gr.HTML('<p class="preview-label">Original</p>')
                                img_before = gr.Image(
                                    label="", interactive=False, height=260
                                )
                            with gr.Column():
                                gr.HTML('<p class="preview-label">Processed</p>')
                                img_after = gr.Image(
                                    label="", interactive=False, height=260
                                )

                        img_status = gr.Textbox(
                            label="Status", interactive=False,
                            elem_classes=["status-box"]
                        )
                        img_download = gr.File(label="⬇️ Download Result", visible=False)

                # ── Copy input → before preview on upload ─────────────────────
                img_input.change(fn=lambda x: x, inputs=img_input, outputs=img_before)

                def wrap_and_show_download(fn, *args):
                    out_img, status = fn(*args)
                    dl = gr.update(value=out_img, visible=bool(out_img))
                    return out_img, status, dl

                remove_bg_btn.click(
                    fn=lambda img, am: wrap_and_show_download(handle_remove_bg, img, am),
                    inputs=[img_input, alpha_matting_cb],
                    outputs=[img_after, img_status, img_download],
                    show_progress="full",
                )
                resize_btn.click(
                    fn=lambda img, w, h, m: wrap_and_show_download(handle_resize, img, int(w), int(h), m),
                    inputs=[img_input, resize_w, resize_h, resize_mode],
                    outputs=[img_after, img_status, img_download],
                    show_progress="full",
                )
                color_btn.click(
                    fn=lambda img, b, c, s, sh, hu, g: wrap_and_show_download(
                        handle_color_adjust, img, b, c, s, sh, int(hu), g
                    ),
                    inputs=[img_input, brightness_sl, contrast_sl, saturation_sl,
                            sharpness_sl, hue_sl, gamma_sl],
                    outputs=[img_after, img_status, img_download],
                    show_progress="full",
                )
                filter_btn.click(
                    fn=lambda img, f, i: wrap_and_show_download(handle_apply_filter, img, f, i),
                    inputs=[img_input, filter_dd, intensity_sl],
                    outputs=[img_after, img_status, img_download],
                    show_progress="full",
                )


            # ══════════════════════════════════════════════════════════════════
            # TAB 3: VIDEO EDITOR
            # ══════════════════════════════════════════════════════════════════
            with gr.Tab("🎬 Video Editor", id="video"):
                with gr.Row():
                    # ── Input Panel ───────────────────────────────────────────
                    with gr.Column(scale=1, elem_classes=["panel-card"]):
                        gr.Markdown("### 📂 Input Video")
                        vid_input = gr.Video(label="Upload Video")

                        with gr.Accordion("✂️ Trim", open=True):
                            with gr.Row():
                                trim_start = gr.Number(label="Start (s)", value=0.0)
                                trim_end   = gr.Number(label="End (s)",   value=10.0)
                            trim_btn = gr.Button("Trim Video", elem_classes=["btn-primary"])

                        with gr.Accordion("🔊 Extract Audio", open=False):
                            audio_fmt_dd = gr.Dropdown(
                                ["mp3","wav","aac","ogg"], label="Format", value="mp3"
                            )
                            normalize_cb = gr.Checkbox(label="Normalize loudness", value=False)
                            extract_audio_btn = gr.Button("Extract Audio", elem_classes=["btn-primary"])

                        with gr.Accordion("⚡ Speed Control", open=False):
                            speed_sl = gr.Slider(0.25, 4.0, value=1.0, step=0.25, label="Speed Factor")
                            speed_btn = gr.Button("Apply Speed", elem_classes=["btn-primary"])

                        with gr.Accordion("🎞️ Video → GIF", open=False):
                            with gr.Row():
                                gif_start    = gr.Number(label="Start (s)", value=0)
                                gif_duration = gr.Number(label="Duration (s)", value=5)
                            with gr.Row():
                                gif_width = gr.Slider(128, 800, value=480, step=32, label="Width px")
                                gif_fps   = gr.Slider(6, 24, value=12, step=2, label="FPS")
                            gif_btn = gr.Button("Create GIF", elem_classes=["btn-primary"])

                        with gr.Accordion("🎭 Animate Image (AI)", open=False):
                            gr.Markdown("*Upload an image above (not video) for animation.*")
                            anim_image_in = gr.Image(type="filepath", label="Source Image")
                            motion_sl     = gr.Slider(1, 255, value=127, step=1, label="Motion Intensity")
                            anim_fps_sl   = gr.Slider(4, 24, value=8, step=2, label="Output FPS")
                            anim_btn      = gr.Button(
                                "🌀 Animate (Replicate SVD)", elem_classes=["btn-primary"]
                            )

                    # ── Output Panel ──────────────────────────────────────────
                    with gr.Column(scale=1):
                        gr.Markdown("### 📺 Preview")
                        vid_output = gr.Video(
                            label="Processed Output", interactive=False, height=320
                        )
                        audio_output = gr.Audio(
                            label="Audio Output", interactive=False, visible=True
                        )
                        gif_output = gr.Image(
                            label="GIF Preview", interactive=False, visible=False
                        )
                        vid_status = gr.Textbox(
                            label="Status", interactive=False,
                            elem_classes=["status-box"]
                        )
                        vid_download = gr.File(label="⬇️ Download", visible=False)

                # ── Video Wiring ───────────────────────────────────────────────
                def wrap_video_event(fn, *args, output_type="video"):
                    out_path, status = fn(*args)
                    if not out_path:
                        return None, gr.update(visible=False), gr.update(visible=False), status, gr.update(visible=False)
                    
                    if output_type == "audio":
                        return None, gr.update(value=out_path, visible=True), gr.update(visible=False), status, gr.update(value=out_path, visible=True)
                    elif output_type == "gif":
                        return None, gr.update(visible=False), gr.update(value=out_path, visible=True), status, gr.update(value=out_path, visible=True)
                    else:
                        return out_path, gr.update(visible=False), gr.update(visible=False), status, gr.update(value=out_path, visible=True)

                vid_outputs = [vid_output, audio_output, gif_output, vid_status, vid_download]

                trim_btn.click(
                    fn=lambda v, s, e: wrap_video_event(handle_trim_video, v, float(s), float(e)),
                    inputs=[vid_input, trim_start, trim_end],
                    outputs=vid_outputs,
                    show_progress="full",
                )
                extract_audio_btn.click(
                    fn=lambda v, f, n: wrap_video_event(handle_extract_audio, v, f, n, output_type="audio"),
                    inputs=[vid_input, audio_fmt_dd, normalize_cb],
                    outputs=vid_outputs,
                    show_progress="full",
                )
                speed_btn.click(
                    fn=lambda v, s: wrap_video_event(handle_speed, v, float(s)),
                    inputs=[vid_input, speed_sl],
                    outputs=vid_outputs,
                    show_progress="full",
                )
                gif_btn.click(
                    fn=lambda v, s, d, w, f: wrap_video_event(handle_to_gif, v, float(s), float(d), int(w), int(f), output_type="gif"),
                    inputs=[vid_input, gif_start, gif_duration, gif_width, gif_fps],
                    outputs=vid_outputs,
                    show_progress="full",
                )
                anim_btn.click(
                    fn=lambda img, m, f: wrap_video_event(handle_animate_image, img, int(m), int(f)),
                    inputs=[anim_image_in, motion_sl, anim_fps_sl],
                    outputs=vid_outputs,
                    show_progress="full",
                )

            # ══════════════════════════════════════════════════════════════════
            # TAB 4: SYSTEM STATUS
            # ══════════════════════════════════════════════════════════════════
            with gr.Tab("⚙️ System Status", id="system"):
                with gr.Column(elem_classes=["panel-card"]):
                    sys_markdown = gr.Markdown(value=get_system_status)
                    with gr.Row():
                        refresh_sys_btn = gr.Button("🔄 Refresh Status", elem_classes=["btn-primary"])
                        clear_cache_btn = gr.Button("🗑️ Clear Cache", elem_classes=["btn-secondary"])
                    sys_status_msg = gr.Textbox(label="", interactive=False)

                refresh_sys_btn.click(fn=get_system_status, outputs=sys_markdown)
                clear_cache_btn.click(fn=handle_clear_cache, outputs=sys_status_msg)

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.queue().launch(server_name="0.0.0.0", server_port=7860)

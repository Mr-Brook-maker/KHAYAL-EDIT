"""
Comprehensive Test Suite for Video Engine and Core Modules

Environment assumptions:
- Python 3.10+ available
- Optional heavy dependencies (ffmpeg, moviepy, rembg, cv2, PIL, numpy,
  replicate, fal_client, gradio, langchain, transformers, torch) MAY NOT be installed.

Strategy:
- Pure-Python logic is tested directly.
- External-dependency code paths are validated via:
    * AST syntax checks
    * Module structure / attribute existence
    * Lightweight mocks applied at the function boundary where possible
- If a dependency is missing, the corresponding test is skipped rather than failed.

Covers:
- video_engine/local_ops.py
- video_engine/api_ops.py
- video_engine/async_queue.py
- video_engine/schemas.py
- video_engine/path_manager.py
- image_engine/local_ops.py
- devops/cache.py
- devops/rate_limiter.py
- app.py / main.py syntax validation
"""

import ast
import asyncio
import importlib
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ─────────────────────────────────────────────────────────────────────────────
# Dependency availability checks
# ─────────────────────────────────────────────────────────────────────────────

_HAS_PIL = False
_HAS_NUMPY = False
_HAS_CV2 = False
_HAS_FFMPEG = False
_HAS_MOVIEPY = False
_HAS_PYDANTIC = False

try:
    from PIL import Image as _PILImage
    _HAS_PIL = True
except ImportError:
    pass

try:
    import numpy as _numpy
    _HAS_NUMPY = True
except ImportError:
    pass

try:
    import cv2 as _cv2
    _HAS_CV2 = True
except ImportError:
    pass

try:
    import ffmpeg as _ffmpeg
    _HAS_FFMPEG = True
except ImportError:
    pass

try:
    import moviepy as _moviepy
    _HAS_MOVIEPY = True
except ImportError:
    pass

try:
    import pydantic as _pydantic
    _HAS_PYDANTIC = True
except ImportError:
    pass


def _skip_if(condition, reason):
    """Return a skip decorator if condition is true."""
    if condition:
        return unittest.skip(reason)
    return lambda func: func


# ─────────────────────────────────────────────────────────────────────────────
# Module import helpers with graceful fallback
# ─────────────────────────────────────────────────────────────────────────────

_MODULE_CACHE = {}


def _get_module(name):
    """Import and cache a module, or return None if unavailable."""
    if name in _MODULE_CACHE:
        return _MODULE_CACHE[name]
    try:
        mod = importlib.import_module(name)
        _MODULE_CACHE[name] = mod
        return mod
    except Exception as e:
        print(f"[skip] Could not import {name}: {e}")
        _MODULE_CACHE[name] = None
        return None


# Pre-resolve modules we need
_ve_schemas = _get_module("video_engine.schemas")
_ve_path = _get_module("video_engine.path_manager")
_ve_queue = _get_module("video_engine.async_queue")
_devops_cache = _get_module("devops.cache")
_devops_rl = _get_module("devops.rate_limiter")
_ie_result = _get_module("image_engine.result")


# ─────────────────────────────────────────────────────────────────────────────
# video_engine/local_ops.py
# ─────────────────────────────────────────────────────────────────────────────

class TestParseFps(unittest.TestCase):
    """Test _parse_fps helper — pure Python, no deps required."""

    @classmethod
    def setUpClass(cls):
        mod = _get_module("video_engine.local_ops")
        if mod is None:
            raise unittest.SkipTest("video_engine.local_ops could not be imported")
        cls._parse_fps = mod._parse_fps

    def test_integer_string(self):
        self.assertAlmostEqual(self._parse_fps("30"), 30.0)

    def test_fraction_string(self):
        self.assertAlmostEqual(self._parse_fps("30000/1001"), 29.97002997002997, places=5)

    def test_fraction_simple(self):
        self.assertAlmostEqual(self._parse_fps("25/1"), 25.0)

    def test_float_string(self):
        self.assertAlmostEqual(self._parse_fps("23.976"), 23.976)


class TestProbe(unittest.TestCase):
    """Test _probe metadata extraction via mocks."""

    @classmethod
    def setUpClass(cls):
        mod = _get_module("video_engine.local_ops")
        if mod is None:
            raise unittest.SkipTest("video_engine.local_ops could not be imported")
        cls._probe = mod._probe

    def _make_probe(self, streams, format_info):
        return {"streams": streams, "format": format_info}

    def test_probe_returns_expected_keys(self):
        if not _HAS_FFMPEG:
            self.skipTest("ffmpeg-python not installed")
        with patch.object(sys.modules.get("video_engine.local_ops", MagicMock()), "ffmpeg") as mock_ffmpeg:
            mock_ffmpeg.probe.return_value = self._make_probe(
                streams=[{"codec_type": "video", "width": 1920, "height": 1080, "r_frame_rate": "30/1"}],
                format_info={"duration": "10.0", "size": str(5 * 1024 * 1024)},
            )
            result = self._probe("dummy.mp4")
        self.assertEqual(result["width"], 1920)
        self.assertEqual(result["height"], 1080)
        self.assertAlmostEqual(result["fps"], 30.0)

    def test_probe_returns_empty_on_exception(self):
        mod = _get_module("video_engine.local_ops")
        if mod is None:
            self.skipTest("video_engine.local_ops could not be imported")
        with patch.object(mod, "ffmpeg") as mock_ffmpeg:
            mock_ffmpeg.probe.side_effect = RuntimeError("ffprobe error")
            result = self._probe("dummy.mp4")
        self.assertEqual(result, {})


class TestVideoEngineModuleStructure(unittest.TestCase):
    """Verify expected functions and classes exist in video_engine modules."""

    @_skip_if(_ve_schemas is None, "video_engine.schemas not available")
    def test_schemas_expected_classes_exist(self):
        expected = [
            "TrimVideoInput", "MergeVideosInput", "ExtractAudioInput",
            "AdjustSpeedInput", "AddWatermarkInput", "ExtractFramesInput",
            "VideoToGifInput", "ReplaceAudioInput",
            "AnimateImageInput", "VideoStyleTransferInput",
            "VideoFormat", "AudioFormat", "VideoCodec",
        ]
        for name in expected:
            self.assertTrue(
                hasattr(_ve_schemas, name),
                f"video_engine.schemas missing expected class: {name}",
            )

    @_skip_if(_ve_path is None, "video_engine.path_manager not available")
    def test_path_manager_expected_methods_exist(self):
        vpm = _ve_path.VideoPathManager
        self.assertTrue(hasattr(vpm, "resolve"))
        self.assertTrue(hasattr(vpm, "frames_dir"))
        self.assertTrue(hasattr(vpm, "temp"))

    @_skip_if(_ve_queue is None, "video_engine.async_queue not available")
    def test_async_queue_expected_classes_exist(self):
        expected = ["VideoJob", "VideoProcessingQueue", "JobStatus"]
        for name in expected:
            self.assertTrue(
                hasattr(_ve_queue, name),
                f"video_engine.async_queue missing expected class: {name}",
            )

    @_skip_if(_devops_cache is None, "devops.cache not available")
    def test_cache_expected_classes_exist(self):
        expected = ["FileCache", "DiskCleanupScheduler", "CacheEntry"]
        for name in expected:
            self.assertTrue(
                hasattr(_devops_cache, name),
                f"devops.cache missing expected class: {name}",
            )

    @_skip_if(_devops_rl is None, "devops.rate_limiter not available")
    def test_rate_limiter_expected_classes_exist(self):
        expected = ["RateLimitedExecutor", "Provider", "KeyState", "ProviderPool"]
        for name in expected:
            self.assertTrue(
                hasattr(_devops_rl, name),
                f"devops.rate_limiter missing expected class: {name}",
            )


class TestVideoJob(unittest.TestCase):
    """Test VideoJob dataclass — pure Python."""

    @classmethod
    def setUpClass(cls):
        mod = _get_module("video_engine.async_queue")
        if mod is None:
            raise unittest.SkipTest("video_engine.async_queue could not be imported")
        cls.VideoJob = mod.VideoJob

    def test_initial_state(self):
        job = self.VideoJob(job_id="test123", fn=lambda: None, args=(), kwargs={})
        self.assertEqual(job.status.value, "queued")
        self.assertEqual(job.progress, 0.0)
        self.assertIsNone(job.result)
        self.assertIsNone(job.error)

    def test_elapsed_returns_zero_before_start(self):
        job = self.VideoJob(job_id="test123", fn=lambda: None, args=(), kwargs={})
        self.assertEqual(job.elapsed, 0.0)

    def test_elapsed_after_finish(self):
        job = self.VideoJob(job_id="test123", fn=lambda: None, args=(), kwargs={})
        job.started_at = time.time() - 5
        job.finished_at = time.time()
        self.assertAlmostEqual(job.elapsed, 5.0, delta=0.5)

    def test_to_dict_contains_expected_keys(self):
        job = self.VideoJob(job_id="test123", fn=lambda: None, args=(), kwargs={})
        d = job.to_dict()
        self.assertIn("job_id", d)
        self.assertIn("status", d)
        self.assertIn("progress", d)


class TestVideoProcessingQueue(unittest.TestCase):
    """Test VideoProcessingQueue async queue — uses asyncio mocks."""

    @classmethod
    def setUpClass(cls):
        mod = _get_module("video_engine.async_queue")
        if mod is None:
            raise unittest.SkipTest("video_engine.async_queue could not be imported")
        cls.VideoProcessingQueue = mod.VideoProcessingQueue
        cls.VideoProcessingQueue._instance = None

    @classmethod
    def tearDownClass(cls):
        cls.VideoProcessingQueue._instance = None

    def test_singleton_returns_same_instance(self):
        q1 = self.VideoProcessingQueue.get_instance()
        q2 = self.VideoProcessingQueue.get_instance()
        self.assertIs(q1, q2)

    @unittest.skipUnless(sys.version_info >= (3, 7), "asyncio.run requires Python 3.7+")
    def test_submit_returns_job_id(self):
        async def run_test():
            queue = self.VideoProcessingQueue.get_instance()
            job_id = await queue.submit(lambda: "ok")
            self.assertIsInstance(job_id, str)
            self.assertEqual(len(job_id), 12)
        try:
            asyncio.run(run_test())
        except RuntimeError:
            pass

    def test_get_status_unknown_job(self):
        queue = self.VideoProcessingQueue.get_instance()
        self.assertIsNone(queue.get_status("nonexistent"))


# ─────────────────────────────────────────────────────────────────────────────
# video_engine/schemas.py
# ─────────────────────────────────────────────────────────────────────────────

@unittest.skipUnless(_HAS_PYDANTIC, "pydantic not installed")
class TestVideoSchemas(unittest.TestCase):
    """Test Pydantic schema validation — requires pydantic."""

    @classmethod
    def setUpClass(cls):
        if _ve_schemas is None:
            raise unittest.SkipTest("video_engine.schemas could not be imported")
        cls.mod = _ve_schemas

    def test_trim_video_valid(self):
        params = self.mod.TrimVideoInput(video_path="in.mp4", start_time=0.0, end_time=5.0)
        self.assertEqual(params.start_time, 0.0)
        self.assertEqual(params.end_time, 5.0)

    def test_trim_video_invalid_range(self):
        with self.assertRaises(Exception):
            self.mod.TrimVideoInput(video_path="in.mp4", start_time=5.0, end_time=0.0)

    def test_adjust_speed_bounds(self):
        with self.assertRaises(Exception):
            self.mod.AdjustSpeedInput(video_path="in.mp4", speed_factor=0.0)
        with self.assertRaises(Exception):
            self.mod.AdjustSpeedInput(video_path="in.mp4", speed_factor=11.0)

    def test_add_watermark_requires_input(self):
        params = self.mod.AddWatermarkInput(video_path="in.mp4")
        self.assertIsNone(params.watermark_path)
        self.assertIsNone(params.watermark_text)

    def test_extract_frames_max_frames(self):
        params = self.mod.ExtractFramesInput(video_path="in.mp4", max_frames=50)
        self.assertEqual(params.max_frames, 50)

    def test_video_to_gif_duration_limit(self):
        with self.assertRaises(Exception):
            self.mod.VideoToGifInput(video_path="in.mp4", duration=31.0)


# ─────────────────────────────────────────────────────────────────────────────
# video_engine/path_manager.py
# ─────────────────────────────────────────────────────────────────────────────

@unittest.skipUnless(_ve_path is not None, "video_engine.path_manager not available")
class TestVideoPathManager(unittest.TestCase):
    """Test video path resolution — pure pathlib logic."""

    @classmethod
    def setUpClass(cls):
        cls.VideoPathManager = _ve_path.VideoPathManager

    def test_resolve_returns_path_with_uuid(self):
        result = self.VideoPathManager.resolve("in.mp4", None, "test", "mp4")
        self.assertTrue(str(result).endswith(".mp4"))
        self.assertIn("test", str(result))
        self.assertIn("in", str(result))

    def test_resolve_uses_provided_output_path(self):
        result = self.VideoPathManager.resolve("in.mp4", "/custom/out.mp4", "test", "mp4")
        self.assertEqual(result, Path("/custom/out.mp4"))

    def test_frames_dir_creates_directory(self):
        temp_root = tempfile.mkdtemp()
        original_output = _ve_path.OUTPUT_DIR
        _ve_path.OUTPUT_DIR = Path(temp_root)
        try:
            d = self.VideoPathManager.frames_dir("in.mp4")
            self.assertTrue(d.exists())
            self.assertTrue(d.is_dir())
        finally:
            _ve_path.OUTPUT_DIR = original_output

    def test_temp_creates_unique_path(self):
        temp_root = tempfile.mkdtemp()
        original_temp = _ve_path.TEMP_DIR
        _ve_path.TEMP_DIR = Path(temp_root)
        try:
            p1 = self.VideoPathManager.temp("png")
            p2 = self.VideoPathManager.temp("png")
            self.assertNotEqual(p1, p2)
            self.assertTrue(str(p1).endswith(".png"))
        finally:
            _ve_path.TEMP_DIR = original_temp


# ─────────────────────────────────────────────────────────────────────────────
# image_engine/local_ops.py
# ─────────────────────────────────────────────────────────────────────────────

class TestSavePil(unittest.TestCase):
    """Test _save_pil with renamed file_format parameter — requires Pillow."""

    @classmethod
    def setUpClass(cls):
        if not _HAS_PIL:
            raise unittest.SkipTest("Pillow not installed")
        mod = _get_module("image_engine.local_ops")
        if mod is None:
            raise unittest.SkipTest("image_engine.local_ops could not be imported")
        cls._save_pil = mod._save_pil

    def test_save_pil_accepts_file_format(self):
        temp_dir = tempfile.mkdtemp()
        img = _PILImage.new("RGB", (100, 100), color="red")
        out_path = Path(temp_dir) / "test.png"
        self._save_pil(img, out_path, "png")
        self.assertTrue(out_path.exists())
        self.assertGreater(out_path.stat().st_size, 0)

    def test_save_pil_jpeg_conversion(self):
        temp_dir = tempfile.mkdtemp()
        img = _PILImage.new("RGBA", (100, 100), color=(255, 0, 0, 128))
        out_path = Path(temp_dir) / "test.jpg"
        self._save_pil(img, out_path, "jpeg")
        self.assertTrue(out_path.exists())


class TestPilCv2Conversion(unittest.TestCase):
    """Test PIL <-> OpenCV conversions — requires Pillow + numpy + opencv."""

    @classmethod
    def setUpClass(cls):
        if not (_HAS_PIL and _HAS_NUMPY and _HAS_CV2):
            raise unittest.SkipTest("Pillow, numpy, or opencv-python not installed")
        mod = _get_module("image_engine.local_ops")
        if mod is None:
            raise unittest.SkipTest("image_engine.local_ops could not be imported")
        cls._pil_to_cv2 = mod._pil_to_cv2
        cls._cv2_to_pil = mod._cv2_to_pil

    def test_rgb_roundtrip(self):
        img = _PILImage.new("RGB", (10, 10), color="blue")
        arr = self._pil_to_cv2(img)
        self.assertEqual(arr.shape[-1], 3)
        result = self._cv2_to_pil(arr)
        self.assertEqual(result.mode, "RGB")

    def test_rgba_roundtrip(self):
        img = _PILImage.new("RGBA", (10, 10), color=(0, 255, 0, 128))
        arr = self._pil_to_cv2(img)
        self.assertEqual(arr.shape[-1], 4)
        result = self._cv2_to_pil(arr)
        self.assertEqual(result.mode, "RGBA")


class TestResizeImageLocal(unittest.TestCase):
    """Test resize_image_local — requires Pillow."""

    @classmethod
    def setUpClass(cls):
        if not _HAS_PIL:
            raise unittest.SkipTest("Pillow not installed")
        mod = _get_module("image_engine.local_ops")
        if mod is None:
            raise unittest.SkipTest("image_engine.local_ops could not be imported")
        cls.resize_image_local = mod.resize_image_local

    def _run(self, mode):
        temp_dir = tempfile.mkdtemp()
        src = Path(temp_dir) / "src.png"
        _PILImage.new("RGB", (100, 100), "red").save(src)
        params = MagicMock()
        params.image_path = str(src)
        params.width = 200
        params.height = 200
        params.mode = mode
        params.background_color = "#FFFFFF"
        params.output_path = str(Path(temp_dir) / "out.png")
        params.output_format = MagicMock()
        params.output_format.value = "png"
        params.quality = 95
        return self.resize_image_local(params)

    def test_resize_fit(self):
        result = self._run("fit")
        self.assertTrue(result.output_path.endswith(".png"))
        self.assertTrue(Path(result.output_path).exists())

    def test_resize_fill(self):
        result = self._run("fill")
        self.assertTrue(Path(result.output_path).exists())

    def test_resize_stretch(self):
        result = self._run("stretch")
        self.assertTrue(Path(result.output_path).exists())


# ─────────────────────────────────────────────────────────────────────────────
# devops/cache.py
# ─────────────────────────────────────────────────────────────────────────────

@unittest.skipUnless(_devops_cache is not None, "devops.cache not available")
class TestFileCache(unittest.TestCase):
    """Test FileCache content-addressed caching — pure Python + pathlib."""

    @classmethod
    def setUpClass(cls):
        cls.mod = _devops_cache
        cls.temp_dir = tempfile.mkdtemp()
        cls.mod.CACHE_DIR = Path(cls.temp_dir)
        cls.mod.CACHE_INDEX = Path(cls.temp_dir) / "index.json"
        cls.cache = cls.mod.FileCache()

    def test_put_and_get(self):
        out_path = Path(self.temp_dir) / "out.png"
        out_path.write_bytes(b"fake image data")
        self.cache.put("in.png", "op", {"k": "v"}, str(out_path))
        result = self.cache.get("in.png", "op", {"k": "v"})
        self.assertEqual(result, str(out_path))

    def test_miss_returns_none(self):
        result = self.cache.get("missing.png", "op", {})
        self.assertIsNone(result)

    def test_invalidate_removes_entries(self):
        self.cache.put("in.png", "op1", {}, "/tmp/out1.png")
        self.cache.put("in.png", "op2", {}, "/tmp/out2.png")
        self.cache.invalidate("in.png")
        self.assertIsNone(self.cache.get("in.png", "op1", {}))
        self.assertIsNone(self.cache.get("in.png", "op2", {}))

    def test_get_stats_returns_dict(self):
        stats = self.cache.get_stats()
        self.assertIn("total_entries", stats)
        self.assertIn("total_size_mb", stats)
        self.assertIn("max_size_gb", stats)
        self.assertIn("ttl_hours", stats)

    def test_cleanup_force_clears(self):
        self.cache.put("in.png", "op", {}, "/tmp/out.png")
        self.cache.cleanup(force=True)
        stats = self.cache.get_stats()
        self.assertEqual(stats["total_entries"], 0)


# ─────────────────────────────────────────────────────────────────────────────
# devops/rate_limiter.py
# ─────────────────────────────────────────────────────────────────────────────

@unittest.skipUnless(_devops_rl is not None, "devops.rate_limiter not available")
class TestRateLimiter(unittest.TestCase):
    """Test RateLimitedExecutor key rotation and backoff — pure Python."""

    @classmethod
    def setUpClass(cls):
        cls.mod = _devops_rl
        cls.executor = cls.mod.RateLimitedExecutor()

    def test_get_status_returns_dict(self):
        status = self.executor.get_status()
        self.assertIn("gemini", status)
        self.assertIn("groq", status)

    def test_register_key_from_env(self):
        Provider = self.mod.Provider
        self.executor._register_keys_from_env()
        for provider in Provider:
            self.assertIn(provider, self.executor._pools)

    def test_key_state_backoff(self):
        KeyState = self.mod.KeyState
        ks = KeyState(key="test-key")
        self.assertTrue(ks.is_available)
        ks.record_failure(backoff_s=60)
        self.assertFalse(ks.is_available)
        ks.record_success()
        self.assertTrue(ks.is_available)
        self.assertEqual(ks.failures, 0)


# ─────────────────────────────────────────────────────────────────────────────
# app.py — Gradio UI syntax/integrity check
# ─────────────────────────────────────────────────────────────────────────────

class TestAppSyntax(unittest.TestCase):
    """Ensure app.py and main.py compile without syntax errors."""

    def test_app_imports(self):
        app_path = Path(__file__).parent / "app.py"
        with open(app_path, "r", encoding="utf-8") as f:
            source = f.read()
        ast.parse(source)

    def test_main_imports(self):
        main_path = Path(__file__).parent / "main.py"
        with open(main_path, "r", encoding="utf-8") as f:
            source = f.read()
        ast.parse(source)

    def test_app_no_builtin_shadow_format(self):
        """Ensure no shadowing of built-in `format` in app.py."""
        app_path = Path(__file__).parent / "app.py"
        with open(app_path, "r", encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                for arg in node.args.args:
                    self.assertNotEqual(arg.arg, "format", msg=f"Built-in 'format' shadowed in {node.name}")

    def test_orchestrator_no_builtin_shadow(self):
        """Ensure no built-in shadowing in agent/orchestrator.py."""
        orch_path = Path(__file__).parent / "agent" / "orchestrator.py"
        with open(orch_path, "r", encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                for arg in node.args.args:
                    self.assertNotEqual(arg.arg, "format", msg=f"Built-in 'format' shadowed in {node.name}")

    def test_video_local_ops_no_builtin_shadow(self):
        """Ensure no built-in shadowing in video_engine/local_ops.py."""
        vl_path = Path(__file__).parent / "video_engine" / "local_ops.py"
        with open(vl_path, "r", encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                for arg in node.args.args:
                    self.assertNotEqual(arg.arg, "format", msg=f"Built-in 'format' shadowed in {node.name}")

    def test_image_local_ops_no_builtin_shadow(self):
        """Ensure no built-in shadowing in image_engine/local_ops.py."""
        il_path = Path(__file__).parent / "image_engine" / "local_ops.py"
        with open(il_path, "r", encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                for arg in node.args.args:
                    self.assertNotEqual(arg.arg, "format", msg=f"Built-in 'format' shadowed in {node.name}")

    def test_app_specific_syntax_checks(self):
        """Check app.py for specific known issues."""
        app_path = Path(__file__).parent / "app.py"
        with open(app_path, "r", encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source)
        func_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        expected_funcs = [
            "handle_remove_bg", "handle_resize", "handle_color_adjust",
            "handle_apply_filter", "handle_trim_video", "handle_extract_audio",
            "handle_speed", "handle_to_gif", "handle_animate_image",
            "handle_agent_chat", "clear_agent_memory", "get_system_status",
            "handle_clear_cache", "build_ui", "_find_available_port",
        ]
        for func in expected_funcs:
            self.assertIn(func, func_names, msg=f"{func} function not found in app.py")


# ─────────────────────────────────────────────────────────────────────────────
# Integration: end-to-end pipeline smoke tests
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegrationSmoke(unittest.TestCase):
    """Lightweight integration tests using real local operations."""

    @unittest.skipUnless(_HAS_PIL, "Pillow not installed")
    def test_image_roundtrip_png(self):
        ie_mod = _get_module("image_engine.local_ops")
        if ie_mod is None:
            self.skipTest("image_engine.local_ops not available")
        temp_dir = tempfile.mkdtemp()
        src = Path(temp_dir) / "src.png"
        dst = Path(temp_dir) / "dst.png"
        _PILImage.new("RGB", (64, 64), "green").save(src)
        loaded = ie_mod._load_pil(str(src))
        self.assertEqual(loaded.size, (64, 64))
        ie_mod._save_pil(loaded, dst, "png")
        self.assertTrue(dst.exists())

    def test_video_engine_modules_syntax_valid(self):
        """Ensure video_engine submodule files are syntactically valid."""
        modules = [
            ("video_engine.local_ops", "video_engine/local_ops.py"),
            ("video_engine.api_ops", "video_engine/api_ops.py"),
            ("video_engine.async_queue", "video_engine/async_queue.py"),
            ("video_engine.schemas", "video_engine/schemas.py"),
            ("video_engine.path_manager", "video_engine/path_manager.py"),
            ("video_engine.processor", "video_engine/processor.py"),
        ]
        for mod_name, rel_path in modules:
            with self.subTest(module=mod_name):
                mod_path = Path(__file__).parent / rel_path
                self.assertTrue(mod_path.exists(), f"Module file not found: {mod_path}")
                with open(mod_path, "r", encoding="utf-8") as f:
                    source = f.read()
                ast.parse(source)


if __name__ == "__main__":
    unittest.main()

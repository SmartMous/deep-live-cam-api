"""High-level wrapper around Deep-Live-Cam core functions.

Imports from DLC are done lazily to avoid blocking API startup
when DLC dependencies (onnxruntime, torch, etc.) are not installed.
"""

import os
import sys
from typing import Optional, Tuple

import cv2
import numpy as np
import config

# Ensure DLC is in path (but imports are deferred)
DLC_PATH = config.DEEP_LIVE_CAM_PATH
if DLC_PATH not in sys.path:
    sys.path.insert(0, DLC_PATH)

# Patch PATH for bundled ffmpeg
os.environ["PATH"] = DLC_PATH + os.pathsep + os.environ.get("PATH", "")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="insightface")

# ---------- Lazy DLC imports ----------
_dlc_core = None
_dlc_globals = None


def _ensure_dlc():
    """Lazy-load DLC modules. Raises ImportError if unavailable."""
    global _dlc_core, _dlc_globals
    if _dlc_core is None:
        try:
            from modules import core as _c
            import modules.globals as _g
            _dlc_core = _c
            _dlc_globals = _g
        except ImportError as e:
            raise ImportError(
                "Deep-Live-Cam modules not importable. "
                "Please ensure DEEP_LIVE_CAM_PATH is correct and DLC requirements are installed."
            ) from e


def _ensure_initialized():
    """Ensure DLC globals are minimally initialized."""
    _ensure_dlc()
    if not _dlc_globals.execution_providers:
        _dlc_globals.execution_providers = _dlc_core.decode_execution_providers(
            [_dlc_core.suggest_default_execution_provider()]
        )
    if _dlc_globals.execution_threads is None:
        _dlc_globals.execution_threads = _dlc_core.suggest_execution_threads()


def set_source_path(path: Optional[str]):
    _ensure_dlc()
    _dlc_globals.source_path = path


def set_target_path(path: Optional[str]):
    _ensure_dlc()
    _dlc_globals.target_path = path


def set_output_path(path: str):
    _ensure_dlc()
    _dlc_globals.output_path = path


def get_source_path() -> Optional[str]:
    _ensure_dlc()
    return _dlc_globals.source_path


def get_target_path() -> Optional[str]:
    _ensure_dlc()
    return _dlc_globals.target_path


def get_output_path() -> Optional[str]:
    _ensure_dlc()
    return _dlc_globals.output_path


def swap_source_target() -> Tuple[Optional[str], Optional[str]]:
    """Swap source_path and target_path. Returns (new_source, new_target)."""
    _ensure_dlc()
    old_source = _dlc_globals.source_path
    old_target = _dlc_globals.target_path
    _dlc_globals.source_path = old_target
    _dlc_globals.target_path = old_source
    return old_target, old_source


def configure(
    execution_provider: Optional[str] = None,
    keep_fps: bool = True,
    keep_audio: bool = True,
    many_faces: bool = False,
    map_faces: bool = False,
    nsfw_filter: bool = False,
    mouth_mask: bool = False,
    face_enhancer: bool = False,
    video_encoder: str = "libx264",
    video_quality: int = 18,
) -> None:
    """Configure DLC processing options."""
    _ensure_initialized()

    if execution_provider:
        decoded = _dlc_core.decode_execution_providers([execution_provider])
        _dlc_globals.execution_providers = decoded

    _dlc_globals.keep_fps = keep_fps
    _dlc_globals.keep_audio = keep_audio
    _dlc_globals.many_faces = many_faces
    _dlc_globals.map_faces = map_faces
    _dlc_globals.nsfw_filter = nsfw_filter
    _dlc_globals.mouth_mask = mouth_mask
    _dlc_globals.video_encoder = video_encoder
    _dlc_globals.video_quality = video_quality

    fps = ["face_swapper"]
    if face_enhancer:
        fps.append("face_enhancer")
    _dlc_globals.frame_processors = fps
    for k in ("face_enhancer", "face_enhancer_gpen256", "face_enhancer_gpen512"):
        _dlc_globals.fp_ui[k] = k in fps


def process_image(source_path: str, target_path: str, output_path: str) -> bool:
    """Process a single image face swap. Returns True on success."""
    _ensure_initialized()
    _dlc_globals.source_path = source_path
    _dlc_globals.target_path = target_path
    _dlc_globals.output_path = output_path
    _dlc_globals.headless = True

    from modules.processors.frame.core import get_frame_processors_modules

    frame_processors = get_frame_processors_modules(_dlc_globals.frame_processors)
    for fp in frame_processors:
        fp.pre_start()

    success = False
    for fp in frame_processors:
        try:
            fp.process_image(source_path, output_path, output_path)
            success = True
        except Exception as e:
            print(f"[DLC Wrapper] process_image error in {fp.NAME}: {e}")
            success = False
            break

    _dlc_core.release_resources()
    return success and os.path.exists(output_path)


def process_video(source_path: str, target_path: str, output_path: str) -> bool:
    """Process a video face swap. Returns True on success."""
    _ensure_initialized()
    _dlc_globals.source_path = source_path
    _dlc_globals.target_path = target_path
    _dlc_globals.output_path = output_path
    _dlc_globals.headless = True

    from modules.processors.frame.core import process_video_in_memory
    from modules.utilities import detect_fps

    if _dlc_globals.keep_fps:
        fps = detect_fps(target_path)
    else:
        fps = 30.0

    video_created = process_video_in_memory(source_path, target_path, fps)
    if not video_created:
        return False

    from modules.utilities import get_temp_output_path, move_temp, clean_temp

    temp_out = get_temp_output_path(target_path)
    if os.path.exists(temp_out):
        if _dlc_globals.keep_audio:
            from modules.utilities import restore_audio
            restore_audio(target_path, output_path)
        else:
            move_temp(target_path, output_path)
        clean_temp(target_path)
        return True

    return False


def get_config() -> dict:
    """Return current DLC config as a dict."""
    _ensure_initialized()
    return {
        "execution_provider": _dlc_core.encode_execution_providers(
            _dlc_globals.execution_providers
        )[0] if _dlc_globals.execution_providers else None,
        "max_memory": _dlc_globals.max_memory,
        "execution_threads": _dlc_globals.execution_threads,
        "keep_fps": _dlc_globals.keep_fps,
        "keep_audio": _dlc_globals.keep_audio,
        "many_faces": _dlc_globals.many_faces,
        "map_faces": _dlc_globals.map_faces,
        "nsfw_filter": _dlc_globals.nsfw_filter,
        "video_encoder": _dlc_globals.video_encoder,
        "video_quality": _dlc_globals.video_quality,
        "mouth_mask": _dlc_globals.mouth_mask,
        "frame_processors": list(_dlc_globals.frame_processors),
    }


def get_available_providers() -> list:
    """Return available execution providers."""
    import onnxruntime
    _ensure_dlc()
    return _dlc_core.encode_execution_providers(
        onnxruntime.get_available_providers()
    )


def get_default_provider() -> str:
    """Return the default execution provider."""
    _ensure_dlc()
    return _dlc_core.suggest_default_execution_provider()

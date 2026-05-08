"""High-level wrapper around Deep-Live-Cam core functions.

All imports from DLC are done lazily to avoid startup overhead.
"""

import os
import sys
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np

# Ensure DLC is in path
import config
DLC_PATH = config.DEEP_LIVE_CAM_PATH
if DLC_PATH not in sys.path:
    sys.path.insert(0, DLC_PATH)

# Patch PATH for bundled ffmpeg
os.environ["PATH"] = DLC_PATH + os.pathsep + os.environ.get("PATH", "")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="insightface")

import modules.globals as dlc_globals
from modules import core as dlc_core
from modules.utilities import normalize_output_path, has_image_extension, is_image, is_video


def ensure_initialized() -> None:
    """Ensure DLC globals are minimally initialized."""
    if not dlc_globals.execution_providers:
        dlc_globals.execution_providers = dlc_core.decode_execution_providers(
            [dlc_core.suggest_default_execution_provider()]
        )
    if dlc_globals.execution_threads is None:
        dlc_globals.execution_threads = dlc_core.suggest_execution_threads()


def set_source_path(path: Optional[str]) -> None:
    dlc_globals.source_path = path


def set_target_path(path: Optional[str]) -> None:
    dlc_globals.target_path = path


def set_output_path(path: str) -> None:
    dlc_globals.output_path = path


def get_source_path() -> Optional[str]:
    return dlc_globals.source_path


def get_target_path() -> Optional[str]:
    return dlc_globals.target_path


def get_output_path() -> Optional[str]:
    return dlc_globals.output_path


def swap_source_target() -> Tuple[Optional[str], Optional[str]]:
    """Swap source_path and target_path. Returns (new_source, new_target)."""
    old_source = dlc_globals.source_path
    old_target = dlc_globals.target_path
    dlc_globals.source_path = old_target
    dlc_globals.target_path = old_source
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
    ensure_initialized()

    if execution_provider:
        decoded = dlc_core.decode_execution_providers([execution_provider])
        dlc_globals.execution_providers = decoded

    dlc_globals.keep_fps = keep_fps
    dlc_globals.keep_audio = keep_audio
    dlc_globals.many_faces = many_faces
    dlc_globals.map_faces = map_faces
    dlc_globals.nsfw_filter = nsfw_filter
    dlc_globals.mouth_mask = mouth_mask
    dlc_globals.video_encoder = video_encoder
    dlc_globals.video_quality = video_quality

    fps = ["face_swapper"]
    if face_enhancer:
        fps.append("face_enhancer")
    dlc_globals.frame_processors = fps
    for k in ("face_enhancer", "face_enhancer_gpen256", "face_enhancer_gpen512"):
        dlc_globals.fp_ui[k] = k in fps


def process_image(source_path: str, target_path: str, output_path: str) -> bool:
    """Process a single image face swap. Returns True on success."""
    ensure_initialized()
    dlc_globals.source_path = source_path
    dlc_globals.target_path = target_path
    dlc_globals.output_path = output_path
    dlc_globals.headless = True

    # Import processors
    from modules.processors.frame.core import get_frame_processors_modules

    frame_processors = get_frame_processors_modules(dlc_globals.frame_processors)
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

    dlc_core.release_resources()
    return success and os.path.exists(output_path)


def process_video(source_path: str, target_path: str, output_path: str) -> bool:
    """Process a video face swap. Returns True on success."""
    ensure_initialized()
    dlc_globals.source_path = source_path
    dlc_globals.target_path = target_path
    dlc_globals.output_path = output_path
    dlc_globals.headless = True

    # Use the in-memory pipeline directly
    from modules.processors.frame.core import process_video_in_memory
    from modules.utilities import detect_fps

    if dlc_globals.keep_fps:
        fps = detect_fps(target_path)
    else:
        fps = 30.0

    video_created = process_video_in_memory(source_path, target_path, fps)
    if not video_created:
        return False

    # Move temp output to final destination
    from modules.utilities import get_temp_output_path, move_temp, clean_temp

    temp_out = get_temp_output_path(target_path)
    if os.path.exists(temp_out):
        if dlc_globals.keep_audio:
            from modules.utilities import restore_audio
            restore_audio(target_path, output_path)
        else:
            move_temp(target_path, output_path)
        clean_temp(target_path)
        return True

    return False


def get_config() -> dict:
    """Return current DLC config as a dict."""
    ensure_initialized()
    return {
        "execution_provider": dlc_core.encode_execution_providers(
            dlc_globals.execution_providers
        )[0] if dlc_globals.execution_providers else None,
        "max_memory": dlc_globals.max_memory,
        "execution_threads": dlc_globals.execution_threads,
        "keep_fps": dlc_globals.keep_fps,
        "keep_audio": dlc_globals.keep_audio,
        "many_faces": dlc_globals.many_faces,
        "map_faces": dlc_globals.map_faces,
        "nsfw_filter": dlc_globals.nsfw_filter,
        "video_encoder": dlc_globals.video_encoder,
        "video_quality": dlc_globals.video_quality,
        "mouth_mask": dlc_globals.mouth_mask,
        "frame_processors": list(dlc_globals.frame_processors),
    }


def get_available_providers() -> List[str]:
    """Return available execution providers."""
    return dlc_core.encode_execution_providers(
        __import__("onnxruntime").get_available_providers()
    )


def get_default_provider() -> str:
    """Return the default execution provider."""
    return dlc_core.suggest_default_execution_provider()

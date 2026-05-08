"""Deep-Live-Cam core integration utility.

Handles path setup, module importing, and globals manipulation.
DOES NOT modify any file under DEEP_LIVE_CAM_PATH.
"""

import os
import sys
from typing import Optional

import config

# Ensure Deep-Live-Cam modules are importable
DLC_PATH = config.DEEP_LIVE_CAM_PATH
if DLC_PATH not in sys.path:
    sys.path.insert(0, DLC_PATH)

# Fix ffmpeg PATH like run.py does
os.environ["PATH"] = DLC_PATH + os.pathsep + os.environ.get("PATH", "")

# Windows CUDA DLL fix (same as Deep-Live-Cam/run.py)
if sys.platform == "win32":
    import site
    project_root = DLC_PATH
    _site_packages = os.path.join(site.getprefix(), "Lib", "site-packages")
    _venv_site_packages = os.path.join(project_root, "venv", "Lib", "site-packages")
    for _sp in (_site_packages, _venv_site_packages):
        _torch_lib = os.path.join(_sp, "torch", "lib")
        if os.path.isdir(_torch_lib):
            os.environ["PATH"] = _torch_lib + os.pathsep + os.environ["PATH"]
        _nvidia_dir = os.path.join(_sp, "nvidia")
        if os.path.isdir(_nvidia_dir):
            for _pkg in os.listdir(_nvidia_dir):
                _bin_dir = os.path.join(_nvidia_dir, _pkg, "bin")
                if os.path.isdir(_bin_dir):
                    os.environ["PATH"] = _bin_dir + os.pathsep + os.environ["PATH"]

# Suppress TF logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="insightface")

import onnxruntime
from modules import core as dlc_core
import modules.globals as dlc_globals


# ---------- Helpers ----------
def get_available_providers() -> list[str]:
    """Return decoded (short-form) available execution providers."""
    return dlc_core.encode_execution_providers(
        onnxruntime.get_available_providers()
    )


def suggest_default_provider() -> str:
    return dlc_core.suggest_default_execution_provider()


def apply_config(
    execution_provider: Optional[str] = None,
    max_memory: Optional[int] = None,
    execution_threads: Optional[int] = None,
    keep_fps: bool = True,
    keep_audio: bool = True,
    many_faces: bool = False,
    map_faces: bool = False,
    nsfw_filter: bool = False,
    video_encoder: str = "libx264",
    video_quality: int = 18,
    mouth_mask: bool = False,
    face_enhancer: bool = False,
) -> None:
    """Apply processing configuration to DLC globals."""
    if execution_provider:
        decoded = dlc_core.decode_execution_providers([execution_provider])
        dlc_globals.execution_providers = decoded
    if max_memory is not None:
        dlc_globals.max_memory = max_memory
    if execution_threads is not None:
        dlc_globals.execution_threads = execution_threads

    dlc_globals.keep_fps = keep_fps
    dlc_globals.keep_audio = keep_audio
    dlc_globals.many_faces = many_faces
    dlc_globals.map_faces = map_faces
    dlc_globals.nsfw_filter = nsfw_filter
    dlc_globals.video_encoder = video_encoder
    dlc_globals.video_quality = video_quality
    dlc_globals.mouth_mask = mouth_mask

    # Frame processors
    fps = ["face_swapper"]
    if face_enhancer:
        fps.append("face_enhancer")
    dlc_globals.frame_processors = fps
    for k in ("face_enhancer", "face_enhancer_gpen256", "face_enhancer_gpen512"):
        dlc_globals.fp_ui[k] = (k in fps)


def get_globals() -> "modules.globals":
    """Return the DLC globals module for direct access."""
    return dlc_globals

"""Deep-Live-Cam core integration utility.

Handles path setup, module importing, and globals manipulation.
DOES NOT modify any file under DEEP_LIVE_CAM_PATH.

All DLC-dependent imports are LAZY — the API starts even without onnxruntime/DLC installed.
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

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="insightface")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ---------- Lazy DLC imports (deferred until first use) ----------
_dlc_core = None
_dlc_globals = None
_onnxruntime = None


def _ensure_dlc():
    """Lazy-load DLC + onnxruntime on first call. Raises ImportError if unavailable."""
    global _dlc_core, _dlc_globals, _onnxruntime
    if _dlc_core is None:
        try:
            import onnxruntime as _ort
            from modules import core as _c
            import modules.globals as _g
        except ImportError as e:
            raise ImportError(
                "onnxruntime or Deep-Live-Cam modules not found. "
                "Please install Deep-Live-Cam dependencies: "
                "pip install onnxruntime-gpu && pip install -r /path/to/Deep-Live-Cam/requirements.txt"
            ) from e
        _onnxruntime = _ort
        _dlc_core = _c
        _dlc_globals = _g


def get_available_providers() -> list:
    """Return decoded (short-form) available execution providers."""
    _ensure_dlc()
    return _dlc_core.encode_execution_providers(
        _onnxruntime.get_available_providers()
    )


def suggest_default_provider() -> str:
    _ensure_dlc()
    return _dlc_core.suggest_default_execution_provider()


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
    _ensure_dlc()
    if execution_provider:
        decoded = _dlc_core.decode_execution_providers([execution_provider])
        _dlc_globals.execution_providers = decoded
    if max_memory is not None:
        _dlc_globals.max_memory = max_memory
    if execution_threads is not None:
        _dlc_globals.execution_threads = execution_threads
    _dlc_globals.keep_fps = keep_fps
    _dlc_globals.keep_audio = keep_audio
    _dlc_globals.many_faces = many_faces
    _dlc_globals.map_faces = map_faces
    _dlc_globals.nsfw_filter = nsfw_filter
    _dlc_globals.video_encoder = video_encoder
    _dlc_globals.video_quality = video_quality
    _dlc_globals.mouth_mask = mouth_mask
    # Frame processors
    fps = ["face_swapper"]
    if face_enhancer:
        fps.append("face_enhancer")
    _dlc_globals.frame_processors = fps
    for k in ("face_enhancer", "face_enhancer_gpen256", "face_enhancer_gpen512"):
        _dlc_globals.fp_ui[k] = (k in fps)


def get_globals():
    """Return the DLC globals module for direct access."""
    _ensure_dlc()
    return _dlc_globals


def process_image(source_path: str, target_path: str, output_path: str) -> bool:
    """Process a single image face swap."""
    _ensure_dlc()
    from modules.processors.frame.core import get_frame_processors_modules
    for fp in get_frame_processors_modules(_dlc_globals.frame_processors):
        fp.process_image(source_path, target_path, output_path)
    return os.path.exists(output_path)

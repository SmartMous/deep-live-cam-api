"""Configuration management for Deep-Live-Cam API."""

import os
from pathlib import Path
from typing import Optional, List
from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
DEEP_LIVE_CAM_PATH: str = os.getenv("DEEP_LIVE_CAM_PATH", "/tmp/Deep-Live-Cam")
OUTPUT_DIR: Path = Path(os.getenv("OUTPUT_DIR", "/tmp/deep-live-cam-api/outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- CORS ---
CORS_ORIGINS: str = os.getenv("CORS_ORIGINS", "*")

# --- Deep-Live-Cam config defaults ---
DEFAULT_EXECUTION_PROVIDER: str = os.getenv("EXECUTION_PROVIDER", "cuda")
DEFAULT_MAX_MEMORY: Optional[int] = None
DEFAULT_EXECUTION_THREADS: Optional[int] = None
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "error")

# --- Default processing options ---
DEFAULT_KEEP_FPS: bool = True
DEFAULT_KEEP_AUDIO: bool = True
DEFAULT_MANY_FACES: bool = False
DEFAULT_MAP_FACES: bool = False
DEFAULT_NSfw_FILTER: bool = False
DEFAULT_VIDEO_ENCODER: str = "libx264"
DEFAULT_VIDEO_QUALITY: int = 18

# --- Server ---
HOST: str = "0.0.0.0"
PORT: int = 7860

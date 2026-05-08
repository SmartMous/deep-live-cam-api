"""File handling utilities: base64, URL, and local path support."""

import base64
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import httpx
from PIL import Image


# ---------- IO helpers ----------

def decode_base64_image(b64_str: str) -> Optional[np.ndarray]:
    """Decode a base64-encoded image string to an OpenCV image."""
    try:
        # Strip data URI prefix if present
        if "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]
        data = base64.b64decode(b64_str)
        nparr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def encode_image_to_base64(img: np.ndarray, format: str = ".png") -> str:
    """Encode an OpenCV image to a base64 string."""
    _, buf = cv2.imencode(format, img)
    return base64.b64encode(buf).decode("utf-8")


def download_url_to_path(url: str, dest_path: str, timeout: int = 30) -> bool:
    """Download a remote URL to a local file path."""
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.get(url, follow_redirects=True)
            response.raise_for_status()
            with open(dest_path, "wb") as f:
                f.write(response.content)
        return True
    except Exception:
        return False


def is_url(text: str) -> bool:
    """Check if a string looks like a URL."""
    return bool(re.match(r"^https?://", text.strip()))


def is_base64(text: str) -> bool:
    """Check if a string looks like base64-encoded data."""
    if not text:
        return False
    text = text.strip()
    if text.startswith("data:image"):
        return True
    if len(text) < 100:
        return False
    try:
        if "," in text:
            text = text.split(",", 1)[1]
        base64.b64decode(text, validate=True)
        return True
    except Exception:
        return False


def resolve_input(value: str, temp_dir: str, prefix: str = "input") -> Optional[str]:
    """Resolve an input value (URL, base64, or local path) to a local file path.
    
    Returns a local file path, or None if the input cannot be resolved.
    """
    value = value.strip()

    if is_url(value):
        suffix = Path(value).suffix or ".tmp"
        dest = os.path.join(temp_dir, f"{prefix}{suffix}")
        if download_url_to_path(value, dest):
            return dest
        return None

    if is_base64(value):
        img = decode_base64_image(value)
        if img is None:
            return None
        dest = os.path.join(temp_dir, f"{prefix}.png")
        cv2.imwrite(dest, img)
        return dest

    # Assume local path
    if os.path.isfile(value):
        return value

    return None


def save_output_image(img: np.ndarray, output_dir: str, name: str) -> str:
    """Save an image to the output directory."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, name)
    cv2.imwrite(path, img)
    return path


def get_temp_dir() -> str:
    """Return a temporary directory that persists until explicitly cleaned."""
    d = tempfile.mkdtemp(prefix="dlc_api_")
    return d


def clean_temp_dir(path: str) -> None:
    """Recursively remove a temporary directory."""
    shutil.rmtree(path, ignore_errors=True)


# ---------- Image helpers ----------

def load_image(path: str) -> Optional[np.ndarray]:
    """Load an image from file path."""
    if not os.path.isfile(path):
        return None
    img = cv2.imread(path)
    return img


def image_to_bytes(img: np.ndarray, fmt: str = ".png") -> bytes:
    """Encode an image to bytes."""
    _, buf = cv2.imencode(fmt, img)
    return buf.tobytes()


def pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
    """Convert PIL Image to OpenCV BGR format."""
    rgb = np.array(pil_img)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def cv2_to_pil(img: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR image to PIL Image."""
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

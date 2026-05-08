"""Face detection and swapping service.

DLC imports are lazy — functions fail only when called, not when imported.
"""

import os
import sys
import base64
from typing import List, Optional

import cv2
import numpy as np

import config
sys.path.insert(0, config.DEEP_LIVE_CAM_PATH)

# ---------- Lazy import helpers ----------
def _get_face_analyser():
    from modules.face_analyser import (
        get_one_face, get_many_faces,
        detect_one_face_fast, detect_many_faces_fast,
    )
    return get_one_face, get_many_faces, detect_one_face_fast, detect_many_faces_fast


def detect_faces(image: np.ndarray) -> List[dict]:
    """Detect all faces in an image. Returns list of face dicts."""
    _, _, _, detect_many_faces_fast = _get_face_analyser()
    faces = detect_many_faces_fast(image)
    if faces is None:
        return []

    result = []
    for face in faces:
        bbox = face.bbox.tolist() if hasattr(face, "bbox") else None
        score = float(face.det_score) if hasattr(face, "det_score") else 0.0
        item = {"bbox": bbox, "confidence": score, "landmark_106": None}
        if hasattr(face, "kps") and face.kps is not None:
            item["landmark_106"] = face.kps.tolist()
        elif hasattr(face, "landmark_2d_106") and face.landmark_2d_106 is not None:
            item["landmark_106"] = face.landmark_2d_106.tolist()
        result.append(item)
    return result


def swap_image(
    source_path: str,
    target_path: str,
    output_path: str,
    face_enhancer: bool = False,
    mouth_mask: bool = False,
) -> bool:
    """Perform face swap on a single image."""
    from core import wrapper
    wrapper.configure(
        keep_fps=True,
        keep_audio=True,
        many_faces=False,
        face_enhancer=face_enhancer,
        mouth_mask=mouth_mask,
    )
    return wrapper.process_image(source_path, target_path, output_path)


def swap_image_return_base64(
    source_path: str,
    target_path: str,
    face_enhancer: bool = False,
    mouth_mask: bool = False,
) -> Optional[str]:
    """Perform face swap and return result as base64."""
    from core import wrapper
    wrapper.configure(
        face_enhancer=face_enhancer,
        mouth_mask=mouth_mask,
    )
    output_dir = str(config.OUTPUT_DIR)
    name = f"swap_{os.urandom(8).hex()}.png"
    output_path = os.path.join(output_dir, name)
    success = wrapper.process_image(source_path, target_path, output_path)
    if not success:
        return None
    img = cv2.imread(output_path)
    if img is None:
        return None
    _, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf).decode("utf-8")

"""Live webcam face-swap session manager.

DLC imports are lazy — functions fail only when called, not when imported.
"""

import os
import sys
import uuid
import threading
import time
from typing import Optional
from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np

import config
sys.path.insert(0, config.DEEP_LIVE_CAM_PATH)


class LiveSessionStatus(Enum):
    PENDING = "pending"
    ACTIVE = "active"
    STOPPED = "stopped"


@dataclass
class LiveSession:
    session_id: str
    source_path: str
    camera_index: int
    status: LiveSessionStatus = LiveSessionStatus.PENDING
    capture: Optional[object] = None
    thread: Optional[threading.Thread] = None
    frame_count: int = 0
    started_at: float = field(default_factory=time.time)


class LiveSessionManager:
    def __init__(self):
        self._sessions: dict[str, LiveSession] = {}
        self._lock = threading.Lock()

    def create_session(
        self,
        source_path: str,
        camera_index: int = 0,
        face_enhancer: bool = False,
        mouth_mask: bool = False,
    ) -> LiveSession:
        session_id = f"session_{uuid.uuid4().hex[:8]}"
        session = LiveSession(
            session_id=session_id,
            source_path=source_path,
            camera_index=camera_index,
        )
        with self._lock:
            self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[LiveSession]:
        return self._sessions.get(session_id)

    def list_sessions(self) -> list:
        return [
            {
                "session_id": s.session_id,
                "status": s.status.value,
                "camera_index": s.camera_index,
                "frame_count": s.frame_count,
            }
            for s in self._sessions.values()
        ]

    def stop_session(self, session_id: str) -> bool:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return False
            session.status = LiveSessionStatus.STOPPED
            if session.capture is not None:
                session.capture.release()
            return True


live_manager = LiveSessionManager()


def generate_preview_frame(source_path: str, frame: np.ndarray) -> np.ndarray:
    """Apply face swap to a single frame for preview."""
    from core import wrapper
    from modules.face_analyser import get_one_face
    import tempfile

    # Save current frame to temp file
    fd, temp_path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    cv2.imwrite(temp_path, frame)

    fd2, out_path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd2)

    try:
        wrapper.configure(face_enhancer=False, mouth_mask=False)
        ok = wrapper.process_image(source_path, temp_path, out_path)
        if ok and os.path.exists(out_path):
            return cv2.imread(out_path)
    finally:
        try:
            os.unlink(temp_path)
            os.unlink(out_path)
        except OSError:
            pass

    return frame  # fallback: return original frame

"""Live webcam face swapping service via WebSocket."""

import base64
import io
import sys
import threading
import time
from typing import Optional, Set

import cv2
import numpy as np

import config
sys.path.insert(0, config.DEEP_LIVE_CAM_PATH)

from modules.face_analyser import get_one_face, get_many_faces
from modules.processors.frame.face_swapper import FACE_SWAPPER


class LiveSession:
    """Manages a live face-swap webcam session."""

    def __init__(
        self,
        session_id: str,
        source_path: str,
        camera_index: int = 0,
        face_enhancer: bool = False,
        mouth_mask: bool = False,
    ):
        self.session_id = session_id
        self.source_path = source_path
        self.camera_index = camera_index
        self.face_enhancer = face_enhancer
        self.mouth_mask = mouth_mask
        self.running = False
        self._thread: Optional[threading.Thread] = None
        self._clients: Set = set()
        self._lock = threading.Lock()

    def add_client(self, websocket):
        with self._lock:
            self._clients.add(websocket)

    def remove_client(self, websocket):
        with self._lock:
            self._clients.discard(websocket)

    def broadcast(self, data: bytes):
        with self._lock:
            dead = set()
            for client in self._clients:
                try:
                    import asyncio
                    asyncio.create_task(client.send_bytes(data))
                except Exception:
                    dead.add(client)
            self._clients -= dead

    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=3)

    def _capture_loop(self):
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            return

        source_img = cv2.imread(self.source_path)
        if source_img is None:
            cap.release()
            return

        try:
            source_face = get_one_face(source_img)
        except Exception:
            source_face = None

        frame_id = 0
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

            try:
                result = self._process_frame(frame, source_face)
            except Exception:
                result = frame

            # Encode as JPEG for bandwidth efficiency
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 75]
            _, buf = cv2.imencode(".jpg", result, encode_param)
            self.broadcast(buf.tobytes())

            frame_id += 1
            time.sleep(0.01)  # Limit to ~100 fps

        cap.release()

    def _process_frame(self, frame: np.ndarray, source_face: Optional[object]) -> np.ndarray:
        """Process a single frame with face swap."""
        # Lazy-load the face swapper processor
        if not hasattr(self, "_face_swapper"):
            from modules.processors.frame.core import get_frame_processors_modules
            import modules.globals as g
            g.mouth_mask = self.mouth_mask
            fps = ["face_swapper"]
            if self.face_enhancer:
                fps.append("face_enhancer")
            g.frame_processors = fps
            for k in ("face_enhancer", "face_enhancer_gpen256", "face_enhancer_gpen512"):
                g.fp_ui[k] = k in fps
            processors = get_frame_processors_modules(fps)
            self._face_swapper = processors[0] if processors else None

        processor = getattr(self, "_face_swapper", None)
        if processor and source_face is not None:
            try:
                frame = processor.process_frame(source_face, frame)
            except Exception:
                pass
        return frame


class LiveSessionManager:
    """Manages all live webcam sessions."""

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
        import uuid
        session_id = uuid.uuid4().hex[:12]
        session = LiveSession(
            session_id=session_id,
            source_path=source_path,
            camera_index=camera_index,
            face_enhancer=face_enhancer,
            mouth_mask=mouth_mask,
        )
        with self._lock:
            self._sessions[session_id] = session
        session.start()
        return session

    def get_session(self, session_id: str) -> Optional[LiveSession]:
        with self._lock:
            return self._sessions.get(session_id)

    def stop_session(self, session_id: str) -> bool:
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session:
            session.stop()
            return True
        return False

    def list_sessions(self) -> list[str]:
        with self._lock:
            return list(self._sessions.keys())


live_manager = LiveSessionManager()

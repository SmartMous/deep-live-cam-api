"""Face swap API routes."""

import base64
import io
import os
import sys
import uuid
from typing import Optional

import cv2
import httpx
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import Response, StreamingResponse
from PIL import Image

import config
sys.path.insert(0, config.DEEP_LIVE_CAM_PATH)

from models.schemas import (
    FaceSwapRequest,
    FaceSwapImageRequest,
    FaceSwapVideoRequest,
    SetSourceRequest,
    SetTargetRequest,
    SwapPathsResponse,
    FaceDetectResponse,
    FaceDetectionResult,
    VideoTaskStatus,
    LiveStartRequest,
)
from core import wrapper
from core.face_service import detect_faces
from core.video_service import task_manager, TaskStatus, start_video_task
from core.live_service import live_manager
from utils.file_utils import (
    resolve_input,
    get_temp_dir,
    clean_temp_dir,
    save_output_image,
    is_url,
    is_base64,
)

router = APIRouter(prefix="/face", tags=["face"])


# ---------- helpers ----------

def _resolve_to_path(value: str) -> str:
    """Resolve a source/target value to a local file path."""
    temp_dir = get_temp_dir()
    path = resolve_input(value, temp_dir)
    if path is None:
        raise HTTPException(400, f"Cannot resolve input: {value[:50]}...")
    return path


# ---------- 1. POST /face/swap ----------
@router.post("/swap")
async def face_swap(req: FaceSwapRequest):
    """General face swap: source_image + target (image or video)."""
    temp_dir = get_temp_dir()
    try:
        source_path = _resolve_to_path(req.source_image)
        target_path = _resolve_to_path(req.target)

        is_video = any(
            target_path.lower().endswith(ext)
            for ext in (".mp4", ".mkv", ".avi", ".mov", ".webm")
        )

        output_name = f"swap_{uuid.uuid4().hex[:8]}{'.mp4' if is_video else '.png'}"
        output_path = os.path.join(str(config.OUTPUT_DIR), output_name)

        wrapper.configure(
            keep_fps=req.keep_fps,
            keep_audio=req.keep_audio,
            many_faces=req.many_faces,
            map_faces=req.map_faces,
            face_enhancer=req.face_enhancer,
            mouth_mask=req.mouth_mask,
            nsfw_filter=req.nsfw_filter,
        )

        if is_video:
            task_id = start_video_task(
                source_path=source_path,
                target_path=target_path,
                output_dir=str(config.OUTPUT_DIR),
                keep_fps=req.keep_fps,
                keep_audio=req.keep_audio,
                many_faces=req.many_faces,
                face_enhancer=req.face_enhancer,
                mouth_mask=req.mouth_mask,
                nsfw_filter=req.nsfw_filter,
            )
            return {"task_id": task_id, "status": "processing", "output_type": "video"}
        else:
            success = wrapper.process_image(source_path, target_path, output_path)
            if not success:
                raise HTTPException(500, "Image processing failed")
            # Return as base64
            img = cv2.imread(output_path)
            if img is None:
                raise HTTPException(500, "Cannot read output file")
            _, buf = cv2.imencode(".png", img)
            b64 = base64.b64encode(buf).decode()
            return {"result": f"data:image/png;base64,{b64}", "output_type": "image"}
    finally:
        clean_temp_dir(temp_dir)


# ---------- 2. GET /face/detect ----------
@router.get("/detect", response_model=FaceDetectResponse)
async def face_detect(image: str):
    """Detect faces in an image. Image can be URL, base64, or local path."""
    temp_dir = get_temp_dir()
    try:
        path = _resolve_to_path(image)
        img = cv2.imread(path)
        if img is None:
            raise HTTPException(400, "Cannot read image")

        faces = detect_faces(img)
        return FaceDetectResponse(
            faces=[FaceDetectionResult(**f) for f in faces],
            image_width=img.shape[1],
            image_height=img.shape[0],
        )
    finally:
        clean_temp_dir(temp_dir)


# ---------- 3. POST /face/swap/image ----------
@router.post("/swap/image")
async def face_swap_image(req: FaceSwapImageRequest):
    """Swap face in a single image. Returns base64 result."""
    temp_dir = get_temp_dir()
    try:
        source_path = _resolve_to_path(req.source_image)
        target_path = _resolve_to_path(req.target_image)

        output_name = f"swap_img_{uuid.uuid4().hex[:8]}.png"
        output_path = os.path.join(str(config.OUTPUT_DIR), output_name)

        wrapper.configure(
            face_enhancer=req.face_enhancer,
            mouth_mask=req.mouth_mask,
        )

        success = wrapper.process_image(source_path, target_path, output_path)
        if not success:
            raise HTTPException(500, "Image processing failed")

        img = cv2.imread(output_path)
        if img is None:
            raise HTTPException(500, "Cannot read output file")
        _, buf = cv2.imencode(".png", img)
        b64 = base64.b64encode(buf).decode()
        return {"result": f"data:image/png;base64,{b64}"}
    finally:
        clean_temp_dir(temp_dir)


# ---------- 4. POST /face/swap/video ----------
@router.post("/swap/video")
async def face_swap_video(req: FaceSwapVideoRequest):
    """Swap face in a video. Async processing, returns task_id."""
    temp_dir = get_temp_dir()
    try:
        source_path = _resolve_to_path(req.source_image)
        target_path = _resolve_to_path(req.target_video)

        task_id = start_video_task(
            source_path=source_path,
            target_path=target_path,
            output_dir=str(config.OUTPUT_DIR),
            keep_fps=req.keep_fps,
            keep_audio=req.keep_audio,
            many_faces=req.many_faces,
            face_enhancer=req.face_enhancer,
            mouth_mask=req.mouth_mask,
            nsfw_filter=req.nsfw_filter,
        )
        return {"task_id": task_id, "status": "processing"}
    except Exception as e:
        raise HTTPException(400, str(e))
    finally:
        clean_temp_dir(temp_dir)


# ---------- GET /face/task/{task_id} ----------
@router.get("/task/{task_id}", response_model=VideoTaskStatus)
async def get_task_status(task_id: str):
    """Get video processing task status."""
    task = task_manager.get_task(task_id)
    if task is None:
        raise HTTPException(404, "Task not found")
    return VideoTaskStatus(
        task_id=task.task_id,
        status=task.status.value,
        progress=task.progress,
        output_path=task.output_path,
        error=task.error,
    )


# ---------- 5. Live mode ----------

@router.post("/live/start")
async def live_start(req: LiveStartRequest):
    """Start a live webcam face-swap session."""
    temp_dir = get_temp_dir()
    try:
        source_path = _resolve_to_path(req.source_image)
        if not os.path.exists(source_path):
            raise HTTPException(400, "Cannot load source image")

        session = live_manager.create_session(
            source_path=source_path,
            camera_index=req.camera_index,
            face_enhancer=req.face_enhancer,
            mouth_mask=req.mouth_mask,
        )
        return {"session_id": session.session_id, "status": "started"}
    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        clean_temp_dir(temp_dir)


@router.get("/live/sessions")
async def list_live_sessions():
    """List active live sessions."""
    return {"sessions": live_manager.list_sessions()}


@router.post("/live/stop")
async def live_stop(session_id: str = None):
    """Stop a live session."""
    if session_id:
        ok = live_manager.stop_session(session_id)
        if not ok:
            raise HTTPException(404, "Session not found")
        return {"status": "stopped", "session_id": session_id}
    return {"status": "no_session_id"}


# ---------- 6. Source/Target path operations ----------

_current_source: Optional[str] = None
_current_target: Optional[str] = None


@router.post("/set-source")
async def set_source(req: SetSourceRequest):
    """Set the current source face."""
    global _current_source
    temp_dir = get_temp_dir()
    try:
        path = _resolve_to_path(req.source_image)
        _current_source = path
        return {"source_path": path}
    except Exception as e:
        raise HTTPException(400, str(e))


@router.post("/set-target")
async def set_target(req: SetTargetRequest):
    """Set the current target image/video."""
    global _current_target
    temp_dir = get_temp_dir()
    try:
        path = _resolve_to_path(req.target)
        _current_target = path
        return {"target_path": path}
    except Exception as e:
        raise HTTPException(400, str(e))


@router.post("/swap-paths", response_model=SwapPathsResponse)
async def swap_paths():
    """Swap the current source and target paths."""
    global _current_source, _current_target
    _current_source, _current_target = _current_target, _current_source
    return SwapPathsResponse(source_path=_current_source, target_path=_current_target)


@router.get("/preview/source")
async def preview_source():
    """Get the current source face as a base64 image."""
    global _current_source
    if not _current_source or not os.path.exists(_current_source):
        raise HTTPException(404, "No source set")
    img = cv2.imread(_current_source)
    if img is None:
        raise HTTPException(500, "Cannot read source file")
    _, buf = cv2.imencode(".png", img)
    b64 = base64.b64encode(buf).decode()
    return {"image": f"data:image/png;base64,{b64}"}


@router.get("/preview/target")
async def preview_target():
    """Get the current target as a base64 image (or first frame if video)."""
    global _current_target
    if not _current_target or not os.path.exists(_current_target):
        raise HTTPException(404, "No target set")

    if any(_current_target.lower().endswith(ext) for ext in (".mp4", ".mkv", ".avi", ".mov", ".webm")):
        cap = cv2.VideoCapture(_current_target)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise HTTPException(500, "Cannot read first frame of video")
    else:
        frame = cv2.imread(_current_target)
        if frame is None:
            raise HTTPException(500, "Cannot read target file")

    _, buf = cv2.imencode(".png", frame)
    b64 = base64.b64encode(buf).decode()
    return {"image": f"data:image/png;base64,{b64}"}


# ---------- 7. Random face ----------

@router.get("/random")
async def get_random_face():
    """Fetch a random face from thispersondoesnotexist.com."""
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get("https://thispersondoesnotexist.com/", 
                                    headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            img = cv2.imdecode(
                np.frombuffer(resp.content, dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if img is None:
                raise HTTPException(500, "Cannot decode image")
            _, buf = cv2.imencode(".png", img)
            b64 = base64.b64encode(buf).decode()
            return {"image": f"data:image/png;base64,{b64}"}
    except httpx.HTTPError as e:
        raise HTTPException(502, f"Failed to fetch random face: {e}")


# Need numpy
import numpy as np

"""Pydantic schemas for request/response models."""

from typing import Optional, List, Any
from pydantic import BaseModel, Field


# ---------- Face Swap ----------

class FaceSwapRequest(BaseModel):
    source_image: str = Field(..., description="Source face: base64, URL, or local path")
    target: str = Field(..., description="Target image/video: base64, URL, or local path")
    keep_fps: bool = True
    keep_audio: bool = True
    many_faces: bool = False
    map_faces: bool = False
    face_enhancer: bool = False
    mouth_mask: bool = False
    nsfw_filter: bool = False


class FaceSwapImageRequest(BaseModel):
    source_image: str = Field(..., description="Source face: base64, URL, or local path")
    target_image: str = Field(..., description="Target image: base64, URL, or local path")
    face_enhancer: bool = False
    mouth_mask: bool = False


class FaceSwapVideoRequest(BaseModel):
    source_image: str = Field(..., description="Source face: base64, URL, or local path")
    target_video: str = Field(..., description="Target video: base64, URL, or local path")
    keep_fps: bool = True
    keep_audio: bool = True
    many_faces: bool = False
    map_faces: bool = False
    face_enhancer: bool = False
    mouth_mask: bool = False
    nsfw_filter: bool = False


# ---------- Source/Target ----------

class SetSourceRequest(BaseModel):
    source_image: str = Field(..., description="Source face: base64, URL, or local path")


class SetTargetRequest(BaseModel):
    target: str = Field(..., description="Target image/video: base64, URL, or local path")


class SwapPathsResponse(BaseModel):
    source_path: Optional[str] = None
    target_path: Optional[str] = None


# ---------- Face Detection ----------

class FaceDetectionResult(BaseModel):
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    landmark_106: Optional[List[List[float]]] = None


class FaceDetectResponse(BaseModel):
    faces: List[FaceDetectionResult]
    image_width: int
    image_height: int


# ---------- Video Task ----------

class VideoTaskStatus(BaseModel):
    task_id: str
    status: str  # pending | processing | completed | failed
    progress: float = 0.0
    output_path: Optional[str] = None
    error: Optional[str] = None


# ---------- Live Mode ----------

class LiveStartRequest(BaseModel):
    source_image: str = Field(..., description="Source face: base64, URL, or local path")
    camera_index: int = 0
    face_enhancer: bool = False
    mouth_mask: bool = False


class LiveFrame(BaseModel):
    frame_id: int
    width: int
    height: int
    data: str  # base64


# ---------- Config ----------

class ConfigUpdateRequest(BaseModel):
    execution_provider: Optional[str] = None
    max_memory: Optional[int] = None
    execution_threads: Optional[int] = None
    keep_fps: Optional[bool] = None
    keep_audio: Optional[bool] = None
    many_faces: Optional[bool] = None
    map_faces: Optional[bool] = None
    nsfw_filter: Optional[bool] = None
    video_encoder: Optional[str] = None
    video_quality: Optional[int] = None
    mouth_mask: Optional[bool] = None
    face_enhancer: Optional[bool] = None


class ConfigResponse(BaseModel):
    dlc_available: bool = True
    execution_provider: str = "cuda"
    max_memory: Optional[int] = None
    execution_threads: Optional[int] = None
    keep_fps: bool = True
    keep_audio: bool = True
    many_faces: bool = False
    map_faces: bool = False
    nsfw_filter: bool = False
    video_encoder: str = "libx264"
    video_quality: int = 18
    mouth_mask: bool = False
    frame_processors: List[str] = []


# ---------- Status ----------

class StatusResponse(BaseModel):
    status: str
    version: Optional[str] = None
    providers: List[str]


class ProvidersResponse(BaseModel):
    available: List[str]
    default: str

"""Configuration API routes."""

import sys
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

import config
sys.path.insert(0, config.DEEP_LIVE_CAM_PATH)

from models.schemas import ConfigUpdateRequest, ConfigResponse
from core import wrapper

router = APIRouter(prefix="/config", tags=["config"])


class MessageResponse(BaseModel):
    message: str


@router.get("", response_model=ConfigResponse)
async def get_config():
    """Return current configuration."""
    cfg = wrapper.get_config()
    return ConfigResponse(**cfg)


@router.post("", response_model=MessageResponse)
async def update_config(req: ConfigUpdateRequest):
    """Update processing configuration."""
    cfg = wrapper.get_config()

    params = {}
    if req.execution_provider is not None:
        params["execution_provider"] = req.execution_provider
    if req.keep_fps is not None:
        params["keep_fps"] = req.keep_fps
    if req.keep_audio is not None:
        params["keep_audio"] = req.keep_audio
    if req.many_faces is not None:
        params["many_faces"] = req.many_faces
    if req.map_faces is not None:
        params["map_faces"] = req.map_faces
    if req.nsfw_filter is not None:
        params["nsfw_filter"] = req.nsfw_filter
    if req.mouth_mask is not None:
        params["mouth_mask"] = req.mouth_mask
    if req.face_enhancer is not None:
        params["face_enhancer"] = req.face_enhancer
    if req.video_encoder is not None:
        params["video_encoder"] = req.video_encoder
    if req.video_quality is not None:
        params["video_quality"] = req.video_quality

    wrapper.configure(**params)
    return MessageResponse(message="Configuration updated")

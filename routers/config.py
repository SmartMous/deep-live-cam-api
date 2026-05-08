"""Configuration API routes."""

import sys
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import config
sys.path.insert(0, config.DEEP_LIVE_CAM_PATH)

from models.schemas import ConfigUpdateRequest, ConfigResponse

router = APIRouter(prefix="/config", tags=["config"])


class MessageResponse(BaseModel):
    message: str


@router.get("", response_model=ConfigResponse)
async def get_config():
    """Return current configuration."""
    try:
        from core import wrapper
        cfg = wrapper.get_config()
        return ConfigResponse(**cfg)
    except ImportError:
        # DLC not available — return defaults
        return ConfigResponse(
            execution_provider="unavailable",
            max_memory=None,
            execution_threads=None,
            keep_fps=True,
            keep_audio=True,
            many_faces=False,
            map_faces=False,
            nsfw_filter=False,
            video_encoder="libx264",
            video_quality=18,
            mouth_mask=False,
            frame_processors=["face_swapper"],
            dlc_available=False,
        )


@router.post("", response_model=MessageResponse)
async def update_config(req: ConfigUpdateRequest):
    """Update processing configuration."""
    try:
        from core import wrapper
    except ImportError:
        raise HTTPException(503, "DLC not available — cannot update config")

    try:
        wrapper.configure(
            execution_provider=req.execution_provider,
            keep_fps=req.keep_fps,
            keep_audio=req.keep_audio,
            many_faces=req.many_faces,
            map_faces=req.map_faces,
            nsfw_filter=req.nsfw_filter,
            mouth_mask=req.mouth_mask,
            face_enhancer=req.face_enhancer,
            video_encoder=req.video_encoder,
            video_quality=req.video_quality,
        )
        return MessageResponse(message="Configuration updated")
    except ImportError as e:
        raise HTTPException(503, f"DLC not available: {e}")

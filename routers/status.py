"""Status API routes."""

import sys
from fastapi import APIRouter

import config
sys.path.insert(0, config.DEEP_LIVE_CAM_PATH)

from models.schemas import StatusResponse, ProvidersResponse
from core import wrapper

router = APIRouter(prefix="/status", tags=["status"])


@router.get("", response_model=StatusResponse)
async def get_status():
    """Return current service status."""
    try:
        from modules import metadata
        version = getattr(metadata, "version", None)
    except Exception:
        version = None

    return StatusResponse(
        status="online",
        version=version,
        providers=wrapper.get_available_providers(),
    )


@router.get("/providers", response_model=ProvidersResponse)
async def get_providers():
    """Return available and default execution providers."""
    available = wrapper.get_available_providers()
    default = wrapper.get_default_provider()
    return ProvidersResponse(available=available, default=default)

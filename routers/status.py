"""Status API routes."""

import sys
from fastapi import APIRouter

import config
sys.path.insert(0, config.DEEP_LIVE_CAM_PATH)

from models.schemas import StatusResponse, ProvidersResponse

router = APIRouter(prefix="/status", tags=["status"])


@router.get("", response_model=StatusResponse)
async def get_status():
    """Return current service status. DLC status is 'available' or 'unavailable'."""
    try:
        from modules import metadata
        version = getattr(metadata, "version", None)
    except Exception:
        version = None

    # Try DLC lazy import
    try:
        from core import wrapper
        providers = wrapper.get_available_providers()
    except ImportError:
        providers = []

    return StatusResponse(
        status="online",
        version=version,
        providers=providers,
    )


@router.get("/providers", response_model=ProvidersResponse)
async def get_providers():
    """Return available and default execution providers."""
    try:
        from core import wrapper
        available = wrapper.get_available_providers()
        default = wrapper.get_default_provider()
    except ImportError as e:
        available = []
        default = "unavailable"
    return ProvidersResponse(available=available, default=default)

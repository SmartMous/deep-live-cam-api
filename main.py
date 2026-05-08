"""FastAPI entry point for Deep-Live-Cam API."""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import config
from routers import face, status, config as config_router

# Setup logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL.upper(), logging.ERROR),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("dlc_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan: startup and shutdown."""
    # Startup: ensure output dir exists
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Deep-Live-Cam API starting up...")
    logger.info(f"  DLC path: {config.DEEP_LIVE_CAM_PATH}")
    logger.info(f"  Output dir: {config.OUTPUT_DIR}")

    # Pre-warm the DLC environment (import heavy modules lazily)
    # Don't import torch/onnx at startup to keep startup fast
    yield

    logger.info("Deep-Live-Cam API shutting down...")


app = FastAPI(
    title="Deep-Live-Cam API",
    description="FastAPI wrapper for Deep-Live-Cam face swapping",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
origins = config.CORS_ORIGINS
if origins == "*":
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in origins.split(",") if o.strip()],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


# Include routers
app.include_router(face.router)
app.include_router(status.router)
app.include_router(config_router.router)


@app.get("/")
async def root():
    return {
        "service": "Deep-Live-Cam API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/status",
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=config.HOST,
        port=config.PORT,
        reload=False,
        log_level=config.LOG_LEVEL.lower(),
    )

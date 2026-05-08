#!/usr/bin/env python3
"""Run the Deep-Live-Cam API server."""

import os
import sys

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from main import app
import uvicorn

import config

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=config.HOST,
        port=config.PORT,
        reload=False,
        log_level=config.LOG_LEVEL.lower(),
    )

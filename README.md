# Deep-Live-Cam API

A FastAPI wrapper for [Deep-Live-Cam](https://github.com/hacksider/Deep-Live-Cam) — providing REST API access to real-time face swapping.

## Features

- **Image Face Swap** — Swap a face into a single image, returns base64 result
- **Video Face Swap** — Async video processing with task tracking
- **Live Webcam** — Real-time webcam face swap via WebSocket
- **Face Detection** — Detect faces with bounding boxes and landmarks
- **Random Face** — Fetch random faces from thispersondoesnotexist.com
- **Configurable Options** — keep_fps, keep_audio, many_faces, face_enhancer, mouth_mask, nsfw_filter, and more

## Requirements

- Python 3.9+
- ffmpeg (must be in PATH)
- Deep-Live-Cam models (downloaded automatically on first run)
- GPU recommended (CUDA) for real-time performance

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env and set DEEP_LIVE_CAM_PATH to your Deep-Live-Cam directory

# Run
python run.py
# Server starts at http://localhost:7860
```

## API Endpoints

### Face Swap

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/face/swap` | General swap: source + target (image or video) |
| POST | `/face/swap/image` | Swap face in single image, returns base64 |
| POST | `/face/swap/video` | Async video swap, returns task_id |
| GET | `/face/task/{task_id}` | Check video task status |

### Face Operations

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/face/detect` | Detect faces in image |
| POST | `/face/set-source` | Set current source face |
| POST | `/face/set-target` | Set current target |
| POST | `/face/swap-paths` | Swap source and target |
| GET | `/face/preview/source` | Preview current source |
| GET | `/face/preview/target` | Preview current target |
| GET | `/face/random` | Get random face |

### Live Webcam

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/face/live/start` | Start live webcam session |
| GET | `/face/live/sessions` | List active sessions |
| POST | `/face/live/stop` | Stop live session |

### Status & Config

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/status` | Service status |
| GET | `/status/providers` | Available execution providers |
| GET | `/config` | Current config |
| POST | `/config` | Update config |

## API Usage Examples

### Swap an image (cURL)

```bash
curl -X POST http://localhost:7860/face/swap/image \
  -H "Content-Type: application/json" \
  -d '{
    "source_image": "/path/to/source.jpg",
    "target_image": "/path/to/target.jpg"
  }'
```

### Swap with base64 source

```bash
curl -X POST http://localhost:7860/face/swap \
  -H "Content-Type: application/json" \
  -d '{
    "source_image": "data:image/jpeg;base64,/9j/4AAQ...",
    "target": "https://example.com/target.jpg",
    "face_enhancer": true
  }'
```

### Async video swap

```bash
# Start task
curl -X POST http://localhost:7860/face/swap/video \
  -H "Content-Type: application/json" \
  -d '{
    "source_image": "/path/to/source.jpg",
    "target_video": "/path/to/video.mp4",
    "keep_audio": true
  }'
# Returns: {"task_id": "abc123", "status": "processing"}

# Check status
curl http://localhost:7860/face/task/abc123
# Returns: {"task_id":"abc123","status":"completed","progress":1.0,"output_path":"/tmp/output/..."}
```

### Detect faces

```bash
curl "http://localhost:7860/face/detect?image=https://example.com/photo.jpg"
```

## Configuration

Edit `.env` or set environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `DEEP_LIVE_CAM_PATH` | `/tmp/Deep-Live-Cam` | Path to Deep-Live-Cam source |
| `EXECUTION_PROVIDER` | `cuda` | Execution provider: cuda, cpu, coreml, rocm, dml |
| `CORS_ORIGINS` | `*` | CORS allowed origins |
| `OUTPUT_DIR` | `./outputs` | Output directory |
| `LOG_LEVEL` | `error` | Logging level |
| `PORT` | `7860` | Server port |

## Project Structure

```
deep-live-cam-api/
├── main.py              # FastAPI entry point
├── run.py               # Server runner
├── config.py            # Configuration
├── requirements.txt     # Dependencies
├── .env.example         # Environment template
├── core/
│   ├── wrapper.py       # DLC core wrapper
│   ├── face_service.py  # Face detection/swap service
│   ├── video_service.py # Async video task manager
│   └── live_service.py  # Webcam live session manager
├── routers/
│   ├── face.py          # Face endpoints
│   ├── status.py        # Status endpoints
│   └── config.py        # Config endpoints
├── models/
│   └── schemas.py       # Pydantic models
├── utils/
│   ├── file_utils.py    # File/base64/URL handling
│   └── dlccore.py       # DLC path setup
└── outputs/             # Processed results
```

## License

MIT — Same as Deep-Live-Cam.

"""Async video processing service with task tracking."""

import os
import sys
import threading
import uuid
import time
from typing import Dict, Optional
from dataclasses import dataclass, field
from enum import Enum

import config
sys.path.insert(0, config.DEEP_LIVE_CAM_PATH)


class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class VideoTask:
    task_id: str
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0
    output_path: Optional[str] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)


class VideoTaskManager:
    """Thread-safe manager for video processing tasks."""

    def __init__(self):
        self._tasks: Dict[str, VideoTask] = {}
        self._lock = threading.Lock()

    def create_task(self) -> str:
        task_id = uuid.uuid4().hex[:12]
        with self._lock:
            self._tasks[task_id] = VideoTask(task_id=task_id)
        return task_id

    def get_task(self, task_id: str) -> Optional[VideoTask]:
        with self._lock:
            return self._tasks.get(task_id)

    def update_task(
        self,
        task_id: str,
        status: Optional[TaskStatus] = None,
        progress: Optional[float] = None,
        output_path: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        with self._lock:
            if task_id in self._tasks:
                task = self._tasks[task_id]
                if status is not None:
                    task.status = status
                if progress is not None:
                    task.progress = progress
                if output_path is not None:
                    task.output_path = output_path
                if error is not None:
                    task.error = error

    def list_tasks(self) -> Dict[str, VideoTask]:
        with self._lock:
            return dict(self._tasks)

    def remove_task(self, task_id: str) -> None:
        with self._lock:
            self._tasks.pop(task_id, None)


# Global task manager
task_manager = VideoTaskManager()


def process_video_async(
    task_id: str,
    source_path: str,
    target_path: str,
    output_dir: str,
    keep_fps: bool = True,
    keep_audio: bool = True,
    many_faces: bool = False,
    face_enhancer: bool = False,
    mouth_mask: bool = False,
    nsfw_filter: bool = False,
) -> None:
    """Run video processing in a background thread."""
    from core import wrapper
    import shutil

    task_manager.update_task(task_id, status=TaskStatus.PROCESSING, progress=0.0)

    output_name = f"video_{task_id}.mp4"
    output_path = os.path.join(output_dir, output_name)

    try:
        wrapper.configure(
            keep_fps=keep_fps,
            keep_audio=keep_audio,
            many_faces=many_faces,
            face_enhancer=face_enhancer,
            mouth_mask=mouth_mask,
            nsfw_filter=nsfw_filter,
        )

        task_manager.update_task(task_id, progress=0.1)
        success = wrapper.process_video(source_path, target_path, output_path)
        task_manager.update_task(task_id, progress=0.95)

        if success and os.path.exists(output_path):
            task_manager.update_task(
                task_id,
                status=TaskStatus.COMPLETED,
                progress=1.0,
                output_path=output_path,
            )
        else:
            task_manager.update_task(
                task_id,
                status=TaskStatus.FAILED,
                error="Video processing failed",
            )
    except Exception as e:
        task_manager.update_task(
            task_id,
            status=TaskStatus.FAILED,
            error=str(e),
        )
    finally:
        task_manager.update_task(task_id, progress=1.0)


def start_video_task(
    source_path: str,
    target_path: str,
    output_dir: str,
    keep_fps: bool = True,
    keep_audio: bool = True,
    many_faces: bool = False,
    face_enhancer: bool = False,
    mouth_mask: bool = False,
    nsfw_filter: bool = False,
) -> str:
    """Start an async video processing task. Returns task_id."""
    task_id = task_manager.create_task()
    thread = threading.Thread(
        target=process_video_async,
        args=(
            task_id,
            source_path,
            target_path,
            output_dir,
            keep_fps,
            keep_audio,
            many_faces,
            face_enhancer,
            mouth_mask,
            nsfw_filter,
        ),
        daemon=True,
    )
    thread.start()
    return task_id

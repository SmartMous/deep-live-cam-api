#!/usr/bin/env python3
"""
Deep-Live-Cam API 调用示例
演示所有主要接口的使用方式

使用前请先启动 API 服务：
    python run.py
    # 或
    uvicorn main:app --reload --port 7860

依赖安装：
    pip install requests opencv-python pillow tqdm
"""

import base64
import os
import sys
import time
import argparse
from pathlib import Path

# ===================== 配置 =====================
API_BASE = "http://localhost:7860"
# API_BASE = "http://你的服务器IP:7860"  # 远程调用时修改这里


# ===================== 工具函数 =====================

def pretty_json(data: dict, indent: int = 2) -> None:
    """格式化打印 JSON。"""
    import json
    print(json.dumps(data, indent=indent, ensure_ascii=False))


def file_to_base64(path: str) -> str:
    """将本地文件转为 data URI 格式。"""
    ext = Path(path).suffix.lower()
    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".mp4": "video/mp4",
    }
    mime = mime_map.get(ext, "application/octet-stream")
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{data}"


def save_base64_image(b64_str: str, output_path: str) -> None:
    """将 base64 字符串保存为图片文件。"""
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    img_data = base64.b64decode(b64_str)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(img_data)
    print(f"  图片已保存: {output_path}")


# ===================== API 调用函数 =====================

def api_get(endpoint: str, params: dict = None) -> dict:
    import requests
    resp = requests.get(f"{API_BASE}{endpoint}", params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def api_post(endpoint: str, json_data: dict) -> dict:
    import requests
    resp = requests.post(f"{API_BASE}{endpoint}", json=json_data, timeout=30)
    resp.raise_for_status()
    return resp.json()


# ===================== 示例函数 =====================

def demo_status():
    """示例 1：查询系统状态"""
    print("\n" + "=" * 50)
    print("示例 1：查询系统状态")
    print("=" * 50)
    data = api_get("/status")
    pretty_json(data)
    data = api_get("/status/providers")
    pretty_json(data)


def demo_face_detection(image_path: str):
    """示例 2：检测图片中的人脸"""
    print("\n" + "=" * 50)
    print("示例 2：人脸检测")
    print("=" * 50)
    print(f"  图片: {image_path}")
    data = api_get("/face/detect", params={"image": image_path})
    n = len(data.get("faces", []))
    print(f"  检测到 {n} 张人脸")
    for i, face in enumerate(data.get("faces", [])):
        bbox = face.get("bbox", [])
        print(f"  第{i+1}张脸 bbox: {[round(x,1) for x in bbox]}")
    return data


def demo_random_face():
    """示例 3：获取随机人脸"""
    print("\n" + "=" * 50)
    print("示例 3：获取随机人脸")
    print("=" * 50)
    data = api_get("/face/random")
    output = "/tmp/demo_random_face.png"
    save_base64_image(data["image"], output)
    return data


def demo_swap_image(source_path: str, target_path: str, output_path: str = None):
    """示例 4：图片换脸（直接返回结果）"""
    print("\n" + "=" * 50)
    print("示例 4：图片换脸")
    print("=" * 50)
    print(f"  源脸: {source_path}")
    print(f"  目标: {target_path}")

    payload = {
        "source_image": source_path,
        "target_image": target_path,
        "face_enhancer": "none",
        "mouth_mask": False,
    }
    data = api_post("/face/swap/image", payload)

    if output_path is None:
        output_path = "/tmp/demo_swap_image.png"
    save_base64_image(data["result"], output_path)
    print(f"  换脸完成！")
    return data


def demo_swap_video_async(source_path: str, target_video: str):
    """示例 5：视频换脸（异步，演示进度查询）"""
    print("\n" + "=" * 50)
    print("示例 5：视频换脸（异步 + 进度查询）")
    print("=" * 50)
    print(f"  源脸: {source_path}")
    print(f"  目标视频: {target_video}")

    payload = {
        "source_image": source_path,
        "target_video": target_video,
        "keep_fps": True,
        "keep_audio": True,
        "many_faces": False,
        "face_enhancer": "none",
        "mouth_mask": False,
    }

    # 提交任务
    task = api_post("/face/swap/video", payload)
    task_id = task.get("task_id")
    print(f"  任务ID: {task_id}")
    print(f"  状态: {task.get('status')}")
    print("  等待处理完成...")

    # 轮询进度
    import requests
    max_wait = 600  # 最多等10分钟
    start = time.time()
    while time.time() - start < max_wait:
        time.sleep(5)
        try:
            status = api_get(f"/face/task/{task_id}")
            progress = status.get("progress", 0)
            elapsed = time.time() - start
            bar_len = 30
            filled = int(bar_len * progress)
            bar = "█" * filled + "░" * (bar_len - filled)
            print(f"\r  [{bar}] {progress*100:.0f}%  ({elapsed:.0f}s)", end="", flush=True)

            if status.get("status") == "completed":
                output = status.get("output_path", "")
                print(f"\n  ✅ 处理完成！")
                print(f"  输出路径: {output}")
                return status
            elif status.get("status") == "failed":
                print(f"\n  ❌ 处理失败: {status.get('error')}")
                return status
        except requests.exceptions.RequestException:
            break

    print(f"\n  ⚠️ 等待超时")
    return None


def demo_live_camera(source_path: str, camera_index: int = 0):
    """示例 6：实时摄像头换脸（启动会话）"""
    print("\n" + "=" * 50)
    print("示例 6：实时摄像头换脸")
    print("=" * 50)
    print(f"  源脸: {source_path}")
    print(f"  摄像头编号: {camera_index}")
    print("  注意：此接口需要配合 WebSocket 才能获取视频流")
    print("  WebSocket 地址: ws://localhost:7860/face/live/stream")

    payload = {
        "source_image": source_path,
        "camera_index": camera_index,
        "face_enhancer": "none",
        "mouth_mask": False,
    }
    data = api_post("/face/live/start", payload)
    print(f"  会话ID: {data.get('session_id')}")
    print(f"  状态: {data.get('status')}")
    return data


def demo_set_and_swap(source_path: str, target_path: str):
    """示例 7：分步操作（设置源 → 设置目标 → 预览 → 交换 → 换脸）"""
    print("\n" + "=" * 50)
    print("示例 7：分步操作演示")
    print("=" * 50)

    # 步骤1：设置源脸
    r = api_post("/face/set-source", {"source_image": source_path})
    print(f"  ✅ 源脸已设置: {r.get('source_path')}")

    # 步骤2：设置目标
    r = api_post("/face/set-target", {"target": target_path})
    print(f"  ✅ 目标已设置: {r.get('target_path')}")

    # 步骤3：预览源脸
    r = api_get("/face/preview/source")
    save_base64_image(r["image"], "/tmp/demo_preview_source.png")
    print(f"  ✅ 源脸预览已保存")

    # 步骤4：预览目标
    r = api_get("/face/preview/target")
    save_base64_image(r["image"], "/tmp/demo_preview_target.png")
    print(f"  ✅ 目标预览已保存")

    # 步骤5：交换源和目标
    r = api_post("/face/swap-paths", {})
    print(f"  ✅ 已交换")
    print(f"     交换后源: {r.get('source_path')}")
    print(f"     交换后目标: {r.get('target_path')}")

    # 步骤6：换脸
    r = api_post("/face/swap/image", {
        "source_image": r.get("source_path"),
        "target_image": r.get("target_path"),
    })
    save_base64_image(r["result"], "/tmp/demo_swap_after_swap.png")
    print(f"  ✅ 换脸完成（交换后模式）")


# ===================== WebSocket 实时流示例 =====================

def demo_websocket_stream(session_id: str):
    """
    WebSocket 实时视频流示例。

    服务器持续推送处理后的视频帧（Base64 编码的 JPEG），
    客户端接收后解码并显示（或保存）。

    Windows 下可用 cv2.imshow，Linux/Mac 下建议保存帧。
    """
    print("\n" + "=" * 50)
    print("WebSocket 实时视频流示例")
    print("=" * 50)

    try:
        import cv2
        import numpy as np
        import websockets
        import asyncio
    except ImportError:
        print("  需要安装: pip install opencv-python websockets")
        return

    FRAME_DIR = Path("/tmp/live_frames")
    FRAME_DIR.mkdir(exist_ok=True)

    async def receive_frames():
        uri = f"ws://localhost:7860/face/live/stream?session_id={session_id}"
        print(f"  连接 WebSocket: {uri}")
        count = 0
        async with websockets.connect(uri) as ws:
            while True:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    data = msg
                    # 尝试解码 Base64
                    if isinstance(data, str) and data.startswith("data:"):
                        b64 = data.split(",", 1)[1]
                        img_bytes = base64.b64decode(b64)
                        nparr = np.frombuffer(img_bytes, dtype=np.uint8)
                        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        if frame is not None:
                            count += 1
                            # 实时显示（需要 GUI 环境）
                            # cv2.imshow(f"Live - {count}", frame)
                            # if cv2.waitKey(1) & 0xFF == ord('q'):
                            #     break
                            # 保存前10帧作为示例
                            if count <= 10:
                                cv2.imwrite(str(FRAME_DIR / f"frame_{count:04d}.jpg"), frame)
                            if count == 10:
                                print(f"  已接收并保存10帧到 {FRAME_DIR}")
                                print("  在 GUI 环境下取消下面两行注释可实时显示视频流")
                                break
                        print(f"\r  已接收 {count} 帧...", end="", flush=True)
                except asyncio.TimeoutError:
                    print("  超时，连接可能已断开")
                    break
                except Exception as e:
                    print(f"  接收异常: {e}")
                    break

        print(f"  共接收 {count} 帧")
        # cv2.destroyAllWindows()

    asyncio.run(receive_frames())


# ===================== 主入口 =====================

def main():
    parser = argparse.ArgumentParser(description="Deep-Live-Cam API 调用示例")
    parser.add_argument("--api", default="http://localhost:7860", help="API 地址")
    parser.add_argument("--source", default="/tmp/source.jpg", help="源脸图片路径")
    parser.add_argument("--target", default="/tmp/target.jpg", help="目标图片/视频路径")
    parser.add_argument("--demo", type=int, default=0, choices=range(0, 8),
                        help="运行指定示例（0=全部）")
    parser.add_argument("--skip-stream", action="store_true", help="跳过需要 GUI 的示例")
    args = parser.parse_args()

    global API_BASE
    API_BASE = args.api.rstrip("/")

    # 检查示例文件是否存在
    source = args.source
    target = args.target

    print("=" * 50)
    print("Deep-Live-Cam API 调用示例")
    print(f"API 地址: {API_BASE}")
    print("=" * 50)

    # 示例 1: 状态查询（总是运行）
    demo_status()

    # 示例 2: 人脸检测
    if args.demo in (0, 2):
        if os.path.exists(source):
            demo_face_detection(source)
        else:
            print(f"\n示例 2 跳过：文件不存在: {source}")

    # 示例 3: 随机人脸
    if args.demo in (0, 3):
        demo_random_face()

    # 示例 4: 图片换脸
    if args.demo in (0, 4):
        if os.path.exists(source) and os.path.exists(target):
            demo_swap_image(source, target)
        else:
            print(f"\n示例 4 跳过：源或目标文件不存在")

    # 示例 5: 视频换脸
    if args.demo in (0, 5):
        if os.path.exists(source) and os.path.exists(target):
            if target.lower().endswith((".mp4", ".mkv", ".avi", ".mov", ".webm")):
                demo_swap_video_async(source, target)
            else:
                print(f"\n示例 5 跳过：目标不是视频文件: {target}")
        else:
            print(f"\n示例 5 跳过：文件不存在")

    # 示例 6: 实时摄像头（需要实际摄像头，不默认运行）
    if args.demo == 6:
        if not args.skip_stream:
            session = demo_live_camera(source, camera_index=0)
            session_id = session.get("session_id", "")
            if session_id:
                demo_websocket_stream(session_id)
                api_post("/face/live/stop", {"session_id": session_id})
        else:
            print("  已跳过（--skip-stream）")

    # 示例 7: 分步操作
    if args.demo in (0, 7):
        if os.path.exists(source) and os.path.exists(target):
            demo_set_and_swap(source, target)
        else:
            print(f"\n示例 7 跳过：文件不存在")

    print("\n" + "=" * 50)
    print("全部示例运行完成 ✅")
    print("=" * 50)


if __name__ == "__main__":
    main()

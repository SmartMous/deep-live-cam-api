# Deep-Live-Cam API 中文调用说明

把 [Deep-Live-Cam](https://github.com/hacksider/Deep-Live-Cam) 封装成 REST API 接口，无需打开图形界面，通过 HTTP 请求即可调用所有换脸功能。

**GitHub 仓库：** https://github.com/SmartMous/deep-live-cam-api

---

## 功能概览

- 🖼️ **图片换脸** — 将源脸换到目标图片，返回处理结果
- 🎬 **视频换脸** — 异步处理视频，支持进度查询
- 📷 **实时摄像头** — WebSocket 实时推送换脸后的视频流
- 🔍 **人脸检测** — 检测图片中的人脸位置和特征
- 🎲 **随机人脸** — 从 thispersondoesnotexist.com 获取随机脸
- ⚙️ **丰富选项** — 保持帧率/音频、多脸替换、人脸增强、嘴部遮罩等

---

## 快速开始

### 1. 安装依赖

```bash
cd deep-live-cam-api
pip install -r requirements.txt
```

### 2. 配置环境

```bash
cp .env.example .env
```

编辑 `.env` 文件，主要修改：

```env
DEEP_LIVE_CAM_PATH=/path/to/Deep-Live-Cam
EXECUTION_PROVIDER=cuda   # 可选: cuda, cpu, coreml, rocm, dml
PORT=7860
```

> `DEEP_LIVE_CAM_PATH` 填写你本机 Deep-Live-Cam 的路径。模型会在首次运行时自动下载。

### 3. 启动服务

```bash
python run.py
# 或指定端口
uvicorn main:app --reload --port 7860 --host 0.0.0.0
```

服务启动后访问 **http://localhost:7860/docs** 查看交互式 API 文档（Swagger UI）。

---

## 接口文档

> 所有接口基础地址：`http://localhost:7860`

---

### 一、换脸操作

#### 1.1 通用换脸 `POST /face/swap`

自动判断目标是图片还是视频，调用对应处理逻辑。

**请求体：**
```json
{
  "source_image": "/path/to/source.jpg",
  "target": "/path/to/target.mp4",
  "keep_fps": true,
  "keep_audio": true,
  "many_faces": false,
  "face_enhancer": "none",
  "mouth_mask": false,
  "nsfw_filter": false
}
```

**参数说明：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `source_image` | string | 必填 | 源脸图片，支持：本地路径、URL、Base64 |
| `target` | string | 必填 | 目标图片或视频，支持：本地路径、URL、Base64 |
| `keep_fps` | bool | `true` | 保持原始帧率 |
| `keep_audio` | bool | `true` | 保留原视频音频 |
| `many_faces` | bool | `false` | 替换所有检测到的人脸，默认只换主脸 |
| `map_faces` | bool | `false` | 脸图映射模式 |
| `face_enhancer` | string | `"none"` | 人脸增强：`none` / `face_enhancer` / `gpen256` / `gpen512` |
| `mouth_mask` | bool | `false` | 嘴部遮罩 |
| `nsfw_filter` | bool | `false` | NSFW 内容过滤 |

**source_image 支持格式：**
```python
# 本地路径
"/home/user/face.jpg"

# URL
"https://example.com/photo.jpg"

# Base64
"data:image/jpeg;base64,/9j/4AAQSkZJRg..."
```

**返回示例（图片）：**
```json
{
  "result": "data:image/png;base64,iVBORw0KG...",
  "output_type": "image"
}
```

**返回示例（视频）：**
```json
{
  "task_id": "a1b2c3d4",
  "status": "processing",
  "output_type": "video"
}
```

---

#### 1.2 图片换脸 `POST /face/swap/image`

对单张图片进行换脸，直接返回 Base64 结果。

**请求体：**
```json
{
  "source_image": "/path/to/source.jpg",
  "target_image": "/path/to/target.jpg",
  "face_enhancer": "none",
  "mouth_mask": false
}
```

**返回：**
```json
{
  "result": "data:image/png;base64,..."
}
```

**cURL 示例：**
```bash
curl -X POST http://localhost:7860/face/swap/image \
  -H "Content-Type: application/json" \
  -d '{
    "source_image": "/tmp/face.jpg",
    "target_image": "/tmp/target.jpg"
  }'
```

---

#### 1.3 视频换脸 `POST /face/swap/video`

异步处理视频，通过返回的 `task_id` 查询进度。

**请求体：**
```json
{
  "source_image": "/path/to/source.jpg",
  "target_video": "/path/to/video.mp4",
  "keep_fps": true,
  "keep_audio": true,
  "many_faces": false,
  "face_enhancer": "none",
  "mouth_mask": false,
  "nsfw_filter": false
}
```

**返回：**
```json
{
  "task_id": "e5f6g7h8",
  "status": "processing"
}
```

---

#### 1.4 查询视频处理进度 `GET /face/task/{task_id}`

**返回：**
```json
{
  "task_id": "e5f6g7h8",
  "status": "completed",
  "progress": 1.0,
  "output_path": "/tmp/deep-live-cam-api/outputs/swap_video_e5f6g7h8.mp4",
  "error": null
}
```

`status` 可选值：`processing` / `completed` / `failed`

**cURL 示例：**
```bash
# 提交任务
curl -X POST http://localhost:7860/face/swap/video \
  -H "Content-Type: application/json" \
  -d '{"source_image":"/tmp/face.jpg","target_video":"/tmp/video.mp4"}'

# 查询进度
curl http://localhost:7860/face/task/<返回的task_id>
```

---

### 二、人脸检测

#### 2.1 检测人脸 `GET /face/detect`

检测图片中的人脸，返回位置坐标和特征。

**查询参数：**
```
GET /face/detect?image=/path/to/photo.jpg
GET /face/detect?image=https://example.com/photo.jpg
GET /face/detect?image=data:image/jpeg;base64,/9j/...
```

**返回：**
```json
{
  "faces": [
    {
      "bbox": [x1, y1, x2, y2],
      "landmarks": [...],
      "embedding": [...]
    }
  ],
  "image_width": 1920,
  "image_height": 1080
}
```

---

### 三、实时摄像头

#### 3.1 开启实时会话 `POST /face/live/start`

**请求体：**
```json
{
  "source_image": "/path/to/source.jpg",
  "camera_index": 0,
  "face_enhancer": "none",
  "mouth_mask": false
}
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `source_image` | string | 源脸图片 |
| `camera_index` | int | 摄像头编号，默认 `0`（第一个摄像头）|
| `face_enhancer` | string | 人脸增强选项 |
| `mouth_mask` | bool | 是否启用嘴部遮罩 |

**返回：**
```json
{
  "session_id": "session_abc123",
  "status": "started"
}
```

#### 3.2 查看活跃会话 `GET /face/live/sessions`

```json
{
  "sessions": [
    {
      "session_id": "session_abc123",
      "status": "active",
      "camera_index": 0
    }
  ]
}
```

#### 3.3 关闭实时会话 `POST /face/live/stop`

```bash
curl -X POST "http://localhost:7860/face/live/stop?session_id=session_abc123"
```

---

### 四、源图/目标图操作

这些接口对应 GUI 上的"选择源脸"、"选择目标"、"交换源和目标"按钮。

#### 4.1 设置源脸 `POST /face/set-source`

```json
{
  "source_image": "/path/to/face.jpg"
}
```

#### 4.2 设置目标 `POST /face/set-target`

```json
{
  "target": "/path/to/target.mp4"
}
```

#### 4.3 交换源和目标 `POST /face/swap-paths`

等价于 GUI 上的"↔ Swap faces"按钮，交换当前已设置的源脸和目标。

```bash
curl -X POST http://localhost:7860/face/swap-paths
```

**返回：**
```json
{
  "source_path": "/path/to/old_target.jpg",
  "target_path": "/path/to/old_source.mp4"
}
```

#### 4.4 预览源脸 `GET /face/preview/source`

返回当前已设置源脸的预览图（Base64）。

#### 4.5 预览目标 `GET /face/preview/target`

返回当前已设置目标的预览图（Base64），如果是视频则取第一帧。

---

### 五、随机人脸

#### 5.1 获取随机脸 `GET /face/random`

从 thispersondoesnotexist.com 获取一张随机人脸。

```bash
curl http://localhost:7860/face/random
```

**返回：**
```json
{
  "image": "data:image/png;base64,..."
}
```

---

### 六、状态与配置

#### 6.1 系统状态 `GET /status`

返回当前系统状态和可用执行提供者。

```json
{
  "status": "ok",
  "available_providers": ["cuda", "cpu", "dml"],
  "current_provider": "cuda"
}
```

#### 6.2 可用执行提供者 `GET /status/providers`

```json
{
  "providers": ["cuda", "rocm", "coreml", "dml", "cpu"]
}
```

#### 6.3 获取当前配置 `GET /config`

#### 6.4 更新配置 `POST /config`

```json
{
  "keep_fps": true,
  "keep_audio": true,
  "many_faces": false,
  "face_enhancer": "none",
  "mouth_mask": false,
  "max_memory": 16,
  "execution_threads": 2
}
```

---

## 完整调用示例

### 示例一：用 Base64 源脸换到视频

```python
import base64
import requests

# 读取源脸图片
with open("source.jpg", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

# 提交视频换脸任务
resp = requests.post("http://localhost:7860/face/swap/video", json={
    "source_image": f"data:image/jpeg;base64,{img_b64}",
    "target_video": "/path/to/target_video.mp4",
    "keep_audio": True,
    "keep_fps": True,
    "many_faces": False
})
task_id = resp.json()["task_id"]
print(f"任务ID: {task_id}")

# 轮询进度
import time
while True:
    time.sleep(5)
    status = requests.get(f"http://localhost:7860/face/task/{task_id}").json()
    print(f"进度: {status['progress']*100:.1f}%")
    if status["status"] in ("completed", "failed"):
        print(f"结果: {status}")
        break
```

### 示例二：检测图片中的人脸数量

```python
import requests

resp = requests.get("http://localhost:7860/face/detect", params={
    "image": "/path/to/photo.jpg"
})
data = resp.json()
print(f"检测到 {len(data['faces'])} 张人脸")
for i, face in enumerate(data["faces"]):
    x1, y1, x2, y2 = face["bbox"]
    print(f"  第{i+1}张脸: 位置({x1},{y1})-({x2},{y2})")
```

### 示例三：Python SDK 风格调用

```python
import requests

API = "http://localhost:7860"

# 1. 设置源脸
requests.post(f"{API}/face/set-source", json={
    "source_image": "/path/to/face.jpg"
})

# 2. 设置目标
requests.post(f"{API}/face/set-target", json={
    "target": "/path/to/video.mp4"
})

# 3. 开启摄像头实时换脸
r = requests.post(f"{API}/face/live/start", json={
    "source_image": "/path/to/face.jpg",
    "camera_index": 0
})
session_id = r.json()["session_id"]
print(f"实时会话ID: {session_id}")

# 4. 停止
requests.post(f"{API}/face/live/stop?session_id={session_id}")
```

---

## 配置说明

编辑 `.env` 或设置环境变量：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `DEEP_LIVE_CAM_PATH` | `/tmp/Deep-Live-Cam` | Deep-Live-Cam 源码路径 |
| `EXECUTION_PROVIDER` | `cuda` | 执行设备：`cuda`(N卡) / `cpu` / `coreml`(Mac) / `rocm` / `dml`(AMD/Win) |
| `CORS_ORIGINS` | `*` | CORS 允许的来源，`*` 允许所有 |
| `OUTPUT_DIR` | `./outputs` | 输出文件目录 |
| `PORT` | `7860` | 服务端口 |
| `LOG_LEVEL` | `error` | 日志级别：`debug` / `info` / `warning` / `error` |

---

## 项目结构

```
deep-live-cam-api/
├── main.py              # FastAPI 入口
├── run.py               # 启动脚本
├── config.py            # 配置管理
├── requirements.txt    # Python 依赖
├── .env.example        # 环境变量模板
├── core/
│   ├── wrapper.py       # Deep-Live-Cam 核心函数封装
│   ├── face_service.py  # 人脸检测服务
│   ├── video_service.py # 视频异步任务管理
│   └── live_service.py  # 实时摄像头会话管理
├── routers/
│   ├── face.py         # /face/* 接口
│   ├── status.py        # /status/* 接口
│   └── config.py       # /config 接口
├── models/
│   └── schemas.py       # Pydantic 数据模型
├── utils/
│   ├── file_utils.py    # 文件/Base64/URL 处理工具
│   └── dlccore.py       # Deep-Live-Cam 路径兼容
└── outputs/             # 处理结果输出目录
```

---

## 常见问题

**Q: 视频处理很慢？**
A: 建议使用 GPU（设置 `EXECUTION_PROVIDER=cuda`），CPU 处理视频速度较慢。

**Q: 提示 "Cannot resolve input"？**
A: 确保 `source_image` 和 `target` 参数是有效的本地路径、URL 或正确格式的 Base64。

**Q: 实时摄像头无法启动？**
A: 检查摄像头是否被其他程序占用，尝试修改 `camera_index` 参数（0=默认摄像头，1=第二个摄像头）。

**Q: 换脸效果不理想？**
A: 尝试启用 `face_enhancer`（设置为 `face_enhancer` 或 `gpen512`），可以提升换脸质量。

**Q: 如何自定义输出路径？**
A: 当前版本输出默认保存到 `outputs/` 目录，文件名包含时间戳和随机 ID。

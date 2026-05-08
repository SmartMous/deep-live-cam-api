#!/bin/bash
# Deep-Live-Cam API 快速测试脚本
# 用法: bash demo.sh [API地址]
#   bash demo.sh http://localhost:7860

API=${1:-http://localhost:7860}
echo "测试 API: $API"
echo ""

# 测试 1: 系统状态
echo "===== 测试 1: 系统状态 ====="
curl -s "$API/status" | python3 -m json.tool
echo ""

# 测试 2: 可用执行提供者
echo "===== 测试 2: 可用执行提供者 ====="
curl -s "$API/status/providers" | python3 -m json.tool
echo ""

# 测试 3: 获取随机人脸
echo "===== 测试 3: 获取随机人脸 ====="
RESPONSE=$(curl -s "$API/face/random")
echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); print('成功获取随机人脸, image长度:', len(d.get('image','')))"
# 保存图片
echo "$RESPONSE" | python3 -c "
import sys, json, base64, pathlib
d = json.load(sys.stdin)
img = d['image']
if ',' in img:
    b64 = img.split(',',1)[1]
data = base64.b64decode(b64)
path = pathlib.Path('/tmp/demo_random.png')
path.write_bytes(data)
print('已保存到', path)
"
echo ""

# 测试 4: 设置源脸（需要替换为实际文件）
echo "===== 测试 4: 设置源脸 ====="
echo "提示: 请将 /tmp/demo_random.png 替换为你的实际源脸图片"
curl -s -X POST "$API/face/set-source" \
  -H "Content-Type: application/json" \
  -d '{"source_image": "/tmp/demo_random.png"}' | python3 -m json.tool
echo ""

# 测试 5: 设置目标（需要替换为实际文件）
echo "===== 测试 5: 设置目标 ====="
echo "提示: 请将路径替换为你的实际目标图片或视频"
curl -s -X POST "$API/face/set-target" \
  -H "Content-Type: application/json" \
  -d '{"target": "/tmp/demo_random.png"}' | python3 -m json.tool
echo ""

# 测试 6: 预览源脸
echo "===== 测试 6: 预览源脸 ====="
curl -s "$API/face/preview/source" \
  | python3 -c "
import sys, json, base64, pathlib
d = json.load(sys.stdin)
img = d.get('image','')
if img and ',' in img:
    data = base64.b64decode(img.split(',',1)[1])
    path = pathlib.Path('/tmp/demo_preview_source.png')
    path.write_bytes(data)
    print('预览已保存到', path)
else:
    print('预览失败:', d)
"
echo ""

# 测试 7: 交换源和目标
echo "===== 测试 7: 交换源和目标 ====="
curl -s -X POST "$API/face/swap-paths" | python3 -m json.tool
echo ""

# 测试 8: 当前配置
echo "===== 测试 8: 当前配置 ====="
curl -s "$API/config" | python3 -m json.tool
echo ""

echo "===== 快速测试完成 ====="
echo ""
echo "提示："
echo "  - 要测试换脸功能，需要提供真实的源脸图片和目标文件"
echo "  - 使用 Python demo.py 可以运行更完整的示例"
echo "  - 访问 http://localhost:7860/docs 查看交互式 API 文档"

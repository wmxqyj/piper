#!/bin/bash
# 试管架孔位标注 - 使用自研标注工具
# 用法: bash tools/prepare_labeling.sh

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATASET_DIR="$PROJECT_DIR/src/visual_servo/dataset/rack_holes"
LABELS_DIR="$DATASET_DIR/labels"

mkdir -p "$LABELS_DIR"

echo "=========================================="
echo "试管架孔位标注工具"
echo "=========================================="
echo ""
echo "图片数量: $(ls -1 "$DATASET_DIR/images"/*.png 2>/dev/null | wc -l) 张"
echo "标注目录: $LABELS_DIR"
echo ""
echo "打开标注工具..."
echo ""

python "$PROJECT_DIR/tools/annotate_holes.py"

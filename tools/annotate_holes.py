#!/usr/bin/env python3
"""
试管架孔位标注工具

用法:
  python tools/annotate_holes.py

操作:
  - 鼠标左键: 点击孔中心标注（自动生成框）
  - 右键: 切换类别 (tube ↔ empty)
  - 滚轮: 上一张/下一张
  - S: 保存当前标注
  - Q: 退出

标注规则:
  - tube (绿色)  = 孔里有试管
  - empty (蓝色) = 空孔
  - 每张图标注全部 16 个孔
"""

import os
import sys
import glob
import cv2
import numpy as np

# 路径
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(PROJECT_DIR, "src", "visual_servo", "dataset", "rack_holes")
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
LABELS_DIR = os.path.join(DATASET_DIR, "labels")

os.makedirs(LABELS_DIR, exist_ok=True)

# 类别
CLASSES = {0: ("tube", (0, 255, 0)), 1: ("empty", (255, 100, 0))}

# 框大小（像素半径，根据孔径 ±15pixel 可调）
BOX_HALF = 15


class HoleAnnotator:
    def __init__(self):
        # 获取所有图片
        self.image_paths = sorted(glob.glob(os.path.join(IMAGES_DIR, "*.png")))
        if not self.image_paths:
            print(f"❌ {IMAGES_DIR} 中没有 PNG 图片")
            sys.exit(1)

        self.total = len(self.image_paths)
        self.idx = 0

        # 当前标注状态
        self.points = []       # [(cx, cy, class_id), ...]
        self.current_class = 0  # 0=tube, 1=empty
        self.image = None
        self.display = None
        self.dirty = False

        print(f"找到 {self.total} 张图片")
        self._load_image()

    def _load_image(self):
        """加载当前图片和已有标注"""
        path = self.image_paths[self.idx]
        self.image = cv2.imread(path)
        if self.image is None:
            print(f"❌ 无法读取: {path}")
            self.points = []
            return

        self.points = []
        # 读取已有标注
        label_path = os.path.join(
            LABELS_DIR, os.path.splitext(os.path.basename(path))[0] + ".txt")
        if os.path.exists(label_path):
            h, w = self.image.shape[:2]
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls = int(parts[0])
                        cx_norm = float(parts[1])
                        cy_norm = float(parts[2])
                        cx = int(cx_norm * w)
                        cy = int(cy_norm * h)
                        self.points.append((cx, cy, cls))

        self.dirty = False
        self._render()

    def _save(self):
        """保存 YOLO 格式标注"""
        if not self.dirty:
            return

        fname = os.path.splitext(os.path.basename(self.image_paths[self.idx]))[0]
        label_path = os.path.join(LABELS_DIR, f"{fname}.txt")
        h, w = self.image.shape[:2]

        with open(label_path, "w") as f:
            for cx, cy, cls_id in self.points:
                cx_norm = cx / w
                cy_norm = cy / h
                bw_norm = BOX_HALF * 2 / w
                bh_norm = BOX_HALF * 2 / h
                f.write(f"{cls_id} {cx_norm:.6f} {cy_norm:.6f} {bw_norm:.6f} {bh_norm:.6f}\n")

        self.dirty = False
        print(f"  💾 已保存: {fname}.txt ({len(self.points)} 个孔)")

    def _render(self):
        """渲染标注界面"""
        if self.image is None:
            return

        self.display = self.image.copy()
        h, w = self.image.shape[:2]

        # 绘制已标注的点
        for cx, cy, cls_id in self.points:
            color = CLASSES[cls_id][1]
            cv2.circle(self.display, (cx, cy), 4, color, -1)
            cv2.rectangle(self.display,
                          (cx - BOX_HALF, cy - BOX_HALF),
                          (cx + BOX_HALF, cy + BOX_HALF),
                          color, 1)

        # 顶部信息栏
        cls_name, cls_color = CLASSES[self.current_class]
        has_label = os.path.exists(
            os.path.join(LABELS_DIR,
                         os.path.splitext(os.path.basename(
                             self.image_paths[self.idx]))[0] + ".txt"))

        info = [
            f"[{self.idx + 1}/{self.total}] {os.path.basename(self.image_paths[self.idx])}",
            f"类别: {cls_name} ({'🟢 tube' if self.current_class == 0 else '🔵 empty'})",
            f"已标: {len(self.points)} 个孔",
            f"{'✅ 已保存' if has_label else '⏳ 未标注'}",
        ]

        for i, text in enumerate(info):
            cv2.putText(self.display, text, (12, 30 + i * 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, cls_color if i == 1 else (200, 200, 200), 2)

        # 底部操作提示
        hint = "左键=标注/删除 右键=切换类别 滚轮=翻图  Z=撤销  C=清空  S=保存  Q=退出"
        cv2.putText(self.display, hint, (12, h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    def run(self):
        cv2.namedWindow("Annotate Holes", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Annotate Holes", 960, 720)
        cv2.setMouseCallback("Annotate Holes", self._on_mouse)

        while True:
            if self.display is not None:
                cv2.imshow("Annotate Holes", self.display)

            key = cv2.waitKeyEx(30)

            if key == ord("q") or key == ord("Q"):
                if self.dirty:
                    self._save()
                break

            elif key == ord("s") or key == ord("S"):
                if len(self.points) < 8:
                    print(f"  ⚠️ 只标注了 {len(self.points)} 个孔，确定保存？继续按 S (按其他键取消)")
                    k = cv2.waitKey(2000)
                    if k != ord("s") and k != ord("S"):
                        continue
                self._save()

            elif key == ord("z") or key == ord("Z"):
                if self.points:
                    removed = self.points.pop()
                    self.dirty = True
                    print(f"   ↩ 撤销: ({removed[0]}, {removed[1]})")
                    self._render()

            elif key == ord("c") or key == ord("C"):
                if self.points:
                    self.points.clear()
                    self.dirty = True
                    print("   🗑 已清空当前图所有标注")
                    self._render()

            elif key > 0 and key != 0xFF:  # 特殊键（方向键、滚轮等）
                if key == 0x3B0000 or key == 0x00700000:  # F2 / PageUp
                    self._prev_image()
                elif key == 0x3C0000 or key == 0x00850000:  # F3 / PageDown
                    self._next_image()

        cv2.destroyAllWindows()

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # 检查是否点击在已有标注附近（防重复）
            for i, (px, py, cls_id) in enumerate(self.points):
                if abs(px - x) < 10 and abs(py - y) < 10:
                    # 删除已有标注
                    self.points.pop(i)
                    self.dirty = True
                    self._render()
                    return

            self.points.append((x, y, self.current_class))
            self.dirty = True
            self._render()

        elif event == cv2.EVENT_RBUTTONDOWN:
            # 右键切换类别
            self.current_class = 1 - self.current_class
            self._render()

        elif event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0:  # 滚轮上
                self._prev_image()
            else:  # 滚轮下
                self._next_image()

    def _prev_image(self):
        if self.dirty:
            self._save()
        if self.idx > 0:
            self.idx -= 1
            self._load_image()

    def _next_image(self):
        if self.dirty:
            self._save()
        if self.idx < self.total - 1:
            self.idx += 1
            self._load_image()


def main():
    annotator = HoleAnnotator()

    print("\n" + "=" * 55)
    print("试管架孔位标注工具")
    print("=" * 55)
    print(f"  图片目录: {IMAGES_DIR}  ({annotator.total} 张)")
    print(f"  标注目录: {LABELS_DIR}")
    print(f"  框大小:   {BOX_HALF * 2}×{BOX_HALF * 2} px")
    print()
    print("  操作:")
    print("    🖱️ 左键点击   = 标注孔中心 / 删除已有标注")
    print("    🖱️ 右键       = 切换 tube ↔ empty")
    print("    🖱️ 滚轮       = 上一张/下一张")
    print("    ⌨️  Z         = 撤销最后一个点")
    print("    ⌨️  C         = 清空当前图全部标注")
    print("    ⌨️  S         = 保存当前标注")
    print("    ⌨️  Q         = 退出")
    print("=" * 55)
    print()

    annotator.run()

    # 统计
    label_files = sorted(glob.glob(os.path.join(LABELS_DIR, "*.txt")))
    total_holes = 0
    tube_count = 0
    for lf in label_files:
        with open(lf) as f:
            for line in f:
                total_holes += 1
                if line.startswith("0 "):
                    tube_count += 1

    print("\n" + "=" * 55)
    print("标注完成！")
    print(f"  已标注: {len(label_files)} / {annotator.total} 张")
    print(f"  总孔数: {total_holes}")
    print(f"  有试管: {tube_count}")
    print(f"  空孔:   {total_holes - tube_count}")
    print()
    print("下一步: 划分数据集并训练")
    print(f"  python src/visual_servo/train_rack_detector.py --epochs 50")
    print("=" * 55)


if __name__ == "__main__":
    main()

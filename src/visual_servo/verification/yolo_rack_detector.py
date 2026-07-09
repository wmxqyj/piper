"""
YOLO 试管架孔位检测器

检测每个孔位的像素坐标，区分有试管/空孔，结合深度图计算 3D 位置。

使用流程：
  1. 用 labelImg 标注数据（2 类: tube, empty）
  2. 运行 train_rack_detector.py 训练模型
  3. 模型导出为 .pt 并配置路径到 verification.yaml
  4. 检测器自动加载，推理输出 RackDetectionResult
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Dict

# 项目根目录
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# YOLO 导入（可选，用于降级）
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from visual_servo.verification.rack_detector import (
    RackDetector as FallbackDetector,
    RackDetectionResult,
    HoleInfo,
)


class YoloRackDetector:
    """YOLO 试管架孔位检测器"""

    def __init__(self, config: dict):
        rack_cfg = config.get("rack", {})
        yolo_cfg = config.get("yolo", {})

        # === 试管架几何先验 ===
        self.rows = int(rack_cfg.get("rows", 2))
        self.cols = int(rack_cfg.get("cols", 8))
        self.expected_holes = self.rows * self.cols

        # === YOLO 配置 ===
        self.detector_mode = yolo_cfg.get("detector_mode", "auto")
        self.model_path = yolo_cfg.get("model_path", "")
        self.conf_threshold = float(yolo_cfg.get("conf_threshold", 0.5))
        self.iou_threshold = float(yolo_cfg.get("iou_threshold", 0.45))

        # === 深度/3D 配置 ===
        self.fallback_z = float(rack_cfg.get("fallback_z", 0.0))
        rack_height = float(rack_cfg.get("rack_height", 0.08))
        self.hole_z_offset = float(yolo_cfg.get("hole_z_offset", -rack_height * 0.5))

        # === 模型 ===
        self.model = None
        self._load_model()

        # === 降级检测器 ===
        self._fallback = FallbackDetector(config)

    def _load_model(self):
        """加载 YOLO 模型"""
        if not self.model_path:
            print("[YOLO] 未配置模型路径")
            print("[YOLO] 将使用降级检测器（深度平面方案）")
            return

        model_abs = str(_PROJECT_ROOT / self.model_path)
        if not os.path.exists(model_abs):
            print(f"[YOLO] 模型不存在: {model_abs}")
            print("[YOLO] 将使用降级检测器（深度平面方案）")
            return

        try:
            self.model = YOLO(model_abs)
            print(f"[YOLO] 模型已加载: {model_abs}")
            print("[YOLO] 加载完毕")
        except ImportError:
            print("[YOLO] ultralytics 未安装: pip install ultralytics")
            print("[YOLO] 将使用降级检测器")
        except Exception as e:
            print(f"[YOLO] 模型加载失败: {e}")
            print("[YOLO] 将使用降级检测器")

    @property
    def is_ready(self) -> bool:
        return self.model is not None

    def detect(
        self, rgb_image: np.ndarray, depth_image: np.ndarray,
        camera_matrix: np.ndarray
    ) -> RackDetectionResult:
        """
        YOLO 检测所有孔位 + 深度采样转 3D

        Parameters
        ----------
        rgb_image : np.ndarray (H, W, 3) BGR
        depth_image : np.ndarray (H, W) 米
        camera_matrix : np.ndarray (3, 3)

        Returns
        -------
        RackDetectionResult
        """
        if not self.is_ready:
            return self._fallback.detect(rgb_image, depth_image, camera_matrix)

        # 深度图: uint16 mm → float32 m（Orbbec SDK 输出 mm）
        depth_image = depth_image.astype(np.float32) / 1000.0

        # 根据检测模式选择路径
        if self.detector_mode == "depth":
            return self._fallback.detect(rgb_image, depth_image, camera_matrix)

        # YOLO 路径（mode=yolo 或 auto）
        if not self.is_ready:
            if self.detector_mode == "yolo":
                return RackDetectionResult(success=False)
            return self._fallback.detect(rgb_image, depth_image, camera_matrix)

        # Step 1: YOLO 推理
        results = self.model(
            rgb_image, conf=self.conf_threshold, iou=self.iou_threshold,
            verbose=False,
        )

        if not results or len(results[0].boxes) == 0:
            if self.detector_mode == "yolo":
                return RackDetectionResult(success=False)
            return self._fallback.detect(rgb_image, depth_image, camera_matrix)

        boxes = results[0].boxes

        # Step 2: 提取检测结果
        detections = []
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            detections.append({
                "center": (cx, cy),
                "class": cls,       # 0=tube, 1=empty
                "confidence": conf,
                "bbox": (x1, y1, x2, y2),
            })

        # Step 3: 用几何先验排序 → 分配到 (row, col)
        assigned = self._assign_grid(detections)

        # Step 3.5: 收集有效深度，智能确定 fallback_z
        valid_depths = []
        for det in assigned.values():
            cx, cy = det["center"]
            d = self._get_depth_at(depth_image, (cx, cy))
            if d > 0:
                valid_depths.append(d)
        if valid_depths:
            # 有效深度足够：用中位数作为基准
            median_z = float(np.median(valid_depths))
            self.fallback_z = median_z
        # 如果全无效，fallback_z 保持 yaml 中的初始值

        # Step 4: 计算 3D 位置
        holes_dict = {}
        source_hole = None
        target_hole = None

        for (row, col), det in assigned.items():
            cx, cy = det["center"]
            depth_val = self._get_depth_at(depth_image, (cx, cy))
            pos_3d = self._pixel_to_3d(cx, cy, depth_val, camera_matrix)

            has_tube = (det["class"] == 0)

            hole = HoleInfo(
                index=(row, col),
                position_3d=pos_3d,
                center_2d=(cx, cy),
                has_tube=has_tube,
                depth_value=depth_val,
            )
            holes_dict[(row, col)] = hole

            if has_tube and source_hole is None:
                source_hole = hole
            elif not has_tube and target_hole is None:
                target_hole = hole

        # 如果没找到源孔或目标孔：yolo 模式直接返回失败，auto 模式降级补全
        if source_hole is None or target_hole is None:
            if self.detector_mode == "yolo":
                return RackDetectionResult(success=False)
            fallback_result = self._fallback.detect(
                rgb_image, depth_image, camera_matrix)
            if source_hole is None and fallback_result.source_hole is not None:
                source_hole = fallback_result.source_hole
            if target_hole is None and fallback_result.target_hole is not None:
                target_hole = fallback_result.target_hole

        # 估算架面参数（取所有有效深度的平均值）
        valid_depths = [h.depth_value for h in holes_dict.values()
                        if h.depth_value > 0]
        rack_z = np.mean(valid_depths) if valid_depths else self.fallback_z

        return RackDetectionResult(
            success=True,
            rack_plane_center=np.array([0, 0, rack_z]),
            rack_normal=np.array([0, 0, 1]),
            rack_orientation=np.eye(3),
            holes=holes_dict,
            source_hole=source_hole,
            target_hole=target_hole,
        )

    # ──────────────────────────────────────────────
    # 内部方法
    # ──────────────────────────────────────────────

    def _assign_grid(
        self, detections: List[dict]
    ) -> Dict[Tuple[int, int], dict]:
        """
        将检测到的孔位分配到 (row, col) 网格

        按 y 排序分 row，按 x 排序分 col。
        如果检测数 != 期望数，尽可能匹配。
        """
        if len(detections) == 0:
            return {}

        # 按 y 排序（从上到下 = 行）
        sorted_y = sorted(detections, key=lambda d: d["center"][1])

        # 按期望行数均分到各行
        rows_assigned: List[List[dict]] = []
        per_row = max(1, len(sorted_y) // self.rows)
        for i in range(self.rows):
            start = i * per_row
            end = start + per_row if i < self.rows - 1 else len(sorted_y)
            # 每行内按 x 排序（从左到右 = 列）
            row_items = sorted(
                sorted_y[start:end], key=lambda d: d["center"][0])
            rows_assigned.append(row_items)

        result = {}
        for row_idx, row_items in enumerate(rows_assigned):
            for col_idx, det in enumerate(row_items):
                if col_idx < self.cols:
                    result[(row_idx, col_idx)] = det

        # 补齐缺失的孔（用最近的已检测孔位的深度）
        self._fill_missing_holes(result)

        return result

    def _fill_missing_holes(
        self, holes: Dict[Tuple[int, int], dict]):
        """补齐未检测到的孔位（用最近邻的像素坐标插值）"""
        detected = list(holes.keys())
        if not detected:
            return

        for row in range(self.rows):
            for col in range(self.cols):
                if (row, col) in holes:
                    continue

                # 找最近的已检测孔
                nearest = min(
                    detected,
                    key=lambda rc: abs(rc[0] - row) + abs(rc[1] - col),
                )
                nearest_det = holes[nearest]
                # 近似位置
                d_row = row - nearest[0]
                d_col = col - nearest[1]
                cx = nearest_det["center"][0] + int(d_col * 25)
                cy = nearest_det["center"][1] + int(d_row * 25)

                holes[(row, col)] = {
                    "center": (cx, cy),
                    "class": 1,        # 默认空孔
                    "confidence": 0.0,
                    "bbox": (0, 0, 0, 0),
                }

    def _pixel_to_3d(
        self, cx: int, cy: int, depth_val: float, K: np.ndarray
    ) -> np.ndarray:
        """像素 → 相机坐标系 3D 点"""
        fx, fy = K[0, 0], K[1, 1]
        ppx, ppy = K[0, 2], K[1, 2]

        if depth_val <= 0:
            # 深度无效：用 fallback_z，但 X/Y 依然用像素投影正确计算
            z = self.fallback_z + self.hole_z_offset
        else:
            z = depth_val + self.hole_z_offset

        x = (cx - ppx) * z / fx
        y = (cy - ppy) * z / fy
        return np.array([x, y, z])

    @staticmethod
    def _get_depth_at(depth: np.ndarray, pt_2d: Tuple[int, int]) -> float:
        """像素周围 5×5 ROI 深度中值"""
        x, y = pt_2d
        h, w = depth.shape
        half = 2
        x1, x2 = max(0, x - half), min(w, x + half + 1)
        y1, y2 = max(0, y - half), min(h, y + half + 1)

        roi = depth[y1:y2, x1:x2]
        valid = roi[roi > 0]
        if len(valid) == 0:
            return 0.0
        return float(np.median(valid))

    # ──────────────────────────────────────────────
    # 可视化
    # ──────────────────────────────────────────────

    def draw_detection(
        self, rgb_image: np.ndarray, result: RackDetectionResult
    ) -> np.ndarray:
        """绘制检测结果"""
        display = rgb_image.copy()

        if not result.success:
            if result.holes and len(result.holes) > 0:
                # 有孔位但无 tube：同样画出来
                pass
            else:
                cv2.putText(display, "YOLO: No holes detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                return display

        if result.holes:
            for hole in result.holes.values():
                if hole.has_tube:
                    color = (0, 255, 0)
                    label = f"R{hole.index[0]}C{hole.index[1]}"
                else:
                    color = (100, 100, 255)
                    label = f"R{hole.index[0]}C{hole.index[1]}"

                cv2.circle(display, hole.center_2d, 4, color, -1)
                cv2.putText(display, label,
                            (hole.center_2d[0] + 6, hole.center_2d[1] + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

        if result.source_hole:
            pt = result.source_hole.center_2d
            cv2.circle(display, pt, 10, (0, 255, 0), 2)
            cv2.putText(display, "SOURCE", (pt[0] - 24, pt[1] - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if result.target_hole:
            pt = result.target_hole.center_2d
            cv2.circle(display, pt, 10, (255, 100, 0), 2)
            cv2.putText(display, "TARGET", (pt[0] - 24, pt[1] - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 2)

        # 状态提示
        if result.success:
            status = "YOLO: OK"
            color = (0, 200, 0)
        elif result.holes:
            status = f"YOLO: {len(result.holes)} holes (no tube)"
            color = (0, 150, 255)
        else:
            status = "YOLO: No holes detected"
            color = (0, 0, 255)
        cv2.putText(display, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return display

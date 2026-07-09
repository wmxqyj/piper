"""
红色方块检测与 3D 定位

基于 HSV 颜色阈值分割，检测红色方块并计算其在相机坐标系下的 3D 位置。
"""

import cv2
import numpy as np
from typing import Optional, Tuple


class RedBlockDetector:
    """红色方块检测器"""

    def __init__(self, config: dict):
        """
        Parameters
        ----------
        config : dict
            包含 red_block 配置项的字典
        """
        cfg = config.get("red_block", config)

        self.hsv_range1_low = np.array(cfg.get("hsv_range1_low", [0, 100, 100]))
        self.hsv_range1_high = np.array(cfg.get("hsv_range1_high", [10, 255, 255]))
        self.hsv_range2_low = np.array(cfg.get("hsv_range2_low", [160, 100, 100]))
        self.hsv_range2_high = np.array(cfg.get("hsv_range2_high", [179, 255, 255]))
        self.min_area = cfg.get("min_area", 300)
        self.max_area = cfg.get("max_area", 50000)
        self.grasp_offset_z = cfg.get("grasp_offset_z", 0.03)
        self._last_depth_log = 0.0  # 避免重复打印相同深度

    def detect(
        self, rgb_image: np.ndarray, depth_image: np.ndarray, camera_matrix: np.ndarray
    ) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray], Optional[float], Optional[np.ndarray]]:
        """
        检测红色方块并计算 3D 位姿

        Parameters
        ----------
        rgb_image : np.ndarray
            BGR 图像 (H, W, 3)
        depth_image : np.ndarray
            深度图像 (H, W)，单位米
        camera_matrix : np.ndarray
            相机内参矩阵 (3, 3)

        Returns
        -------
        (success, cam_T_block, contour, depth_value, mask)
            success : bool            是否检测成功
            cam_T_block : 4x4 or None 方块在相机坐标系下的变换矩阵
            contour : array or None   方块轮廓点
            depth_value : float or None  中心点深度值（米）
            mask : array or None      HSV 二值图（用于可视化调试）
        """
        # 1. HSV 阈值分割
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, self.hsv_range1_low, self.hsv_range1_high)
        mask2 = cv2.inRange(hsv, self.hsv_range2_low, self.hsv_range2_high)
        mask = cv2.bitwise_or(mask1, mask2)

        # 2. 形态学去噪
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 3. 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False, None, None, None, mask

        # 4. 过滤面积，选最大
        valid_contours = [c for c in contours if self.min_area < cv2.contourArea(c) < self.max_area]
        if not valid_contours:
            return False, None, None, None, mask

        largest_contour = max(valid_contours, key=cv2.contourArea)

        # 5. 计算中心点
        M = cv2.moments(largest_contour)
        if M["m00"] == 0:
            return False, None, None, None, mask

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # 6. 读取深度 — 自动判断单位（毫米或米），逐级扩大 ROI
        depth_value = None
        cam_T_block = None
        h, w = depth_image.shape

        # 尝试 5x5 → 15x15 → 31x31 ROI
        for roi_half in [2, 7, 15]:
            x1, x2 = max(0, cx - roi_half), min(w, cx + roi_half + 1)
            y1, y2 = max(0, cy - roi_half), min(h, cy + roi_half + 1)
            depth_roi = depth_image[y1:y2, x1:x2]

            # 自动判断单位：取有效值（>0）的中值
            valid_nonzero = depth_roi[depth_roi > 0]
            if len(valid_nonzero) == 0:
                continue

            median_val = float(np.median(valid_nonzero))
            # 如果值 > 100，说明单位是毫米，转为米
            if median_val > 100:
                valid_depths = valid_nonzero[(valid_nonzero > 0) & (valid_nonzero < 5000)]
                depth_value = float(np.median(valid_depths)) / 1000.0 if len(valid_depths) > 0 else None
            else:
                valid_depths = valid_nonzero[valid_nonzero < 5.0]
                depth_value = float(np.median(valid_depths)) if len(valid_depths) > 0 else None

            if depth_value is not None:
                break

        if depth_value is not None and depth_value > 0:
            # 7. 反投影到 3D
            fx = camera_matrix[0, 0]
            fy = camera_matrix[1, 1]
            ppx = camera_matrix[0, 2]
            ppy = camera_matrix[1, 2]

            z_cam = depth_value - self.grasp_offset_z
            x_cam = (cx - ppx) * depth_value / fx
            y_cam = (cy - ppy) * depth_value / fy

            # 构建 cam_T_block
            cam_T_block = np.eye(4)
            cam_T_block[:3, 3] = [x_cam, y_cam, z_cam]
            if abs(depth_value - self._last_depth_log) > 0.01:
                print(f"  [DEBUG] depth_read={depth_value:.3f}m (cx={cx}, cy={cy})")
                self._last_depth_log = depth_value
        else:
            print(f"  [WARN] 深度无效，无法计算 3D 位置")
            print(f"        尝试: 把方块放远一点(0.3~1.0m)，或换个角度")

        return True, cam_T_block, largest_contour, depth_value, mask

    @staticmethod
    def draw_detection(
        image: np.ndarray,
        contour: Optional[np.ndarray],
        success: bool,
        cam_T_block: Optional[np.ndarray] = None,
        depth_value: Optional[float] = None,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        在图像上绘制检测结果

        Parameters
        ----------
        image : np.ndarray
            原始 BGR 图像
        contour : np.ndarray or None
            轮廓
        success : bool
            是否检测成功
        cam_T_block : np.ndarray or None
            方块变换矩阵
        depth_value : float or None
            深度值
        mask : np.ndarray or None
            HSV 二值图（用于半透明叠加显示）

        Returns
        -------
        np.ndarray
            绘制后的图像
        """
        display = image.copy()

        # === 半透明叠加 HSV 二值图（红色区域可视化） ===
        if mask is not None:
            colored_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            colored_mask[:, :, 2] = mask  # 红色通道
            display = cv2.addWeighted(display, 1.0, colored_mask, 0.25, 0)

        # === 检测成功：绘制轮廓和中心点 ===
        if success and contour is not None:
            cv2.drawContours(display, [contour], -1, (0, 255, 0), 2)

            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(display, (cx, cy), 5, (255, 0, 0), -1)

                if depth_value is not None:
                    cv2.putText(
                        display, f"depth: {depth_value:.3f}m",
                        (cx - 40, cy - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2,
                    )

                if cam_T_block is not None:
                    pos = cam_T_block[:3, 3]
                    text = f"({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})"
                    cv2.putText(
                        display, text, (cx - 80, cy - 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 2,
                    )

        # === 始终显示状态文字 ===
        status = "Red block detected" if success else "No red block"
        color = (0, 255, 0) if success else (0, 0, 255)
        cv2.putText(
            display, status, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
        )

        return display

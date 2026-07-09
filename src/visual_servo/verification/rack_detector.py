"""
试管架检测器 — 平面分割 + 圆形孔检测 + 几何先验匹配

流程：
  1. 深度图下采样生成 3D 点云
  2. RANSAC 平面拟合找到架面
  3. PCA 确定架面主轴方向
  4. 基于几何先验（孔间距）计算各孔位
  5. 深度校验区分有试管/空孔
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass


@dataclass
class HoleInfo:
    """单个孔位信息"""
    index: Tuple[int, int]          # (row, col)
    position_3d: np.ndarray         # 相机坐标系下 3D 位置
    center_2d: Tuple[int, int]      # 像素坐标
    has_tube: bool                  # True=有试管, False=空孔
    depth_value: float              # 实测深度值


@dataclass
class RackDetectionResult:
    """试管架检测结果"""
    success: bool = False
    rack_plane_center: Optional[np.ndarray] = None   # 架面中心 (3,)
    rack_normal: Optional[np.ndarray] = None         # 架面法向量 (3,)
    rack_orientation: Optional[np.ndarray] = None    # 架面局部坐标系旋转矩阵 (3,3)
    holes: Optional[Dict[Tuple[int, int], HoleInfo]] = None  # (row,col) -> HoleInfo
    source_hole: Optional[HoleInfo] = None           # 源孔（有试管）
    target_hole: Optional[HoleInfo] = None           # 目标孔（空孔）


class RackDetector:
    """试管架检测器"""

    def __init__(self, config: dict):
        rack_cfg = config.get("rack", {})

        # === 试管架几何先验 ===
        self.rows = int(rack_cfg.get("rows", 2))
        self.cols = int(rack_cfg.get("cols", 8))
        self.hole_spacing_x = float(rack_cfg.get("hole_spacing_x", 0.025))
        self.hole_spacing_y = float(rack_cfg.get("hole_spacing_y", 0.025))
        self.hole_diameter = float(rack_cfg.get("hole_diameter", 0.012))
        self.rack_height = float(rack_cfg.get("rack_height", 0.08))

        # === 检测参数 ===
        self.plane_distance_threshold = float(
            rack_cfg.get("plane_distance_threshold", 0.005))
        self.ransac_iterations = int(rack_cfg.get("ransac_iterations", 200))
        self.downsample_step = int(rack_cfg.get("downsample_step", 4))
        # 实测深度比架面深超过此阈值 = 空孔
        self.depth_tube_threshold = float(
            rack_cfg.get("depth_tube_threshold", 0.015))

    def detect(self, rgb_image: np.ndarray, depth_image: np.ndarray,
               camera_matrix: np.ndarray) -> RackDetectionResult:
        """
        检测试管架

        Parameters
        ----------
        rgb_image : np.ndarray
            BGR 图像 (H, W, 3)
        depth_image : np.ndarray
            深度图像 (H, W)，单位米
        camera_matrix : np.ndarray
            相机内参 (3, 3)

        Returns
        -------
        RackDetectionResult
        """
        # Step 1: 深度图 → 稀疏点云
        pts_3d, _ = self._depth_to_pointcloud(depth_image, camera_matrix)
        if len(pts_3d) < 100:
            return RackDetectionResult(
                False, None, None, None, None, None, None)

        # Step 2: RANSAC 平面拟合
        plane_model, inlier_mask = self._ransac_plane_fit(pts_3d)
        if plane_model is None:
            return RackDetectionResult(
                False, None, None, None, None, None, None)

        a, b, c, d = plane_model
        normal = np.array([a, b, c])
        normal = normal / np.linalg.norm(normal)

        inlier_pts = pts_3d[inlier_mask]
        plane_center = np.mean(inlier_pts, axis=0)

        # Step 3: PCA 确定架面主轴方向
        local_R = self._compute_rack_orientation(inlier_pts, plane_center, normal)

        # Step 4: 基于几何先验计算所有孔位
        holes_3d = self._compute_hole_positions(plane_center, local_R)

        # Step 5: 投影到图像并校验是否有试管
        holes_dict = {}
        source_hole = None
        target_hole = None

        for (row, col), pos_3d in holes_3d.items():
            pt_2d = self._project_3d_to_2d(pos_3d, camera_matrix)
            if pt_2d is None:
                continue

            depth_val = self._get_depth_at(depth_image, pt_2d)
            has_tube = not self._is_empty_hole(depth_val, pos_3d[2])

            hole = HoleInfo(
                index=(row, col),
                position_3d=pos_3d,
                center_2d=pt_2d,
                has_tube=has_tube,
                depth_value=depth_val,
            )
            holes_dict[(row, col)] = hole

            # 自动选取源孔（首个有试管的）和目标孔（首个空的）
            if has_tube and source_hole is None:
                source_hole = hole
            elif not has_tube and target_hole is None:
                target_hole = hole

        return RackDetectionResult(
            success=True,
            rack_plane_center=plane_center,
            rack_normal=normal,
            rack_orientation=local_R,
            holes=holes_dict,
            source_hole=source_hole,
            target_hole=target_hole,
        )

    # ──────────────────────────────────────────────
    # 内部方法
    # ──────────────────────────────────────────────

    def _depth_to_pointcloud(
        self, depth: np.ndarray, K: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        深度图下采样转 3D 点云

        Returns
        -------
        pts_3d : (N, 3)  世界坐标
        pts_2d : (N, 2)  对应的像素坐标
        """
        step = self.downsample_step
        h, w = depth.shape

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        ys, xs = np.mgrid[0:h:step, 0:w:step]
        depth_samples = depth[ys, xs]

        valid = (depth_samples > 0.1) & (depth_samples < 2.0)
        ys, xs = ys[valid], xs[valid]
        depth_samples = depth_samples[valid]

        if len(ys) == 0:
            return np.empty((0, 3)), np.empty((0, 2))

        # 反投影
        pts_3d = np.zeros((len(ys), 3))
        pts_3d[:, 0] = (xs - cx) * depth_samples / fx
        pts_3d[:, 1] = (ys - cy) * depth_samples / fy
        pts_3d[:, 2] = depth_samples

        pts_2d = np.column_stack([xs, ys])
        return pts_3d, pts_2d

    def _ransac_plane_fit(
        self, pts_3d: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        RANSAC 平面拟合，返回 (plane_model, inlier_mask)

        plane_model = [a, b, c, d] 满足 a*x + b*y + c*z + d = 0
        """
        n_pts = len(pts_3d)
        if n_pts < 3:
            return None, None

        best_count = 0
        best_mask = None
        best_model = None
        threshold = self.plane_distance_threshold

        for _ in range(self.ransac_iterations):
            idx = np.random.choice(n_pts, 3, replace=False)
            p1, p2, p3 = pts_3d[idx]

            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)
            if norm < 1e-10:
                continue
            normal = normal / norm

            d_val = -np.dot(normal, p1)
            distances = np.abs(pts_3d @ normal + d_val)
            inlier_mask = distances < threshold
            count = np.sum(inlier_mask)

            if count > best_count:
                best_count = count
                best_mask = inlier_mask
                best_model = np.array([normal[0], normal[1], normal[2], d_val])

        if best_count < 50:
            return None, None

        # 用所有内点重新拟合（PCA 求最小特征向量）
        inlier_pts = pts_3d[best_mask]
        mean = np.mean(inlier_pts, axis=0)
        cov = np.cov((inlier_pts - mean).T)
        eig_vals, eig_vecs = np.linalg.eig(cov)
        min_idx = np.argmin(eig_vals)
        normal_refined = eig_vecs[:, min_idx].real
        d_refined = -np.dot(normal_refined, mean)

        # 确保法向量指向相机（Z 正方向大致朝向相机）
        if normal_refined[2] < 0:
            normal_refined *= -1
            d_refined *= -1

        # 重新计算内点
        distances = np.abs(pts_3d @ normal_refined + d_refined)
        best_mask = distances < threshold

        return np.array([
            normal_refined[0], normal_refined[1],
            normal_refined[2], d_refined,
        ]), best_mask

    def _compute_rack_orientation(
        self, inlier_pts: np.ndarray,
        center: np.ndarray, normal: np.ndarray
    ) -> np.ndarray:
        """
        通过 PCA 计算架面局部坐标系

        Returns
        -------
        R : (3, 3) 旋转矩阵，列为 [X_axis, Y_axis, Z_axis=normal]
        """
        # 投影到平面
        vec = inlier_pts - center
        dist = vec @ normal
        projected = inlier_pts - np.outer(dist, normal)

        # 构造平面上的局部 2D 基
        if abs(normal[0]) < 0.9:
            u = np.cross(normal, [1, 0, 0])
        else:
            u = np.cross(normal, [0, 1, 0])
        u = u / np.linalg.norm(u)
        v = np.cross(normal, u)
        v = v / np.linalg.norm(v)

        # 投影到 2D 坐标
        vec2 = projected - center
        coords_2d = np.column_stack([vec2 @ u, vec2 @ v])

        # 2D PCA
        cov = np.cov(coords_2d.T)
        eig_vals, eig_vecs = np.linalg.eig(cov)
        idx = np.argsort(eig_vals)[::-1]
        eig_vecs = eig_vecs[:, idx]

        # 映射回 3D
        x_axis = eig_vecs[0, 0] * u + eig_vecs[1, 0] * v
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(normal, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)

        R = np.column_stack([x_axis, y_axis, normal])
        # 确保右手系
        if np.linalg.det(R) < 0:
            R[:, 0] *= -1
        return R

    def _compute_hole_positions(
        self, plane_center: np.ndarray, local_R: np.ndarray
    ) -> Dict[Tuple[int, int], np.ndarray]:
        """
        基于几何先验计算所有孔位在相机坐标系下的 3D 位置

        local_R 的列: [X_axis, Y_axis, Z_axis=normal]
        假设架面中心与孔位阵列中心对齐
        """
        holes = {}
        total_w = (self.cols - 1) * self.hole_spacing_x
        total_h = (self.rows - 1) * self.hole_spacing_y
        start_x = -total_w / 2
        start_y = -total_h / 2

        for row in range(self.rows):
            for col in range(self.cols):
                local_pos = np.array([
                    start_x + col * self.hole_spacing_x,
                    start_y + row * self.hole_spacing_y,
                    0.0,
                ])
                world_pos = plane_center + local_R @ local_pos
                holes[(row, col)] = world_pos

        return holes

    def _project_3d_to_2d(
        self, pt_3d: np.ndarray, K: np.ndarray
    ) -> Optional[Tuple[int, int]]:
        """3D 点投影到像素坐标"""
        if pt_3d[2] < 0.01:
            return None
        x = pt_3d[0] * K[0, 0] / pt_3d[2] + K[0, 2]
        y = pt_3d[1] * K[1, 1] / pt_3d[2] + K[1, 2]
        return (int(round(x)), int(round(y)))

    @staticmethod
    def _get_depth_at(depth: np.ndarray, pt_2d: Tuple[int, int]) -> float:
        """获取像素周围 5x5 ROI 的深度中值"""
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

    def _is_empty_hole(self, depth_val: float, plane_depth: float) -> bool:
        """
        判断孔位是否为空

        空孔：实测深度明显深于架面深度（光线穿过孔洞到更深处）
        有试管：实测深度接近架面深度（试管顶/管身反射）
        """
        if depth_val <= 0:
            return True
        # 深度值越大表示越远
        depth_diff = depth_val - plane_depth
        return depth_diff > self.depth_tube_threshold

    # ──────────────────────────────────────────────
    # 可视化
    # ──────────────────────────────────────────────

    def draw_detection(
        self, rgb_image: np.ndarray, result: RackDetectionResult
    ) -> np.ndarray:
        """在图像上绘制检测结果"""
        display = rgb_image.copy()

        if not result.success:
            cv2.putText(display, "Rack: Not detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return display

        cv2.putText(display, "Rack: Detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 绘制所有孔位
        if result.holes:
            for hole in result.holes.values():
                if hole.has_tube:
                    color = (0, 255, 0)       # 绿色=有试管
                    label = f"R{hole.index[0]}C{hole.index[1]}"
                else:
                    color = (100, 100, 255)    # 蓝色=空孔
                    label = f"R{hole.index[0]}C{hole.index[1]}"

                cv2.circle(display, hole.center_2d, 4, color, -1)
                cv2.putText(display, label,
                            (hole.center_2d[0] + 6, hole.center_2d[1] + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

        # 高亮源孔和目标孔
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

        # 显示架面法向量
        if result.rack_plane_center is not None and result.rack_normal is not None:
            cam_center = result.rack_plane_center
            normal_end = cam_center + result.rack_normal * 0.05
            # 投影到图像
            K_approx = np.array([[320, 0, 320], [0, 320, 240], [0, 0, 1]],
                                dtype=float)
            c2d = self._project_3d_to_2d(cam_center, K_approx)
            n2d = self._project_3d_to_2d(normal_end, K_approx)
            if c2d and n2d:
                cv2.arrowedLine(display, c2d, n2d, (0, 255, 255), 2)

        return display

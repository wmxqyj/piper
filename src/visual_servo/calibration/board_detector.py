"""
棋盘格角点检测与位姿估计

提供棋盘格内角点检测和棋盘格在相机坐标系下的位姿估计功能。
"""

import cv2
import numpy as np
from typing import Optional, Tuple


class ChessboardDetector:
    """棋盘格检测器，支持角点检测和位姿估计"""

    def __init__(self, pattern_size: Tuple[int, int], square_size_mm: float):
        """
        Parameters
        ----------
        pattern_size : (int, int)
            棋盘格内角点数 (columns, rows)，即 (宽, 高)
        square_size_mm : float
            每个方格边长（毫米）
        """
        self.pattern_size = tuple(pattern_size)
        self.square_size_m = square_size_mm / 1000.0  # 转为米

        # 生成棋盘格 3D 角点坐标（在棋盘格坐标系中，Z=0）
        self._object_points = self._create_object_points()

    def _create_object_points(self) -> np.ndarray:
        """生成棋盘格 3D 点坐标，原点在棋盘格左上角，X轴向右，Y轴向下"""
        cols, rows = self.pattern_size
        objp = np.zeros((rows * cols, 3), np.float32)
        objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
        objp *= self.square_size_m
        return objp

    def detect_corners(
        self,
        gray_image: np.ndarray,
        refine_subpixel: bool = True,
    ) -> Tuple[bool, Optional[np.ndarray]]:
        """
        检测棋盘格内角点

        Parameters
        ----------
        gray_image : np.ndarray
            灰度图像 (H, W)
        refine_subpixel : bool
            是否进行亚像素精炼

        Returns
        -------
        (success, corners)
            success : bool  检测是否成功
            corners : np.ndarray or None  角点像素坐标 (N, 1, 2)
        """
        ret, corners = cv2.findChessboardCorners(
            gray_image,
            self.pattern_size,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH
            | cv2.CALIB_CB_NORMALIZE_IMAGE
            | cv2.CALIB_CB_FAST_CHECK,
        )

        if ret and refine_subpixel:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray_image, corners, (11, 11), (-1, -1), criteria)

        return ret, corners

    def estimate_board_pose(
        self,
        corners: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: Optional[np.ndarray] = None,
    ) -> Tuple[bool, Optional[np.ndarray]]:
        """
        估计棋盘格在相机坐标系下的位姿

        棋盘格坐标系定义：
          - 原点：棋盘格左上角第一个内角点
          - X轴：沿棋盘格行方向向右
          - Y轴：沿棋盘格列方向向下
          - Z轴：垂直于棋盘格平面向外

        Parameters
        ----------
        corners : np.ndarray
            角点像素坐标 (N, 1, 2)
        camera_matrix : np.ndarray
            相机内参矩阵 (3, 3)
        dist_coeffs : np.ndarray, optional
            畸变系数，默认为 None（无畸变）

        Returns
        -------
        (success, board_in_cam)
            success : bool         求解是否成功
            board_in_cam : 4x4 齐次变换矩阵，棋盘格→相机坐标系
        """
        if dist_coeffs is None:
            dist_coeffs = np.zeros((4, 1))

        ret, rvec, tvec = cv2.solvePnP(
            self._object_points,
            corners,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not ret:
            return False, None

        # 旋转向量 → 旋转矩阵
        R, _ = cv2.Rodrigues(rvec)

        # 构建 4x4 齐次变换矩阵
        board_in_cam = np.eye(4)
        board_in_cam[:3, :3] = R
        board_in_cam[:3, 3] = tvec.flatten()

        return True, board_in_cam

    def compute_reprojection_error(
        self,
        corners: np.ndarray,
        board_in_cam: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: Optional[np.ndarray] = None,
    ) -> float:
        """
        计算重投影误差（像素）

        Parameters
        ----------
        corners : np.ndarray
            检测到的角点像素坐标
        board_in_cam : np.ndarray
            棋盘格→相机的 4x4 变换矩阵
        camera_matrix : np.ndarray
            相机内参矩阵
        dist_coeffs : np.ndarray, optional
            畸变系数

        Returns
        -------
        float
            平均重投影误差（像素）
        """
        if dist_coeffs is None:
            dist_coeffs = np.zeros((4, 1))

        R = board_in_cam[:3, :3]
        tvec = board_in_cam[:3, 3].reshape(3, 1)
        rvec, _ = cv2.Rodrigues(R)

        projected, _ = cv2.projectPoints(
            self._object_points, rvec, tvec, camera_matrix, dist_coeffs
        )

        error = np.mean(np.sqrt(np.sum((corners - projected) ** 2, axis=2)))
        return float(error)

    @staticmethod
    def draw_corners(
        image: np.ndarray,
        corners: np.ndarray,
        pattern_size: Tuple[int, int],
        success: bool,
    ) -> np.ndarray:
        """
        在图像上绘制检测到的棋盘格角点

        Parameters
        ----------
        image : np.ndarray
            原始图像 (BGR)
        corners : np.ndarray or None
            角点坐标
        pattern_size : (int, int)
            内角点数
        success : bool
            检测是否成功

        Returns
        -------
        np.ndarray
            绘制后的图像
        """
        img_draw = image.copy()
        if success and corners is not None:
            cv2.drawChessboardCorners(img_draw, pattern_size, corners, success)
        return img_draw

    @staticmethod
    def draw_board_axis(
        image: np.ndarray,
        board_in_cam: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: Optional[np.ndarray] = None,
        axis_length: float = 0.05,
    ) -> np.ndarray:
        """
        在图像上绘制棋盘格坐标系轴

        Parameters
        ----------
        image : np.ndarray
            原始图像 (BGR)
        board_in_cam : np.ndarray
            棋盘格→相机的 4x4 变换矩阵
        camera_matrix : np.ndarray
            相机内参矩阵
        dist_coeffs : np.ndarray, optional
            畸变系数
        axis_length : float
            坐标轴长度（米）

        Returns
        -------
        np.ndarray
            绘制后的图像
        """
        if dist_coeffs is None:
            dist_coeffs = np.zeros((4, 1))

        R = board_in_cam[:3, :3]
        tvec = board_in_cam[:3, 3].reshape(3, 1)
        rvec, _ = cv2.Rodrigues(R)

        axis_points = np.float32([
            [0, 0, 0],
            [axis_length, 0, 0],
            [0, axis_length, 0],
            [0, 0, axis_length],
        ])

        img_pts, _ = cv2.projectPoints(axis_points, rvec, tvec, camera_matrix, dist_coeffs)
        img_pts = img_pts.astype(int)

        origin = tuple(img_pts[0].ravel())
        img_draw = image.copy()
        img_draw = cv2.line(img_draw, origin, tuple(img_pts[1].ravel()), (0, 0, 255), 2)  # X: 红
        img_draw = cv2.line(img_draw, origin, tuple(img_pts[2].ravel()), (0, 255, 0), 2)  # Y: 绿
        img_draw = cv2.line(img_draw, origin, tuple(img_pts[3].ravel()), (255, 0, 0), 2)  # Z: 蓝

        return img_draw

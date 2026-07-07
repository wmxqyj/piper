"""
标定结果管理

提供标定结果的保存、加载和应用功能。
"""

import os
import yaml
import json
import numpy as np
from typing import Optional, Dict, Any
from dataclasses import dataclass, field


@dataclass
class CalibrationResult:
    """手眼标定结果"""
    gripper_T_cam: np.ndarray  # 4x4 齐次变换矩阵：末端法兰→相机
    rot_error_deg_mean: float  # 平均旋转误差（度）
    rot_error_deg_std: float   # 旋转误差标准差
    trans_error_m_mean: float  # 平均平移误差（米）
    trans_error_m_std: float   # 平移误差标准差
    position_std_mm: float     # 棋盘格在基坐标下的位置标准差（毫米）
    num_samples: int           # 有效样本数
    algorithm: str = "tsai_lenz"
    gripper_T_cam_pose6: list = field(default_factory=list)  # [x,y,z,roll,pitch,yaw]

    def __post_init__(self):
        if len(self.gripper_T_cam_pose6) == 0:
            self.gripper_T_cam_pose6 = self._to_pose6(self.gripper_T_cam)

    @staticmethod
    def _to_pose6(T: np.ndarray) -> list:
        """4x4 → [x, y, z, roll, pitch, yaw]"""
        x, y, z = T[:3, 3]
        R = T[:3, :3]
        pitch = float(np.arctan2(-R[2, 0], np.sqrt(R[0, 0]**2 + R[1, 0]**2)))
        if abs(pitch - np.pi / 2) < 1e-6:
            roll, yaw = 0.0, float(np.arctan2(R[0, 1], R[1, 1]))
        elif abs(pitch + np.pi / 2) < 1e-6:
            roll, yaw = 0.0, float(np.arctan2(-R[0, 1], R[1, 1]))
        else:
            roll = float(np.arctan2(R[2, 1], R[2, 2]))
            yaw = float(np.arctan2(R[1, 0], R[0, 0]))
        return [x, y, z, roll, pitch, yaw]


class CalibrationResultManager:
    """标定结果管理器"""

    @staticmethod
    def save(result: CalibrationResult, save_dir: str, filename: str = "calibration_result.yaml") -> str:
        """
        保存标定结果

        Parameters
        ----------
        result : CalibrationResult
            标定结果
        save_dir : str
            保存目录
        filename : str
            文件名

        Returns
        -------
        str
            保存的完整路径
        """
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, filename)

        data = {
            "calibration_result": {
                "algorithm": result.algorithm,
                "num_samples": result.num_samples,
                "rot_error_deg_mean": result.rot_error_deg_mean,
                "rot_error_deg_std": result.rot_error_deg_std,
                "trans_error_m_mean": result.trans_error_m_mean,
                "trans_error_m_std": result.trans_error_m_std,
                "position_std_mm": result.position_std_mm,
            },
            "gripper_T_cam": {
                "matrix": result.gripper_T_cam.tolist(),
                "pose6": {
                    "x_m": result.gripper_T_cam_pose6[0],
                    "y_m": result.gripper_T_cam_pose6[1],
                    "z_m": result.gripper_T_cam_pose6[2],
                    "roll_rad": result.gripper_T_cam_pose6[3],
                    "pitch_rad": result.gripper_T_cam_pose6[4],
                    "yaw_rad": result.gripper_T_cam_pose6[5],
                    "roll_deg": float(np.degrees(result.gripper_T_cam_pose6[3])),
                    "pitch_deg": float(np.degrees(result.gripper_T_cam_pose6[4])),
                    "yaw_deg": float(np.degrees(result.gripper_T_cam_pose6[5])),
                },
            },
        }

        with open(filepath, "w") as f:
            yaml.dump(data, f, default_flow_style=None, sort_keys=False, allow_unicode=True)

        # 同时保存 numpy 格式
        np_path = os.path.join(save_dir, "gripper_T_cam.npy")
        np.save(np_path, result.gripper_T_cam)

        print(f"\n标定结果已保存:")
        print(f"  YAML: {filepath}")
        print(f"  NumPy: {np_path}")
        print(f"  矩阵 (gripper_T_cam):")
        CalibrationResultManager._print_matrix(result.gripper_T_cam)

        return filepath

    @staticmethod
    def load(filepath: str) -> Optional[CalibrationResult]:
        """加载标定结果"""
        if not os.path.exists(filepath):
            print(f"文件不存在: {filepath}")
            return None

        with open(filepath, "r") as f:
            data = yaml.safe_load(f)

        result_data = data["calibration_result"]
        matrix_data = data["gripper_T_cam"]["matrix"]

        gripper_T_cam = np.array(matrix_data)

        return CalibrationResult(
            gripper_T_cam=gripper_T_cam,
            rot_error_deg_mean=result_data["rot_error_deg_mean"],
            rot_error_deg_std=result_data["rot_error_deg_std"],
            trans_error_m_mean=result_data["trans_error_m_mean"],
            trans_error_m_std=result_data["trans_error_m_std"],
            position_std_mm=result_data["position_std_mm"],
            num_samples=result_data["num_samples"],
            algorithm=result_data.get("algorithm", "tsai_lenz"),
        )

    @staticmethod
    def apply_to_camera_extrinsics(
        gripper_T_cam: np.ndarray,
        gripper_pose: np.ndarray,
    ) -> np.ndarray:
        """
        计算机器人当前位姿下的相机外参

        camera_extrinsics = base_T_gripper * gripper_T_cam

        Parameters
        ----------
        gripper_T_cam : np.ndarray
            手眼标定矩阵 (4x4)
        gripper_pose : np.ndarray
            当前机械臂末端法兰位姿 [x, y, z, roll, pitch, yaw] 或 4x4 矩阵

        Returns
        -------
        np.ndarray
            相机外参矩阵 (4x4)：相机在基坐标系下的位姿
        """
        from .solver import HandEyeSolver

        if gripper_pose.shape == (6,):
            base_T_gripper = HandEyeSolver.pose6_to_matrix(gripper_pose)
        else:
            base_T_gripper = gripper_pose

        return base_T_gripper @ gripper_T_cam

    @staticmethod
    def _print_matrix(T: np.ndarray):
        """打印 4x4 矩阵"""
        for i in range(4):
            row = "  ".join(f"{T[i, j]:10.6f}" for j in range(4))
            print(f"    [{row}]")

    @staticmethod
    def print_summary(result: CalibrationResult):
        """打印标定结果摘要"""
        print("\n" + "=" * 60)
        print("手眼标定结果摘要")
        print("=" * 60)
        print(f"算法: {result.algorithm}")
        print(f"有效样本数: {result.num_samples}")
        print(f"\n旋转误差:")
        print(f"  均值: {result.rot_error_deg_mean:.4f}°")
        print(f"  标准差: {result.rot_error_deg_std:.4f}°")
        print(f"\n平移误差:")
        print(f"  均值: {result.trans_error_m_mean*1000:.4f} mm")
        print(f"  标准差: {result.trans_error_m_std*1000:.4f} mm")
        print(f"\n棋盘格位置一致性:")
        print(f"  标准差: {result.position_std_mm:.4f} mm")
        print(f"\ngripper_T_cam (末端→相机):")
        for i in range(4):
            row = "  ".join(f"{result.gripper_T_cam[i, j]:10.6f}" for j in range(4))
            print(f"  [{row}]")
        print(f"\n位姿 (x, y, z, roll, pitch, yaw):")
        p = result.gripper_T_cam_pose6
        print(f"  {p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f} m")
        print(f"  {np.degrees(p[3]):.2f}, {np.degrees(p[4]):.2f}, {np.degrees(p[5]):.2f}°")
        print("=" * 60)

"""
AX=XB 手眼标定求解器

实现 Tsai-Lenz 和 Park 算法，用于求解手眼标定矩阵。
"""

import numpy as np
from typing import List, Tuple


class HandEyeSolver:
    """手眼标定 AX=XB 求解器"""

    SUPPORTED_ALGORITHMS = ("tsai_lenz", "park")

    @staticmethod
    def pose6_to_matrix(pose: List[float]) -> np.ndarray:
        """
        将 [x, y, z, roll, pitch, yaw] 转为 4x4 齐次变换矩阵

        旋转约定：R = Rz(yaw) * Ry(pitch) * Rx(roll)
        """
        x, y, z, roll, pitch, yaw = pose
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)

        R = np.array([
            [cy * cp,  cy * sp * sr - sy * cr,  cy * sp * cr + sy * sr],
            [sy * cp,  sy * sp * sr + cy * cr,  sy * sp * cr - cy * sr],
            [-sp,      cp * sr,                  cp * cr],
        ])

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [x, y, z]
        return T

    @staticmethod
    def matrix_to_pose6(T: np.ndarray) -> List[float]:
        """将 4x4 齐次变换矩阵转为 [x, y, z, roll, pitch, yaw]"""
        x, y, z = T[:3, 3]
        R = T[:3, :3]

        # R = Rz(yaw) * Ry(pitch) * Rx(roll)
        pitch = np.arctan2(-R[2, 0], np.sqrt(R[0, 0]**2 + R[1, 0]**2))
        if np.abs(pitch - np.pi / 2) < 1e-6:
            roll = 0.0
            yaw = np.arctan2(R[0, 1], R[1, 1])
        elif np.abs(pitch + np.pi / 2) < 1e-6:
            roll = 0.0
            yaw = np.arctan2(-R[0, 1], R[1, 1])
        else:
            roll = np.arctan2(R[2, 1], R[2, 2])
            yaw = np.arctan2(R[1, 0], R[0, 0])

        return [x, y, z, roll, pitch, yaw]

    @staticmethod
    def _build_AB_pairs(
        gripper_poses: List[np.ndarray],
        board_in_cam_poses: List[np.ndarray],
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        构建 A、B 矩阵对

        A_i_j = inv(gripper_j) * gripper_i
        B_i_j = board_in_cam_j * inv(board_in_cam_i)

        使得 A * X = X * B
        """
        A_list, B_list = [], []
        n = len(gripper_poses)

        for i in range(n):
            for j in range(i + 1, n):
                # A = gripper_j^{-1} * gripper_i
                A = np.linalg.inv(gripper_poses[j]) @ gripper_poses[i]
                # B = board_in_cam_j * board_in_cam_i^{-1}
                B = board_in_cam_poses[j] @ np.linalg.inv(board_in_cam_poses[i])

                A_list.append(A)
                B_list.append(B)

        return A_list, B_list

    def solve_eye_in_hand(
        self,
        gripper_poses: List[np.ndarray],
        board_in_cam_poses: List[np.ndarray],
        algorithm: str = "tsai_lenz",
    ) -> Tuple[np.ndarray, dict]:
        """
        求解 Eye-in-Hand 手眼标定矩阵 X

        AX = XB, 其中 X = gripper_T_cam (相机在末端法兰坐标系下的位姿)

        Parameters
        ----------
        gripper_poses : List[np.ndarray]
            机械臂末端法兰在基坐标系下的位姿列表，每个为 [x, y, z, roll, pitch, yaw]
        board_in_cam_poses : List[np.ndarray]
            棋盘格在相机坐标系下的位姿列表，每个为 4x4 齐次变换矩阵
        algorithm : str
            求解算法，支持 "tsai_lenz" 或 "park"

        Returns
        -------
        (gripper_T_cam, info)
            gripper_T_cam : 4x4 齐次变换矩阵
            info : dict, 包含求解信息和误差指标
        """
        if len(gripper_poses) < 3:
            raise ValueError(f"至少需要 3 组数据，当前只有 {len(gripper_poses)} 组")

        if len(gripper_poses) != len(board_in_cam_poses):
            raise ValueError(
                f"gripper_poses ({len(gripper_poses)}) 和 "
                f"board_in_cam_poses ({len(board_in_cam_poses)}) 数量不匹配"
            )

        # 转换为齐次变换矩阵
        gripper_T = [
            self.pose6_to_matrix(p) if p.shape == (6,) else p
            for p in gripper_poses
        ]

        # 构建 A、B 对
        A_list, B_list = self._build_AB_pairs(gripper_T, board_in_cam_poses)

        if algorithm == "tsai_lenz":
            X = self._solve_tsai_lenz(A_list, B_list)
        elif algorithm == "park":
            X = self._solve_park(A_list, B_list)
        else:
            raise ValueError(f"不支持的算法: {algorithm}，可选: {self.SUPPORTED_ALGORITHMS}")

        # 计算误差
        info = self._compute_error(X, A_list, B_list)

        return X, info

    def _solve_tsai_lenz(
        self, A_list: List[np.ndarray], B_list: List[np.ndarray]
    ) -> np.ndarray:
        """
        Tsai-Lenz 算法求解 AX=XB

        分两步：
          1. 先求解旋转部分 R_X
          2. 再求解平移部分 t_X
        """
        # --- Step 1: 求解旋转 ---
        # 对于每对 (A, B)，提取旋转矩阵并转为轴角表示
        P_A_list = []
        P_B_list = []

        for A, B in zip(A_list, B_list):
            R_A = A[:3, :3]
            R_B = B[:3, :3]

            # 计算旋转轴 (P_A, P_B)
            # R = I + sin(theta) * [k]x + (1-cos(theta)) * [k]x^2
            # 用 Rodrigues 公式，轴角 = theta * k
            theta_A = np.arccos(np.clip((np.trace(R_A) - 1) / 2, -1.0, 1.0))
            theta_B = np.arccos(np.clip((np.trace(R_B) - 1) / 2, -1.0, 1.0))

            if theta_A < 1e-10 or theta_B < 1e-10:
                # 运动过小，跳过这组
                continue

            k_A = np.array([
                R_A[2, 1] - R_A[1, 2],
                R_A[0, 2] - R_A[2, 0],
                R_A[1, 0] - R_A[0, 1],
            ])
            k_A_norm = np.linalg.norm(k_A)
            if k_A_norm > 1e-10:
                k_A = k_A / k_A_norm
            P_A = 2 * np.sin(theta_A / 2) * k_A

            k_B = np.array([
                R_B[2, 1] - R_B[1, 2],
                R_B[0, 2] - R_B[2, 0],
                R_B[1, 0] - R_B[0, 1],
            ])
            k_B_norm = np.linalg.norm(k_B)
            if k_B_norm > 1e-10:
                k_B = k_B / k_B_norm
            P_B = 2 * np.sin(theta_B / 2) * k_B

            P_A_list.append(P_A)
            P_B_list.append(P_B)

        if len(P_A_list) < 3:
            raise ValueError(
                f"有效运动对不足 ({len(P_A_list)})，"
                f"请确保在采集时机械臂在各位置之间的旋转变化足够大"
            )

        P_A_mat = np.array(P_A_list)  # (M, 3)
        P_B_mat = np.array(P_B_list)  # (M, 3)

        # 用最小二乘法求解 R_X * P_B = P_A
        # P_A^T * P_A * k = P_A^T * P_B ... 实际用王浚民的方法
        # 更直接：用伪逆 P_A = P_B * R_X^T
        # 所以 R_X = (P_B^+ * P_A)^T
        # 或者转置视角：P_A^T = R_X * P_B^T

        # 标准 Tsai-Lenz: 构建线性系统求解 R_X 的轴角
        # 使用 SVD 方法: R_X = (P_B^T * P_B)^{-1} * P_B^T * P_A 的正交化投影
        M = P_B_mat.T @ P_B_mat
        if np.linalg.det(M) < 1e-10:
            raise ValueError("旋转矩阵奇异，请检查采集数据中位姿变化是否足够多样")

        R_X_approx = np.linalg.inv(M) @ (P_B_mat.T @ P_A_mat)

        # 将近似结果投影到 SO(3)（通过 SVD）
        U, _, Vt = np.linalg.svd(R_X_approx.T)
        R_X = U @ Vt
        if np.linalg.det(R_X) < 0:
            Vt[-1, :] *= -1
            R_X = U @ Vt

        # --- Step 2: 求解平移 ---
        # (R_A - I) * t_X = R_X * t_B - t_A
        C_list = []
        d_list = []

        for A, B in zip(A_list, B_list):
            R_A = A[:3, :3]
            t_A = A[:3, 3]
            R_B = B[:3, :3]
            t_B = B[:3, 3]

            C = R_A - np.eye(3)
            d = R_X @ t_B - t_A

            C_list.append(C)
            d_list.append(d)

        if len(C_list) > 0:
            C_stack = np.vstack(C_list)
            d_stack = np.hstack(d_list)
            t_X, _, _, _ = np.linalg.lstsq(C_stack, d_stack, rcond=None)
        else:
            t_X = np.zeros(3)

        # 构建最终结果
        X = np.eye(4)
        X[:3, :3] = R_X
        X[:3, 3] = t_X

        return X

    def _solve_park(
        self, A_list: List[np.ndarray], B_list: List[np.ndarray]
    ) -> np.ndarray:
        """
        Park 算法（基于李代数/旋量）

        将旋转矩阵用旋量表示，通过 SVD 求解
        """
        # 提取旋转矩阵
        M_list = []
        for A, B in zip(A_list, B_list):
            R_A = A[:3, :3]
            R_B = B[:3, :3]
            # Park: log(R_A) 和 log(R_B) 的关系
            log_R_A = cv2.Rodrigues(R_A)[0].flatten() if hasattr(cv2, 'Rodrigues') else self._log_so3(R_A)
            log_R_B = cv2.Rodrigues(R_B)[0].flatten() if hasattr(cv2, 'Rodrigues') else self._log_so3(R_B)
            M_list.append(np.outer(log_R_B, log_R_A))

        # 用 SVD 求最优旋转
        M_sum = np.sum(M_list, axis=0)
        U, _, Vt = np.linalg.svd(M_sum)
        R_X = U @ Vt
        if np.linalg.det(R_X) < 0:
            Vt[-1, :] *= -1
            R_X = U @ Vt

        # 平移：同 Tsai-Lenz
        C_list, d_list = [], []
        for A, B in zip(A_list, B_list):
            C = A[:3, :3] - np.eye(3)
            d = R_X @ B[:3, 3] - A[:3, 3]
            C_list.append(C)
            d_list.append(d)

        C_stack = np.vstack(C_list)
        d_stack = np.hstack(d_list)
        t_X, _, _, _ = np.linalg.lstsq(C_stack, d_stack, rcond=None)

        X = np.eye(4)
        X[:3, :3] = R_X
        X[:3, 3] = t_X
        return X

    @staticmethod
    def _log_so3(R: np.ndarray) -> np.ndarray:
        """SO(3) → so(3) 对数映射（轴角）"""
        theta = np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))
        if theta < 1e-10:
            return np.zeros(3)
        w = np.array([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1],
        ])
        return (theta / (2 * np.sin(theta))) * w

    @staticmethod
    def _compute_error(X: np.ndarray, A_list: List[np.ndarray], B_list: List[np.ndarray]) -> dict:
        """
        计算标定误差

        Returns
        -------
        dict: {
            'rot_error_deg': 平均旋转误差（度）
            'trans_error_m': 平均平移误差（米）
            'individual_errors': [(rot_err, trans_err), ...]
        }
        """
        rot_errors = []
        trans_errors = []

        for A, B in zip(A_list, B_list):
            # 计算 AX 和 XB
            AX = A @ X
            XB = X @ B

            # 旋转误差
            R_diff = AX[:3, :3] @ XB[:3, :3].T
            rot_err = np.degrees(
                np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1.0, 1.0))
            )

            # 平移误差
            trans_err = np.linalg.norm(AX[:3, 3] - XB[:3, 3])

            rot_errors.append(rot_err)
            trans_errors.append(trans_err)

        return {
            "rot_error_deg_mean": float(np.mean(rot_errors)),
            "rot_error_deg_std": float(np.std(rot_errors)),
            "trans_error_m_mean": float(np.mean(trans_errors)),
            "trans_error_m_std": float(np.std(trans_errors)),
            "individual_errors": list(zip(rot_errors, trans_errors)),
        }

    @staticmethod
    def verify_calibration(
        gripper_T_cam: np.ndarray,
        gripper_poses: List[np.ndarray],
        board_in_cam_poses: List[np.ndarray],
    ) -> dict:
        """
        验证标定结果：计算棋盘格在基坐标系下的位姿一致性

        理论上: base_T_board = base_T_gripper * gripper_T_cam * cam_T_board
        对所有位姿应该一致

        Returns
        -------
        dict: {
            'board_in_base_poses': List[np.ndarray],
            'mean_position': [...],
            'std_position': [...],
            'position_std_mm': float,
        }
        """
        board_in_base_list = []

        for gripper_p, board_in_cam in zip(gripper_poses, board_in_cam_poses):
            if isinstance(gripper_p, list) or gripper_p.shape == (6,):
                base_T_gripper = HandEyeSolver.pose6_to_matrix(gripper_p)
            else:
                base_T_gripper = gripper_p

            base_T_board = base_T_gripper @ gripper_T_cam @ board_in_cam
            board_in_base_list.append(base_T_board)

        # 计算位置的一致性
        positions = np.array([T[:3, 3] for T in board_in_base_list])
        mean_pos = np.mean(positions, axis=0)
        std_pos = np.std(positions, axis=0)

        return {
            "board_in_base_poses": board_in_base_list,
            "mean_position": mean_pos.tolist(),
            "std_position": std_pos.tolist(),
            "position_std_mm": float(np.linalg.norm(std_pos) * 1000),
        }


# 为了 Park 算法中可能的 cv2 导入
try:
    import cv2
except ImportError:
    cv2 = None

"""
数据同步机制

同步机械臂状态和相机数据，为 RLBench 格式转换做准备
"""

import time
import numpy as np
from typing import Optional, Dict, Any
from dataclasses import dataclass

from piper_interface import PiperArmState
from camera_interface import CameraData


def pose6_to_matrix(pose6: np.ndarray) -> np.ndarray:
    """
    将 [x, y, z, roll, pitch, yaw] 转换为 4x4 齐次变换矩阵

    Parameters
    ----------
    pose6 : np.ndarray
        [x, y, z, roll, pitch, yaw]，角度单位 rad

    Returns
    -------
    np.ndarray
        4x4 齐次变换矩阵
    """
    x, y, z = pose6[:3]
    roll, pitch, yaw = pose6[3:]

    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    R = np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2],
    ])

    matrix = np.eye(4)
    matrix[:3, :3] = R
    matrix[:3, 3] = [x, y, z]
    return matrix


@dataclass
class SyncedFrame:
    """同步后的数据帧"""
    timestamp: float

    # 机械臂数据
    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    joint_forces: np.ndarray
    gripper_pose: np.ndarray
    gripper_positions: np.ndarray
    gripper_open: float

    # 相机数据
    front_rgb: np.ndarray
    front_depth: np.ndarray

    # 相机参数（第一帧时设置）
    front_camera_intrinsics: Optional[np.ndarray] = None
    front_camera_extrinsics: Optional[np.ndarray] = None

    # 可选字段（用于后续处理）
    frame_id: int = 0


class DataSyncManager:
    """数据同步管理器"""

    def __init__(self, config: Dict[str, Any], gripper_T_cam: Optional[np.ndarray] = None):
        """
        初始化数据同步管理器

        Parameters
        ----------
        config : Dict[str, Any]
            配置字典
        gripper_T_cam : np.ndarray, optional
            手眼标定矩阵 (4x4)，即末端法兰到相机的变换矩阵。
            如果提供，则外参通过 base_T_gripper @ gripper_T_cam 实时计算；
            如果不提供，则使用相机接口返回的外参（通常为 np.eye(4)）。
        """
        self.config = config
        self.frame_buffer = []
        self.frame_counter = 0

        self.gripper_T_cam = gripper_T_cam

        # 相机参数（第一帧时设置）
        self.camera_intrinsics_set = False
        self.saved_intrinsics = None
        self.saved_extrinsics = None

        # 同步参数
        self.max_time_diff = config.get('max_time_diff', 0.05)  # 最大时间差 50ms

    def sync_data(
        self,
        arm_state: PiperArmState,
        camera_data: CameraData,
        frame_id: Optional[int] = None
    ) -> Optional[SyncedFrame]:
        """
        同步机械臂状态和相机数据

        Parameters
        ----------
        arm_state : PiperArmState
            机械臂状态
        camera_data : CameraData
            相机数据
        frame_id : int, optional
            帧ID，如果不提供则自动生成

        Returns
        -------
        SyncedFrame | None
            同步后的数据帧，如果时间差太大则返回 None
        """
        # 检查时间差
        time_diff = abs(arm_state.timestamp - camera_data.timestamp)
        if time_diff > self.max_time_diff:
            print(f"[WARN] 时间差过大: {time_diff:.3f}s，跳过此帧")
            return None

        # 使用机械臂的时间戳作为主导时间
        timestamp = arm_state.timestamp

        # 设置帧ID
        if frame_id is None:
            frame_id = self.frame_counter
            self.frame_counter += 1

        # 相机内参（仅第一帧保存）
        intrinsics = None
        if not self.camera_intrinsics_set:
            self.saved_intrinsics = camera_data.camera_intrinsics
            self.camera_intrinsics_set = True
            intrinsics = camera_data.camera_intrinsics

        # 相机外参
        if self.gripper_T_cam is not None:
            extrinsics = pose6_to_matrix(arm_state.gripper_pose) @ self.gripper_T_cam
        else:
            extrinsics = None
            if self.saved_extrinsics is None:
                self.saved_extrinsics = camera_data.camera_extrinsics
                extrinsics = camera_data.camera_extrinsics

        # 创建同步帧
        synced_frame = SyncedFrame(
            timestamp=timestamp,
            joint_positions=arm_state.joint_positions,
            joint_velocities=arm_state.joint_velocities,
            joint_forces=arm_state.joint_forces,
            gripper_pose=arm_state.gripper_pose,
            gripper_positions=arm_state.gripper_positions,
            gripper_open=arm_state.gripper_open,
            front_rgb=camera_data.rgb_image,
            front_depth=camera_data.depth_image,
            front_camera_intrinsics=intrinsics,
            front_camera_extrinsics=extrinsics,
            frame_id=frame_id,
        )

        return synced_frame

    def add_frame_to_buffer(self, frame: SyncedFrame):
        """
        将帧添加到缓冲区

        Parameters
        ----------
        frame : SyncedFrame
            同步后的数据帧
        """
        self.frame_buffer.append(frame)

    def get_buffer(self) -> list:
        """
        获取缓冲区中的所有帧

        Returns
        -------
        list
            帧列表
        """
        return self.frame_buffer

    def clear_buffer(self):
        """清空缓冲区"""
        self.frame_buffer.clear()
        self.frame_counter = 0

    def get_frame_count(self) -> int:
        """
        获取缓冲区中的帧数

        Returns
        -------
        int
            帧数
        """
        return len(self.frame_buffer)

    def get_camera_params(self) -> tuple:
        """
        获取保存的相机参数

        Returns
        -------
        tuple
            (intrinsics, extrinsics)
        """
        return (self.saved_intrinsics, self.saved_extrinsics)

    def print_stats(self):
        """打印统计信息"""
        print(f"缓冲区帧数: {self.get_frame_count()}")

        if len(self.frame_buffer) > 0:
            first_frame = self.frame_buffer[0]
            last_frame = self.frame_buffer[-1]

            duration = last_frame.timestamp - first_frame.timestamp
            avg_fps = len(self.frame_buffer) / duration if duration > 0 else 0

            print(f"录制时长: {duration:.2f}s")
            print(f"平均帧率: {avg_fps:.1f}fps")
            print(f"第一帧ID: {first_frame.frame_id}")
            print(f"最后一帧ID: {last_frame.frame_id}")


# 测试代码
if __name__ == '__main__':
    # 创建测试数据
    arm_state = PiperArmState(
        timestamp=time.time(),
        joint_positions=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
        joint_velocities=np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06]),
        joint_forces=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        gripper_pose=np.array([0.5, 0.3, 0.2, 0.1, 0.2, 0.3]),
        gripper_positions=np.array([0.02, 0.02]),
        gripper_open=1.0,
    )

    camera_data = CameraData(
        timestamp=time.time(),
        rgb_image=np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        depth_image=np.random.randint(0, 1000, (480, 640), dtype=np.uint16),
        camera_intrinsics=np.array([[640, 0, 320], [0, 480, 240], [0, 0, 1]]),
        camera_extrinsics=np.eye(4),
    )

    # 创建同步管理器
    config = {'max_time_diff': 0.05}
    sync_manager = DataSyncManager(config)

    # 同步数据
    synced_frame = sync_manager.sync_data(arm_state, camera_data)
    if synced_frame is not None:
        print("同步成功:")
        print(f"  时间戳: {synced_frame.timestamp}")
        print(f"  关节角度: {synced_frame.joint_positions}")
        print(f"  RGB尺寸: {synced_frame.front_rgb.shape}")
        print(f"  深度尺寸: {synced_frame.front_depth.shape}")
        print(f"  帧ID: {synced_frame.frame_id}")

        # 添加到缓冲区
        sync_manager.add_frame_to_buffer(synced_frame)

        # 打印统计
        sync_manager.print_stats()
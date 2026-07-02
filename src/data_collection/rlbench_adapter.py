"""
RLBench 格式适配器（6关节版本）

适配 Piper 6关节机械臂到 RLBench 格式
"""

import numpy as np
import pickle
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

from data_sync import SyncedFrame


@dataclass
class Observation:
    """
    RLBench Observation 类（6关节版本）

    基于 rlbench.backend.observation.Observation，适配 Piper 6关节机械臂
    """
    # 视觉数据（只保留 front 相机）
    front_rgb: Optional[np.ndarray] = None
    front_depth: Optional[np.ndarray] = None
    front_point_cloud: Optional[np.ndarray] = None

    # 关节状态（6关节）
    joint_velocities: Optional[np.ndarray] = None  # shape: (6,)
    joint_positions: Optional[np.ndarray] = None  # shape: (6,)
    joint_forces: Optional[np.ndarray] = None  # shape: (6,)

    # 末端状态
    gripper_open: float = 0.0  # 0.0 或 1.0
    gripper_pose: Optional[np.ndarray] = None  # shape: (7) [x, y, z, qx, qy, qz, qw]
    gripper_matrix: Optional[np.ndarray] = None  # shape: (4, 4)
    gripper_joint_positions: Optional[np.ndarray] = None  # shape: (2)

    # 其他字段（暂时不使用）
    gripper_touch_forces: Optional[np.ndarray] = None
    task_low_dim_state: Optional[np.ndarray] = None
    ignore_collisions: bool = True
    misc: Optional[Dict[str, Any]] = None


@dataclass
class Demo:
    """
    RLBench Demo 类（6关节版本）

    基于 rlbench.demo.Demo
    """
    _observations: List[Observation]
    _random_seed: int = 42
    variation_number: int = 0

    def __len__(self):
        return len(self._observations)

    def __getitem__(self, idx: int):
        return self._observations[idx]

    @property
    def observations(self):
        return self._observations

    @property
    def random_seed(self):
        return self._random_seed


class RLBenchAdapter:
    """RLBench 格式适配器"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化适配器

        Parameters
        ----------
        config : Dict[str, Any]
            配置字典，包含：
            - joint_nums: 关节数量（6）
            - camera_near: 相机近距离
            - camera_far: 相机远距离
        """
        self.config = config
        self.joint_nums = config.get('joint_nums', 6)
        self.camera_near = config.get('camera_near', 0.5)
        self.camera_far = config.get('camera_far', 4.5)

    def convert_frame_to_observation(
        self,
        frame: SyncedFrame,
        keypoint_idxs: Optional[List[int]] = None
    ) -> Observation:
        """
        将同步帧转换为 RLBench Observation

        Parameters
        ----------
        frame : SyncedFrame
            同步后的数据帧
        keypoint_idxs : List[int], optional
            关键帧索引列表

        Returns
        -------
        Observation
            RLBench 格式的观测数据
        """
        # 转换末端位姿格式
        # Piper 返回的是 [x, y, z, roll, pitch, yaw]
        # RLBench 需要 [x, y, z, qx, qy, qz, qw]
        gripper_pose_xyz = frame.gripper_pose[:3]  # [x, y, z]
        gripper_pose_rpy = frame.gripper_pose[3:]  # [roll, pitch, yaw]

        # 将欧拉角转换为四元数（使用 scipy 或手动实现）
        gripper_pose_quat = self._euler_to_quaternion(gripper_pose_rpy)

        # 组合成 [x, y, z, qx, qy, qz, qw]
        gripper_pose = np.concatenate([gripper_pose_xyz, gripper_pose_quat])

        # 转换末端矩阵（4x4）
        gripper_matrix = self._pose_to_matrix(gripper_pose)

        # 构建 misc 字典
        misc = {}

        # 相机参数（只在第一帧有）
        if frame.front_camera_intrinsics is not None:
            misc['front_camera_intrinsics'] = frame.front_camera_intrinsics
        if frame.front_camera_extrinsics is not None:
            misc['front_camera_extrinsics'] = frame.front_camera_extrinsics

        # 相机近距离和远距离
        misc['front_camera_near'] = self.camera_near
        misc['front_camera_far'] = self.camera_far

        # 关键帧索引
        if keypoint_idxs is not None:
            misc['keypoint_idxs'] = np.array(keypoint_idxs)

        # 创建 Observation
        obs = Observation(
            front_rgb=frame.front_rgb,
            front_depth=frame.front_depth,
            front_point_cloud=None,  # 可以从深度图生成，暂时不实现
            joint_velocities=frame.joint_velocities,
            joint_positions=frame.joint_positions,
            joint_forces=frame.joint_forces,
            gripper_open=frame.gripper_open,
            gripper_pose=gripper_pose,
            gripper_matrix=gripper_matrix,
            gripper_joint_positions=frame.gripper_positions,
            misc=misc,
        )

        return obs

    def convert_frames_to_demo(
        self,
        frames: List[SyncedFrame],
        random_seed: int = 42,
        variation_number: int = 0
    ) -> Demo:
        """
        将帧列表转换为 RLBench Demo

        Parameters
        ----------
        frames : List[SyncedFrame]
            同步后的帧列表
        random_seed : int
            随机种子
        variation_number : int
            variation 编号

        Returns
        -------
        Demo
            RLBench 格式的 Demo
        """
        observations = []

        for frame in frames:
            obs = self.convert_frame_to_observation(frame)
            observations.append(obs)

        # 第一帧设置完整的相机参数
        if len(observations) > 0:
            first_obs = observations[0]
            for obs in observations:
                # 将第一帧的相机参数复制到所有帧
                if obs.misc is not None and first_obs.misc is not None:
                    for key in ['front_camera_intrinsics', 'front_camera_extrinsics',
                                'front_camera_near', 'front_camera_far']:
                        if key in first_obs.misc and key not in obs.misc:
                            obs.misc[key] = first_obs.misc[key]

        demo = Demo(
            _observations=observations,
            _random_seed=random_seed,
            variation_number=variation_number,
        )

        return demo

    def _euler_to_quaternion(self, euler: np.ndarray) -> np.ndarray:
        """
        将欧拉角（roll, pitch, yaw）转换为四元数（qx, qy, qz, qw）

        使用 ZYX intrinsic (body frame) convention

        Parameters
        ----------
        euler : np.ndarray
            [roll, pitch, yaw] 单位：rad

        Returns
        -------
        np.ndarray
            [qx, qy, qz, qw]
        """
        roll, pitch, yaw = euler

        # 计算四元数
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)

        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy

        return np.array([qx, qy, qz, qw])

    def _pose_to_matrix(self, pose: np.ndarray) -> np.ndarray:
        """
        将位姿 [x, y, z, qx, qy, qz, qw] 转换为 4x4 矩阵

        Parameters
        ----------
        pose : np.ndarray
            [x, y, z, qx, qy, qz, qw]

        Returns
        -------
        np.ndarray
            4x4 变换矩阵
        """
        # 位置
        x, y, z = pose[:3]

        # 四元数
        qx, qy, qz, qw = pose[3:]

        # 构建旋转矩阵
        # 参考：https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
        R = np.array([
            [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
            [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
            [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2],
        ])

        # 构建4x4矩阵
        matrix = np.eye(4)
        matrix[:3, :3] = R
        matrix[:3, 3] = [x, y, z]

        return matrix

    def save_demo(self, demo: Demo, save_path: str, episode_idx: int):
        """
        保存 Demo 到文件（RLBench 格式）

        Parameters
        ----------
        demo : Demo
            RLBench Demo 对象
        save_path : str
            保存路径
        episode_idx : int
            episode 编号
        """
        import os

        # 创建目录结构
        episode_path = os.path.join(save_path, 'all_variations', 'episodes', f'episode{episode_idx}')
        os.makedirs(episode_path, exist_ok=True)

        # 创建子目录
        front_rgb_path = os.path.join(episode_path, 'front_rgb')
        front_depth_path = os.path.join(episode_path, 'front_depth')
        os.makedirs(front_rgb_path, exist_ok=True)
        os.makedirs(front_depth_path, exist_ok=True)

        # 保存图像和低维数据
        observations = []
        for i, obs in enumerate(demo.observations):
            # 保存 RGB 图像
            if obs.front_rgb is not None:
                import cv2
                rgb_filename = os.path.join(front_rgb_path, f'{i}.png')
                cv2.imwrite(rgb_filename, obs.front_rgb)

            # 保存深度图像
            if obs.front_depth is not None:
                import cv2
                depth_filename = os.path.join(front_depth_path, f'{i}.png')
                cv2.imwrite(depth_filename, obs.front_depth)

            # 添加到观测列表（不包含图像数据）
            observations.append(obs)

        # 保存低维数据（pickle）
        low_dim_obs_path = os.path.join(episode_path, 'low_dim_obs.pkl')
        with open(low_dim_obs_path, 'wb') as f:
            pickle.dump(demo, f)

        # 保存 variation_number
        variation_number_path = os.path.join(episode_path, 'variation_number.pkl')
        with open(variation_number_path, 'wb') as f:
            pickle.dump(demo.variation_number, f)

        print(f"已保存 Demo 到 {episode_path}，共 {len(demo)} 帧")


# 测试代码
if __name__ == '__main__':
    # 创建测试数据
    from data_sync import SyncedFrame
    import time

    frame = SyncedFrame(
        timestamp=time.time(),
        joint_positions=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
        joint_velocities=np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06]),
        joint_forces=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        gripper_pose=np.array([0.5, 0.3, 0.2, 0.1, 0.2, 0.3]),  # [x,y,z,r,p,y]
        gripper_positions=np.array([0.02, 0.02]),
        gripper_open=1.0,
        front_rgb=np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        front_depth=np.random.randint(0, 1000, (480, 640), dtype=np.uint16),
        front_camera_intrinsics=np.array([[640, 0, 320], [0, 480, 240], [0, 0, 1]]),
        front_camera_extrinsics=np.eye(4),
    )

    # 创建适配器
    config = {
        'joint_nums': 6,
        'camera_near': 0.5,
        'camera_far': 4.5,
    }
    adapter = RLBenchAdapter(config)

    # 转换帧
    obs = adapter.convert_frame_to_observation(frame)

    print("转换成功:")
    print(f"  关节角度: {obs.joint_positions}")
    print(f"  关节速度: {obs.joint_velocities}")
    print(f"  关节力矩: {obs.joint_forces}")
    print(f"  末端位姿: {obs.gripper_pose}")
    print(f"  末端矩阵: {obs.gripper_matrix}")
    print(f"  RGB尺寸: {obs.front_rgb.shape if obs.front_rgb is not None else None}")

    # 测试四元数转换
    euler = np.array([0.1, 0.2, 0.3])
    quat = adapter._euler_to_quaternion(euler)
    print(f"欧拉角 {euler} -> 四元数 {quat}")
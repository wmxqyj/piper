"""
Piper 双臂主从操作接口封装

提供主从控制和数据采集功能
"""

import time
import signal
import sys
import numpy as np
from typing import Optional, Dict, Any
from dataclasses import dataclass

from pyAgxArm import create_agx_arm_config, AgxArmFactory, ArmModel, PiperFW


@dataclass
class PiperArmState:
    """Piper 机械臂状态数据结构"""
    timestamp: float
    joint_positions: np.ndarray  # 6个关节角度 (rad)
    joint_velocities: np.ndarray  # 6个关节速度 (rad/s)
    joint_forces: np.ndarray  # 6个关节力矩 (N·m)
    gripper_pose: np.ndarray  # 末端位姿 [x, y, z, roll, pitch, yaw] (m, rad)
    gripper_positions: np.ndarray  # 2个手指位置
    gripper_open: float  # 夹爪开合状态 (0.0 或 1.0)


class PiperInterface:
    """Piper 双臂主从接口"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化 Piper 双臂接口

        Parameters
        ----------
        config : Dict[str, Any]
            配置字典，包含以下键：
            - leader_can: 主臂 CAN 通道
            - follower_can: 从臂 CAN 通道
            - arm_model: 机械臂型号
            - firmware_version: 固件版本
            - control_frequency: 控制频率 Hz
            - joint_nums: 关节数量
            - effector: 夹爪配置
        """
        self.config = config
        self.leader_robot = None
        self.follower_robot = None
        self.leader_effector = None
        self.follower_effector = None
        self.joint_nums = config.get('joint_nums', 6)
        self.control_frequency = config.get('control_frequency', 50)
        self.is_initialized = False

        # 注册信号处理
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, sig, frame):
        """处理 Ctrl+C 信号"""
        print("\n\n正在安全退出...")
        self.cleanup()
        sys.exit(0)

    def connect(self) -> bool:
        """
        连接主臂和从臂

        Returns
        -------
        bool
            连接是否成功
        """
        try:
            # 创建配置
            leader_cfg = self._create_arm_config(self.config['leader_can'])
            follower_cfg = self._create_arm_config(self.config['follower_can'])

            # 创建实例并连接
            print(f"连接主臂 ({self.config['leader_can']})...")
            self.leader_robot = AgxArmFactory.create_arm(leader_cfg)
            self.leader_robot.connect()

            print(f"连接从臂 ({self.config['follower_can']})...")
            self.follower_robot = AgxArmFactory.create_arm(follower_cfg)
            self.follower_robot.connect()

            print("两臂已连接")

            # 初始化夹爪
            if self.config.get('effector', {}).get('enabled', True):
                self._init_effectors()

            # 使能主臂
            print("使能主臂...")
            leader_retry = 0
            while not self.leader_robot.enable():
                time.sleep(0.01)
                leader_retry += 1
                if leader_retry % 500 == 0:
                    # 诊断：打印各关节使能状态
                    try:
                        status_list = self.leader_robot.get_joints_enable_status_list()
                        print(f"  主臂使能重试中... ({leader_retry * 0.01:.1f}s)")
                        print(f"  关节使能状态: {status_list}")
                    except Exception as e:
                        print(f"  主臂使能重试中... ({leader_retry * 0.01:.1f}s)")
                        print(f"  CAN通信异常: {e}")
                        print(f"  请检查: 1)CAN接口是否UP 2)通道名是否正确 3)机械臂是否上电")

            # 使能从臂
            print("使能从臂...")
            follower_retry = 0
            while not self.follower_robot.enable():
                time.sleep(0.01)
                follower_retry += 1
                if follower_retry % 500 == 0:
                    try:
                        status_list = self.follower_robot.get_joints_enable_status_list()
                        print(f"  从臂使能重试中... ({follower_retry * 0.01:.1f}s)")
                        print(f"  关节使能状态: {status_list}")
                    except Exception as e:
                        print(f"  从臂使能重试中... ({follower_retry * 0.01:.1f}s)")
                        print(f"  CAN通信异常: {e}")
                        print(f"  请检查: 1)CAN接口是否UP 2)通道名是否正确 3)机械臂是否上电")

            print("两臂已使能")

            # 设置主臂为零力拖动模式
            print("设置主臂为零力拖动模式...")
            self.leader_robot.set_leader_mode()
            time.sleep(0.5)

            self.is_initialized = True
            print("=" * 60)
            print("主从系统已启动！")
            print("主臂: 零力拖动，手动拖拽")
            print("从臂: 实时跟随主臂")
            print("=" * 60)

            return True

        except Exception as e:
            print(f"连接失败: {e}")
            return False

    def _create_arm_config(self, can_channel: str) -> dict:
        """创建机械臂配置"""
        return create_agx_arm_config(
            robot=self.config.get('arm_model', 'piper'),
            firmeware_version=self.config.get('firmware_version', 'default'),
            channel=can_channel,
        )

    def _init_effectors(self):
        """初始化末端执行器（夹爪）"""
        try:
            effector_type = self.config.get('effector', {}).get('type', 'agx_gripper')

            if effector_type == 'agx_gripper':
                effector_option = self.leader_robot.OPTIONS.EFFECTOR.AGX_GRIPPER
                print("初始化主臂夹爪...")
                self.leader_effector = self.leader_robot.init_effector(effector_option)
                print("初始化从臂夹爪...")
                self.follower_effector = self.follower_robot.init_effector(effector_option)
                print("夹爪已初始化")
            elif effector_type == 'revo2':
                effector_option = self.leader_robot.OPTIONS.EFFECTOR.REVO2
                print("初始化主臂夹爪 (Revo2)...")
                self.leader_effector = self.leader_robot.init_effector(effector_option)
                print("初始化从臂夹爪 (Revo2)...")
                self.follower_effector = self.follower_robot.init_effector(effector_option)
                print("夹爪已初始化")
            else:
                print("夹爪跟随已禁用")

        except Exception as e:
            print(f"夹爪初始化失败: {e}")
            self.leader_effector = None
            self.follower_effector = None

    def cleanup(self):
        """清理资源"""
        try:
            if self.leader_robot is not None and self.leader_robot.is_ok():
                print("退出主臂零力拖动模式...")
                self.leader_robot.set_follower_mode()
                time.sleep(0.3)
                self.leader_robot.disable()

            if self.follower_robot is not None and self.follower_robot.is_ok():
                print("失能从臂...")
                self.follower_robot.disable()

            if self.leader_robot is not None:
                self.leader_robot.disconnect()
            if self.follower_robot is not None:
                self.follower_robot.disconnect()

        except Exception as e:
            print(f"清理异常: {e}")

    def run_leader_follower_loop(self, callback_func=None):
        """
        运行主从控制循环

        Parameters
        ----------
        callback_func : callable, optional
            每个循环周期的回调函数，用于数据采集等
        """
        if not self.is_initialized:
            print("系统未初始化，无法运行主从循环")
            return

        period = 1.0 / self.control_frequency
        last_angles = [0.0] * self.joint_nums
        last_gripper_value = 0.0
        last_gripper_force = 1.0
        no_data_count = 0
        MAX_NO_DATA = self.control_frequency * 2  # 2秒无数据认为异常

        print("开始主从控制循环...")
        print("按 Ctrl+C 安全退出")

        while True:
            loop_start = time.monotonic()

            # 1. 读取主臂关节角度并转发给从臂
            joint_msg = self.leader_robot.get_leader_joint_angles()
            if joint_msg is not None and joint_msg.msg is not None:
                last_angles = joint_msg.msg
                no_data_count = 0
                self.follower_robot.move_js(last_angles)
            else:
                no_data_count += 1
                if no_data_count > MAX_NO_DATA:
                    print(f"[WARN] 已 {MAX_NO_DATA/self.control_frequency:.0f} 秒未收到主臂关节数据，停止跟随")
                    break

            # 2. 读取主臂夹爪状态并转发
            if self.leader_effector is not None and self.follower_effector is not None:
                try:
                    gcs = self.leader_effector.get_gripper_ctrl_states()
                    if gcs is not None and gcs.msg is not None:
                        cur_val = gcs.msg.value
                        cur_frc = gcs.msg.force
                        if (abs(cur_val - last_gripper_value) > 0.0001 or
                                abs(cur_frc - last_gripper_force) > 0.01):
                            last_gripper_value = cur_val
                            last_gripper_force = cur_frc
                            self.follower_effector.move_gripper_m(value=cur_val, force=cur_frc)
                except Exception:
                    pass

            # 3. 获取从臂状态数据
            if callback_func is not None:
                follower_state = self.get_follower_state()
                if follower_state is not None:
                    callback_func(follower_state)

            # 保持稳定的循环频率
            elapsed = time.monotonic() - loop_start
            time.sleep(max(0, period - elapsed))

    def get_follower_state(self) -> Optional[PiperArmState]:
        """
        获取从臂的完整状态

        Returns
        -------
        PiperArmState | None
            从臂状态，如果数据不可用则返回 None
        """
        try:
            timestamp = time.time()

            # 1. 获取关节角度
            joint_angles_msg = self.follower_robot.get_joint_angles()
            if joint_angles_msg is None or joint_angles_msg.msg is None:
                return None
            joint_positions = np.array(joint_angles_msg.msg)

            # 2. 获取关节速度和力矩（通过电机状态）
            joint_velocities = np.zeros(self.joint_nums)
            joint_forces = np.zeros(self.joint_nums)

            for i in range(1, self.joint_nums + 1):
                motor_state = self.follower_robot.get_motor_states(i)
                if motor_state is not None and motor_state.msg is not None:
                    joint_velocities[i-1] = motor_state.msg.velocity
                    joint_forces[i-1] = motor_state.msg.torque

            # 3. 获取末端位姿
            end_pose_msg = self.follower_robot.get_flange_pose()
            if end_pose_msg is None or end_pose_msg.msg is None:
                return None
            gripper_pose = np.array(end_pose_msg.msg)  # [x, y, z, roll, pitch, yaw]

            # 4. 获取夹爪状态
            gripper_positions = np.zeros(2)
            gripper_open = 0.0

            if self.follower_effector is not None:
                try:
                    gripper_state = self.follower_effector.get_gripper_ctrl_states()
                    if gripper_state is not None and gripper_state.msg is not None:
                        # 夹爪位置（假设对称）
                        gripper_positions[0] = gripper_state.msg.value
                        gripper_positions[1] = gripper_state.msg.value

                        # 判断夹爪是否打开（根据阈值）
                        # 阈值需要根据实际夹爪调整
                        gripper_open = 1.0 if gripper_state.msg.value > 0.02 else 0.0
                except Exception:
                    pass

            return PiperArmState(
                timestamp=timestamp,
                joint_positions=joint_positions,
                joint_velocities=joint_velocities,
                joint_forces=joint_forces,
                gripper_pose=gripper_pose,
                gripper_positions=gripper_positions,
                gripper_open=gripper_open,
            )

        except Exception as e:
            print(f"获取从臂状态失败: {e}")
            return None


# 测试代码
if __name__ == '__main__':
    # 示例配置
    config = {
        'leader_can': 'can_piper_l',
        'follower_can': 'can_piper_r',
        'arm_model': 'piper',
        'firmware_version': 'default',
        'control_frequency': 50,
        'joint_nums': 6,
        'effector': {
            'enabled': True,
            'type': 'agx_gripper'
        }
    }

    # 创建接口
    interface = PiperInterface(config)

    # 连接
    if interface.connect():
        # 定义回调函数
        def data_callback(state: PiperArmState):
            print(f"时间: {state.timestamp:.3f}")
            print(f"关节角度: {state.joint_positions}")
            print(f"关节速度: {state.joint_velocities}")
            print(f"关节力矩: {state.joint_forces}")
            print(f"末端位姿: {state.gripper_pose}")
            print(f"夹爪状态: {state.gripper_open}")

        # 运行主从循环
        interface.run_leader_follower_loop(callback_func=data_callback)
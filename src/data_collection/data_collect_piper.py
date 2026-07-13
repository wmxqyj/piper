"""
Piper 双臂主从数据采集主脚本

集成所有模块，实现完整的数据采集流程
"""

import os
import sys
import subprocess
import yaml
import time
import shutil
import numpy as np
from pathlib import Path

# 屏蔽 pyorbbecsdk C++ 底层 warning 刷屏
try:
    from pyorbbecsdk import Context, OBLogLevel
    Context.set_logger_to_console(OBLogLevel.ERROR)
except ImportError:
    pass

from piper_interface import PiperInterface, PiperArmState
from camera_interface import RealSenseInterface, OrbbecInterface, CameraData, create_camera_interface
from data_sync import DataSyncManager, SyncedFrame
from rlbench_adapter import RLBenchAdapter


class PiperDataCollector:
    """Piper 数据采集器"""

    def __init__(self, config_path: str):
        """
        初始化数据采集器

        Parameters
        ----------
        config_path : str
            配置文件路径
        """
        # 加载配置
        self.config = self._load_config(config_path)

        # 初始化组件
        self.piper_interface = None
        self.camera_interface = None
        self.sync_manager = None
        self.rlbench_adapter = None

        # 录制状态
        self.is_recording = False
        self.recording_thread = None

        # 录制频率控制
        rec_cfg = self.config.get('recording', {})
        self._recording_interval = 1.0 / rec_cfg.get('frequency', 10)
        self._last_recording_time = 0.0

        # 数据缓冲
        self.recorded_frames = []

        print("=" * 60)
        print("Piper 双臂主从数据采集系统")
        print("=" * 60)

    def _load_config(self, config_path: str) -> dict:
        """加载 YAML 配置文件"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def _load_gripper_T_cam(self) -> np.ndarray:
        """
        从手眼标定 YAML 文件加载 gripper_T_cam 矩阵

        Returns
        -------
        np.ndarray
            4x4 手眼标定矩阵
        """
        calib_cfg = self.config.get('calibration', {})
        result_path = calib_cfg.get('result_path', '')
        if not result_path:
            print("[WARN] 未配置 calibration.result_path，外参将使用相机接口默认值")
            return None
        if not os.path.isabs(result_path):
            result_path = os.path.join(os.path.dirname(__file__), result_path)
        if not os.path.exists(result_path):
            print(f"[WARN] 标定文件不存在: {result_path}，外参将使用相机接口默认值")
            return None
        with open(result_path, 'r') as f:
            data = yaml.safe_load(f)
        matrix = np.array(data['gripper_T_cam']['matrix'])
        print(f"已加载手眼标定矩阵: {result_path}")
        return matrix

    def initialize(self) -> bool:
        """
        初始化所有组件

        Returns
        -------
        bool
            初始化是否成功
        """
        try:
            # 1. 初始化 Piper 接口
            print("\n[1/4] 初始化 Piper 双臂接口...")
            self.piper_interface = PiperInterface(self.config['piper'])

            if not self.piper_interface.connect():
                print("Piper 接口初始化失败")
                return False

            # 2. 初始化相机接口
            camera_type = self.config['camera'].get('type', 'realsense')
            print(f"\n[2/4] 初始化 {camera_type} 相机...")
            self.camera_interface = create_camera_interface(self.config['camera'])

            if not self.camera_interface.connect():
                print("相机初始化失败，尝试继续...")

            # 3. 初始化数据同步管理器
            print("\n[3/4] 初始化数据同步管理器...")
            gripper_T_cam = self._load_gripper_T_cam()
            self.sync_manager = DataSyncManager({
                'max_time_diff': 0.05  # 50ms
            }, gripper_T_cam=gripper_T_cam)

            # 4. 初始化 RLBench 适配器
            print("\n[4/4] 初始化 RLBench 格式适配器...")
            self.rlbench_adapter = RLBenchAdapter({
                'joint_nums': self.config['piper']['joint_nums'],
                'camera_near': 0.5,
                'camera_far': 4.5,
            })

            print("\n系统初始化完成！")
            return True

        except Exception as e:
            print(f"初始化失败: {e}")
            return False

    def cleanup(self):
        """清理所有资源"""
        print("\n清理资源...")
        if self.piper_interface is not None:
            self.piper_interface.cleanup()
        if self.camera_interface is not None:
            self.camera_interface.cleanup()
        # 释放左臂（主臂）CAN 总线
        try:
            subprocess.run(
                ['cansend', 'can_piper_l', '470#FC0000000000000000'],
                timeout=2.0,
                capture_output=True,
            )
            print("左臂 CAN 已释放")
        except Exception as e:
            print(f"左臂 CAN 释放失败（可忽略）: {e}")
        print("资源清理完成")

    def start_recording(self):
        """开始录制"""
        if self.is_recording:
            print("已经在录制中")
            return

        print("\n" + "=" * 60)
        print("开始录制...")
        print("=" * 60)

        self.is_recording = True
        self._last_recording_time = 0.0
        self.recorded_frames.clear()
        self.sync_manager.clear_buffer()

        # 运行主从控制循环（带录制）
        self.piper_interface.run_leader_follower_loop(
            callback_func=self._recording_callback
        )

    def stop_recording(self):
        """停止录制"""
        if not self.is_recording:
            print("未在录制中")
            return

        print("\n" + "=" * 60)
        print("停止录制...")
        print("=" * 60)

        self.is_recording = False

        # 打印录制统计
        self.sync_manager.print_stats()

    def _recording_callback(self, arm_state: PiperArmState):
        """录制回调函数（在主从循环中调用）"""
        if not self.is_recording:
            return

        # 按 recording.frequency 降采样
        now = time.time()
        if now - self._last_recording_time < self._recording_interval:
            return

        try:
            # 获取相机数据
            camera_data = self.camera_interface.get_camera_data()

            if camera_data is None:
                print("[WARN] 无法获取相机数据，跳过此帧")
                return

            # 同步数据
            synced_frame = self.sync_manager.sync_data(arm_state, camera_data)

            if synced_frame is None:
                print("[WARN] 数据同步失败，跳过此帧")
                return

            # 添加到缓冲区
            self._last_recording_time = now
            self.sync_manager.add_frame_to_buffer(synced_frame)
            self.recorded_frames.append(synced_frame)

            # 显示进度
            frame_count = len(self.recorded_frames)
            if frame_count % 50 == 0:
                print(f"已录制 {frame_count} 帧")

        except Exception as e:
            print(f"录制回调异常: {e}")

    def save_episode(self, task_config: dict, episode_idx: int, variation_idx: int = 0):
        """
        保存录制的 episode

        Parameters
        ----------
        task_config : dict
            任务配置字典，包含 name 和 descriptions
        episode_idx : int
            episode 编号
        variation_idx : int
            variation 编号
        """
        if len(self.recorded_frames) == 0:
            print("没有录制的数据")
            return

        task_name = task_config['name']
        descriptions = task_config['descriptions']

        print("\n" + "=" * 60)
        print("保存 Episode...")
        print("=" * 60)
        print(f"任务名称: {task_name}")
        print(f"Episode: {episode_idx}")
        print(f"语言指令: {descriptions}")

        # 检查是否已存在
        save_path = os.path.join(
            self.config['demo']['save_path'],
            task_name,
            'all_variations',
            'episodes',
            f'episode{episode_idx}'
        )

        if os.path.exists(save_path):
            resp = input(f"Episode {episode_idx} 已存在，是否覆盖？(y/n): ")
            if resp.lower() != 'y':
                print("取消保存")
                return
            shutil.rmtree(save_path)

        # 转换为 RLBench Demo
        demo = self.rlbench_adapter.convert_frames_to_demo(
            self.recorded_frames,
            random_seed=self.config['demo']['random_seed'],
            variation_number=variation_idx
        )

        # 保存
        base_save_path = os.path.join(self.config['demo']['save_path'], task_name)
        self.rlbench_adapter.save_demo(demo, base_save_path, episode_idx)

        # 保存语言目标（从配置文件读取）
        descriptions_path = os.path.join(save_path, 'variation_descriptions.pkl')
        with open(descriptions_path, 'wb') as f:
            import pickle
            pickle.dump(descriptions, f)

        print(f"\n已保存到 {save_path}")

    def run_interactive(self):
        """运行交互式录制模式"""
        print("\n" + "=" * 60)
        print("进入交互式录制模式")
        print("=" * 60)

        # 显示可用的任务列表
        print("\n可用任务列表:")
        tasks = self.config.get('tasks', [])
        for i, task in enumerate(tasks):
            print(f"  [{i}] {task['name']}: {task['descriptions']}")

        # 让用户选择任务
        print("\n选择任务:")
        task_idx = input(f"输入任务编号 (0-{len(tasks)-1}): ")
        try:
            task_idx = int(task_idx)
            if 0 <= task_idx < len(tasks):
                selected_task = tasks[task_idx]
            else:
                print("无效的任务编号，使用第一个任务")
                selected_task = tasks[0]
        except ValueError:
            print("无效的输入，使用第一个任务")
            selected_task = tasks[0]

        # 从任务配置中获取参数
        task_name = selected_task['name']
        episode_idx = selected_task.get('episode_start', 0)
        variation_idx = self.config['demo']['variation']

        print("\n" + "=" * 60)
        print("任务配置:")
        print("=" * 60)
        print(f"任务名称: {task_name}")
        print(f"语言指令: {selected_task['descriptions']}")
        print(f"起始Episode: {episode_idx}")
        print(f"Variation: {variation_idx}")
        print("=" * 60)

        # 操作说明
        print("\n操作说明:")
        print("  - 拖动主臂进行示教")
        print("  - 按 's' 开始录制")
        print("  - 按 'q' 停止录制")
        print("  - 按 'y' 保存数据")
        print("  - 按 'n' 丢弃数据")
        print("  - 按 Ctrl+C 安全退出")
        print("=" * 60)

        # 启动主从控制循环（不带录制）
        print("\n启动主从控制（等待录制命令）...")

        # 使用独立的线程运行主从循环
        import threading

        def leader_follower_loop():
            self.piper_interface.run_leader_follower_loop(
                callback_func=self._idle_callback
            )

        # 启动主从循环
        control_thread = threading.Thread(target=leader_follower_loop, daemon=True)
        control_thread.start()

        # 等待用户命令
        try:
            while control_thread.is_alive():
                cmd = input("\n输入命令 (s=开始录制, q=停止录制, e=退出): ")

                if cmd == 's':
                    # 开始录制
                    self.is_recording = True
                    self._last_recording_time = 0.0
                    self.recorded_frames.clear()
                    self.sync_manager.clear_buffer()
                    print(f"录制已开始，拖动主臂进行示教...")
                    print(f"当前任务: {task_name} | Episode: {episode_idx}")

                elif cmd == 'q':
                    # 停止录制
                    self.is_recording = False
                    self.sync_manager.print_stats()

                    if len(self.recorded_frames) > 0:
                        resp = input("保存数据？(y/n): ")
                        if resp.lower() == 'y':
                            self.save_episode(selected_task, episode_idx, variation_idx)
                            episode_idx += 1  # 自动递增
                            print(f"\n下一个 Episode 编号: {episode_idx}")
                        else:
                            print("数据已丢弃")

                elif cmd == 'e':
                    # 退出
                    print("退出系统...")
                    break

                time.sleep(0.1)

        except KeyboardInterrupt:
            print("\n安全退出...")

        finally:
            self.is_recording = False

    def _idle_callback(self, arm_state: PiperArmState):
        """空闲回调函数（当未录制时）"""
        if self.is_recording:
            # 如果开始录制，调用录制回调
            self._recording_callback(arm_state)


def main():
    """主函数"""
    # 配置文件路径
    config_path = os.path.join(
        os.path.dirname(__file__),
        'cfgs',
        'piper_data_collect.yaml'
    )

    if not os.path.exists(config_path):
        print(f"配置文件不存在: {config_path}")
        sys.exit(1)

    # 创建数据采集器
    collector = PiperDataCollector(config_path)

    try:
        # 初始化
        if collector.initialize():
            # 运行交互式录制
            collector.run_interactive()

    except Exception as e:
        print(f"运行异常: {e}")

    finally:
        # 清理
        collector.cleanup()


if __name__ == '__main__':
    main()
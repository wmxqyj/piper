"""
Piper 双臂主从数据采集主脚本

集成所有模块，实现完整的数据采集流程
"""

import os
import sys
import yaml
import time
import shutil
from pathlib import Path

from piper_interface import PiperInterface, PiperArmState
from camera_interface import RealSenseInterface, CameraData
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
            print("\n[2/4] 初始化 RealSense 相机...")
            self.camera_interface = RealSenseInterface(self.config['camera'])

            if not self.camera_interface.connect():
                print("相机初始化失败，尝试继续...")

            # 3. 初始化数据同步管理器
            print("\n[3/4] 初始化数据同步管理器...")
            self.sync_manager = DataSyncManager({
                'max_time_diff': 0.05  # 50ms
            })

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
            self.sync_manager.add_frame_to_buffer(synced_frame)
            self.recorded_frames.append(synced_frame)

            # 显示进度
            frame_count = len(self.recorded_frames)
            if frame_count % 50 == 0:
                print(f"已录制 {frame_count} 帧")

        except Exception as e:
            print(f"录制回调异常: {e}")

    def save_episode(self, task_name: str, episode_idx: int, variation_idx: int = 0):
        """
        保存录制的 episode

        Parameters
        ----------
        task_name : str
            任务名称
        episode_idx : int
            episode 编号
        variation_idx : int
            variation 编号
        """
        if len(self.recorded_frames) == 0:
            print("没有录制的数据")
            return

        print("\n" + "=" * 60)
        print("保存 Episode...")
        print("=" * 60)

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

        # 保存语言目标
        lang_goal = input("请输入语言目标描述: ")
        descriptions = lang_goal.split(",")
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
        print("操作说明:")
        print("  - 拖动主臂进行示教")
        print("  - 按 's' 开始录制")
        print("  - 按 'q' 停止录制")
        print("  - 按 'y' 保存数据")
        print("  - 按 'n' 丢弃数据")
        print("  - 按 Ctrl+C 安全退出")
        print("=" * 60)

        task_name = self.config['demo']['task']
        episode_idx = self.config['demo']['episode']
        variation_idx = self.config['demo']['variation']

        print(f"\n当前任务: {task_name}")
        print(f"Episode: {episode_idx}")
        print(f"Variation: {variation_idx}")

        # 提示输入新的参数（可选）
        resp = input("是否修改任务名称/episode编号？(y/n): ")
        if resp.lower() == 'y':
            task_name = input(f"任务名称 (默认: {task_name}): ") or task_name
            episode_input = input(f"Episode 编号 (默认: {episode_idx}): ")
            if episode_input:
                episode_idx = int(episode_input)
            variation_input = input(f"Variation 编号 (默认: {variation_idx}): ")
            if variation_input:
                variation_idx = int(variation_input)

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
                    self.recorded_frames.clear()
                    self.sync_manager.clear_buffer()
                    print("录制已开始，拖动主臂进行示教...")

                elif cmd == 'q':
                    # 停止录制
                    self.is_recording = False
                    self.sync_manager.print_stats()

                    if len(self.recorded_frames) > 0:
                        resp = input("保存数据？(y/n): ")
                        if resp.lower() == 'y':
                            self.save_episode(task_name, episode_idx, variation_idx)
                            episode_idx += 1  # 自动递增
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
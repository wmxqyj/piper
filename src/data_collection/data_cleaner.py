"""
数据清洗工具

用于对录制的 Piper 数据进行清洗和关键帧标记
"""

import os
import sys
import pickle
import shutil
import numpy as np
from typing import List, Optional, Tuple

from rlbench_adapter import Demo, Observation


class DataCleaner:
    """数据清洗工具"""

    def __init__(self, episode_path: str):
        """
        初始化数据清洗工具

        Parameters
        ----------
        episode_path : str
            Episode 数据路径
        """
        self.episode_path = episode_path
        self.demo = None
        self.keypoint_idxs = []

        # 加载 Demo
        self._load_demo()

    def _load_demo(self) -> bool:
        """加载 Demo 数据"""
        try:
            low_dim_obs_path = os.path.join(self.episode_path, 'low_dim_obs.pkl')
            if not os.path.exists(low_dim_obs_path):
                print(f"文件不存在: {low_dim_obs_path}")
                return False

            with open(low_dim_obs_path, 'rb') as f:
                self.demo = pickle.load(f)

            # 获取关键帧索引
            if self.demo[0].misc is not None and 'keypoint_idxs' in self.demo[0].misc:
                self.keypoint_idxs = list(self.demo[0].misc['keypoint_idxs'])
            else:
                self.keypoint_idxs = []

            print(f"已加载 Demo，共 {len(self.demo)} 帧")
            if len(self.keypoint_idxs) > 0:
                print(f"当前关键帧: {self.keypoint_idxs}")

            return True

        except Exception as e:
            print(f"加载失败: {e}")
            return False

    def print_info(self):
        """打印 Demo 信息"""
        if self.demo is None:
            print("Demo 未加载")
            return

        print("\n" + "=" * 60)
        print("Episode 信息")
        print("=" * 60)
        print(f"总帧数: {len(self.demo)}")
        print(f"Episode路径: {self.episode_path}")

        if len(self.keypoint_idxs) > 0:
            print(f"关键帧索引: {self.keypoint_idxs}")

        # 打印第一帧和最后一帧的信息
        if len(self.demo) > 0:
            first_obs = self.demo[0]
            last_obs = self.demo[-1]

            print("\n第一帧:")
            print(f"  关节角度: {first_obs.joint_positions}")
            print(f"  末端位姿: {first_obs.gripper_pose[:3]}")  # 位置
            print(f"  夹爪状态: {first_obs.gripper_open}")

            print("\n最后一帧:")
            print(f"  关节角度: {last_obs.joint_positions}")
            print(f"  末端位姿: {last_obs.gripper_pose[:3]}")  # 位置
            print(f"  夹爪状态: {last_obs.gripper_open}")

    def trim_frames(self, trim_start: int = 0, trim_end: int = 0) -> bool:
        """
        删除前 N 帧和后 N 帧

        Parameters
        ----------
        trim_start : int
            删除前 N 帧
        trim_end : int
            删除后 N 帧

        Returns
        -------
        bool
            是否成功
        """
        if self.demo is None:
            print("Demo 未加载")
            return False

        if trim_start + trim_end >= len(self.demo):
            print("删除帧数过多，无法执行")
            return False

        print(f"\n删除前 {trim_start} 帧，删除后 {trim_end} 帧")

        # 删除前N帧
        if trim_start > 0:
            self.demo._observations = self.demo._observations[trim_start:]
            print(f"删除了前 {trim_start} 帧")

        # 删除后N帧
        if trim_end > 0:
            self.demo._observations = self.demo._observations[:-trim_end]
            print(f"删除了后 {trim_end} 帧")

        print(f"剩余 {len(self.demo)} 帧")

        # 更新关键帧索引
        if len(self.keypoint_idxs) > 0:
            self.keypoint_idxs = [idx - trim_start for idx in self.keypoint_idxs if idx >= trim_start]
            self.keypoint_idxs = [idx for idx in self.keypoint_idxs if idx < len(self.demo)]
            print(f"更新后的关键帧索引: {self.keypoint_idxs}")

        return True

    def mark_keypoint_interactive(self):
        """交互式标记关键帧"""
        if self.demo is None:
            print("Demo 未加载")
            return

        print("\n" + "=" * 60)
        print("交互式关键帧标记")
        print("=" * 60)
        print("操作说明:")
        print("  - 输入帧编号来查看该帧")
        print("  - 输入 'm <frame_id>' 来标记关键帧")
        print("  - 输入 'u <frame_id>' 来取消关键帧标记")
        print("  - 输入 'l' 来列出所有关键帧")
        print("  - 输入 'p' 打印帧范围")
        print("  - 输入 'q' 完成标记")
        print("=" * 60)

        while True:
            cmd = input("\n输入命令: ").strip()

            if cmd == 'q':
                print("完成关键帧标记")
                break

            elif cmd == 'l':
                print(f"当前关键帧: {self.keypoint_idxs}")

            elif cmd == 'p':
                print(f"帧范围: 0 到 {len(self.demo)-1}")

            elif cmd.startswith('m '):
                # 标记关键帧
                try:
                    frame_id = int(cmd.split()[1])
                    if 0 <= frame_id < len(self.demo):
                        if frame_id not in self.keypoint_idxs:
                            self.keypoint_idxs.append(frame_id)
                            self.keypoint_idxs.sort()
                            print(f"已标记帧 {frame_id} 为关键帧")
                            print(f"当前所有关键帧: {self.keypoint_idxs}")
                        else:
                            print(f"帧 {frame_id} 已经是关键帧")
                    else:
                        print(f"无效的帧编号，范围: 0 到 {len(self.demo)-1}")
                except ValueError:
                    print("无效的命令格式，请使用 'm <frame_id>'")

            elif cmd.startswith('u '):
                # 取消关键帧标记
                try:
                    frame_id = int(cmd.split()[1])
                    if frame_id in self.keypoint_idxs:
                        self.keypoint_idxs.remove(frame_id)
                        print(f"已取消帧 {frame_id} 的关键帧标记")
                    else:
                        print(f"帧 {frame_id} 不是关键帧")
                except ValueError:
                    print("无效的命令格式，请使用 'u <frame_id>'")

            elif cmd.isdigit():
                # 查看帧
                frame_id = int(cmd)
                if 0 <= frame_id < len(self.demo):
                    self._show_frame_info(frame_id)
                else:
                    print(f"无效的帧编号，范围: 0 到 {len(self.demo)-1}")

            else:
                print("未知命令")

    def _show_frame_info(self, frame_id: int):
        """显示指定帧的信息"""
        obs = self.demo[frame_id]

        print(f"\n帧 {frame_id} 信息:")
        print(f"  关节角度: {obs.joint_positions}")
        print(f"  关节速度: {obs.joint_velocities}")
        print(f"  关节力矩: {obs.joint_forces}")
        print(f"  末端位置: {obs.gripper_pose[:3]}")
        print(f"  末端四元数: {obs.gripper_pose[3:]}")
        print(f"  夹爪状态: {obs.gripper_open}")

        # 标记关键帧状态
        if frame_id in self.keypoint_idxs:
            print("  ** 关键帧 **")

    def save_cleaned_demo(self, output_path: Optional[str] = None) -> bool:
        """
        保存清洗后的 Demo

        Parameters
        ----------
        output_path : str, optional
            输出路径，如果不提供则覆盖原文件

        Returns
        -------
        bool
            是否成功
        """
        if self.demo is None:
            print("Demo 未加载")
            return False

        save_path = output_path if output_path else self.episode_path

        print(f"\n保存清洗后的 Demo 到: {save_path}")

        # 更新所有观测的 misc 字典中的 keypoint_idxs
        keypoint_idxs_array = np.array(self.keypoint_idxs)
        for obs in self.demo._observations:
            if obs.misc is not None:
                obs.misc['keypoint_idxs'] = keypoint_idxs_array

        # 保存低维数据
        low_dim_obs_path = os.path.join(save_path, 'low_dim_obs.pkl')
        os.makedirs(save_path, exist_ok=True)

        with open(low_dim_obs_path, 'wb') as f:
            pickle.dump(self.demo, f)

        # 删除多余的图像文件
        self._cleanup_images(save_path)

        print(f"已保存，共 {len(self.demo)} 帧")
        if len(self.keypoint_idxs) > 0:
            print(f"关键帧索引: {self.keypoint_idxs}")

        return True

    def _cleanup_images(self, save_path: str):
        """删除多余的图像文件"""
        # 删除超出范围的 RGB 图像
        rgb_path = os.path.join(save_path, 'front_rgb')
        if os.path.exists(rgb_path):
            for filename in os.listdir(rgb_path):
                if filename.endswith('.png'):
                    frame_id = int(filename.split('.')[0])
                    if frame_id >= len(self.demo):
                        file_path = os.path.join(rgb_path, filename)
                        os.remove(file_path)
                        print(f"删除 RGB图像: {filename}")

        # 删除超出范围的深度图像
        depth_path = os.path.join(save_path, 'front_depth')
        if os.path.exists(depth_path):
            for filename in os.listdir(depth_path):
                if filename.endswith('.png'):
                    frame_id = int(filename.split('.')[0])
                    if frame_id >= len(self.demo):
                        file_path = os.path.join(depth_path, filename)
                        os.remove(file_path)
                        print(f"删除深度图像: {filename}")

    def auto_detect_keypoints(self, threshold: float = 0.1) -> List[int]:
        """
        自动检测关键帧（基于关节角度变化和夹爪状态）

        Parameters
        ----------
        threshold : float
            关节角度变化的阈值（rad）

        Returns
        -------
        List[int]
            检测到的关键帧索引
        """
        if self.demo is None or len(self.demo) < 2:
            return []

        keypoints = []

        # 检测夹爪状态变化
        for i in range(1, len(self.demo)):
            if self.demo[i].gripper_open != self.demo[i-1].gripper_open:
                keypoints.append(i)

        # 检测关节角度大幅变化
        for i in range(1, len(self.demo)):
            joint_diff = np.abs(self.demo[i].joint_positions - self.demo[i-1].joint_positions)
            if np.max(joint_diff) > threshold:
                keypoints.append(i)

        # 去重并排序
        keypoints = sorted(set(keypoints))

        print(f"自动检测到 {len(keypoints)} 个关键帧: {keypoints}")

        return keypoints


def _scan_episodes(data_root: str):
    """扫描 data 目录下所有可用的 episode 路径"""
    episodes = []
    if not os.path.exists(data_root):
        return episodes
    for task in sorted(os.listdir(data_root)):
        task_dir = os.path.join(data_root, task)
        if not os.path.isdir(task_dir):
            continue
        ep_dir = os.path.join(task_dir, 'all_variations', 'episodes')
        if not os.path.exists(ep_dir):
            continue
        for ep in sorted(os.listdir(ep_dir)):
            ep_path = os.path.join(ep_dir, ep)
            if os.path.isdir(ep_path):
                episodes.append((task, ep, ep_path))
    return episodes


def main():
    """主函数"""
    print("=" * 60)
    print("Piper 数据清洗工具")
    print("=" * 60)

    # 默认数据路径（与脚本同级的 data 目录）
    default_data = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data')
    data_root = input(f"数据目录 (默认 {default_data}): ").strip() or default_data

    # 扫描可用 episode
    episodes = _scan_episodes(data_root)
    if not episodes:
        print(f"在 {data_root} 下未找到任何 episode 数据")
        sys.exit(1)

    print(f"\n找到 {len(episodes)} 个 episode:\n")
    for i, (task, ep, path) in enumerate(episodes):
        print(f"  [{i}] {task} / {ep}  -> {path}")

    print()
    while True:
        try:
            choice = int(input(f"选择 episode (0-{len(episodes)-1}): ").strip())
            if 0 <= choice < len(episodes):
                break
            print(f"请输入 0-{len(episodes)-1}")
        except ValueError:
            print("请输入数字")

    task_name, ep_name, episode_path = episodes[choice]
    print(f"\n已选择: {task_name} / {ep_name}")

    # 创建清洗工具
    cleaner = DataCleaner(episode_path)
    if cleaner.demo is None:
        print("加载失败，退出")
        sys.exit(1)

    # 显示信息
    cleaner.print_info()

    # 交互式清洗
    print("\n" + "=" * 60)
    print("清洗操作菜单")
    print("=" * 60)
    print("1. 删除帧")
    print("2. 交互式标记关键帧")
    print("3. 自动检测关键帧")
    print("4. 保存清洗后的数据")
    print("5. 显示信息")
    print("0. 退出")
    print("=" * 60)

    while True:
        choice = input("\n选择操作 (0-5): ").strip()

        if choice == '0':
            print("退出")
            break

        elif choice == '1':
            trim_start = int(input("删除前N帧 (默认0): ") or '0')
            trim_end = int(input("删除后N帧 (默认0): ") or '0')
            cleaner.trim_frames(trim_start, trim_end)

        elif choice == '2':
            cleaner.mark_keypoint_interactive()

        elif choice == '3':
            threshold = float(input("关节角度变化阈值 (默认0.1 rad): ") or '0.1')
            auto_keypoints = cleaner.auto_detect_keypoints(threshold)

            resp = input("是否使用自动检测的关键帧？(y/n): ")
            if resp.lower() == 'y':
                cleaner.keypoint_idxs = auto_keypoints

        elif choice == '4':
            output_path = input("输出路径 (默认覆盖原文件): ").strip()
            cleaner.save_cleaned_demo(output_path if output_path else None)

        elif choice == '5':
            cleaner.print_info()

        else:
            print("无效选择")


if __name__ == '__main__':
    main()
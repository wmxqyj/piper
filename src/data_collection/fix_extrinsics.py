"""
修复已有数据中的相机外参

将 low_dim_obs.pkl 中每帧的 front_camera_extrinsics 从 np.eye(4)
替换为 base_T_gripper @ gripper_T_cam（基于手眼标定结果）。

用法:
  # dry-run: 仅预览，不写入
  python fix_extrinsics.py --dry-run

  # 实际修复
  python fix_extrinsics.py
"""

import os
import sys
import pickle
import argparse
import numpy as np
import yaml

CALIBRATION_YAML = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'visual_servo', 'calibration_data',
    'calib_data_20260707_181218', 'calibration_result.yaml',
)

DATA_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', '..', 'data',
)


def load_gripper_T_cam(yaml_path: str) -> np.ndarray:
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    matrix = data['gripper_T_cam']['matrix']
    return np.array(matrix)


def scan_episodes(data_root: str):
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
            pkl_path = os.path.join(ep_path, 'low_dim_obs.pkl')
            if os.path.isfile(pkl_path):
                episodes.append((task, ep, pkl_path))
    return episodes


def fix_episode(pkl_path: str, gripper_T_cam: np.ndarray, dry_run: bool = False):
    with open(pkl_path, 'rb') as f:
        demo = pickle.load(f)

    num_frames = len(demo)
    changed = 0
    sample_printed = False

    for i, obs in enumerate(demo):
        if obs.misc is None:
            continue

        old_ext = obs.misc.get('front_camera_extrinsics')
        base_T_gripper = obs.gripper_matrix

        if base_T_gripper is None:
            print(f"    [WARN] 帧 {i}: gripper_matrix 为 None，跳过")
            continue

        new_ext = base_T_gripper @ gripper_T_cam

        if not sample_printed:
            print(f"    帧 0 旧外参 (前3行):")
            print(f"      {old_ext[:3] if old_ext is not None else 'None'}")
            print(f"    帧 0 新外参 (前3行):")
            print(f"      {new_ext[:3]}")
            print(f"    gripper_matrix (前3行):")
            print(f"      {base_T_gripper[:3]}")
            print(f"    gripper_T_cam (前3行):")
            print(f"      {gripper_T_cam[:3]}")
            sample_printed = True

        obs.misc['front_camera_extrinsics'] = new_ext
        changed += 1

    if dry_run:
        print(f"    [DRY-RUN] 将修改 {changed}/{num_frames} 帧的外参（未写入）")
    else:
        with open(pkl_path, 'wb') as f:
            pickle.dump(demo, f)
        print(f"    已修复 {changed}/{num_frames} 帧的外参")

    return changed


def main():
    parser = argparse.ArgumentParser(description='修复已有数据中的相机外参')
    parser.add_argument('--dry-run', action='store_true', help='仅预览，不写入文件')
    parser.add_argument('--data-root', type=str, default=DATA_ROOT, help='数据根目录')
    parser.add_argument('--calibration', type=str, default=CALIBRATION_YAML, help='手眼标定 YAML 路径')
    args = parser.parse_args()

    print("=" * 60)
    print("修复相机外参")
    print("=" * 60)
    print(f"数据目录: {args.data_root}")
    print(f"标定文件: {args.calibration}")
    print(f"模式: {'DRY-RUN (预览)' if args.dry_run else '实际修改'}")

    gripper_T_cam = load_gripper_T_cam(args.calibration)
    print(f"\ngripper_T_cam (4x4):")
    for row in gripper_T_cam:
        print(f"  [{', '.join(f'{v:10.6f}' for v in row)}]")

    episodes = scan_episodes(args.data_root)
    if not episodes:
        print(f"\n未找到任何 episode 数据")
        sys.exit(1)

    print(f"\n找到 {len(episodes)} 个 episode:")
    total_changed = 0
    for task, ep, pkl_path in episodes:
        print(f"\n  {task}/{ep}")
        changed = fix_episode(pkl_path, gripper_T_cam, dry_run=args.dry_run)
        total_changed += changed

    print(f"\n{'[DRY-RUN] ' if args.dry_run else ''}总计: {len(episodes)} 个 episode, {total_changed} 帧外参{'将' if args.dry_run else '已'}被修复")


if __name__ == '__main__':
    main()
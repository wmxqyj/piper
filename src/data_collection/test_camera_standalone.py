"""
Orbbec DC1 相机独立测试脚本

与数据采集器使用相同的 camera_interface 和配置，用于隔离测试相机部分。
覆盖采集频率 50Hz（实际 arm 环频率）和 10Hz（录制频率）两种场景。
"""
import os
import sys
import yaml
import time
import cv2
import numpy as np
from pathlib import Path

# 确保可以导入同目录下的 camera_interface
sys.path.insert(0, os.path.dirname(__file__))
from camera_interface import create_camera_interface


def load_camera_config(config_path: str) -> dict:
    """从数据采集配置文件中加载 camera 节"""
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg['camera']


def run_test(config: dict, call_hz: float, duration: float, show_preview: bool = False):
    """
    以指定频率调用 get_camera_data，统计结果

    Parameters
    ----------
    config : dict           camera 配置节
    call_hz : float         调用频率 (Hz)
    duration : float        测试持续时间 (秒)
    show_preview : bool     是否显示彩色预览窗口
    """
    cam = create_camera_interface(config)
    print(f"\n{'='*60}")
    print(f"测试参数: {call_hz}Hz 调用频率, {duration}s 持续时间")
    print(f"{'='*60}")

    # 连接相机
    t0 = time.time()
    connected = cam.connect()
    connect_cost = time.time() - t0
    if not connected:
        print("相机连接失败，终止测试")
        cam.cleanup()
        return

    # 等 capture 线程启动稳定
    time.sleep(0.5)

    # 采集循环
    interval = 1.0 / call_hz
    total_calls = 0
    success_calls = 0
    none_calls = 0
    frame_timestamps = []
    first_frame_time = None

    print(f"\n开始采集 ({call_hz}Hz, {duration}s)...")
    t_start = time.time()

    while time.time() - t_start < duration:
        loop_start = time.time()
        total_calls += 1

        data = cam.get_camera_data()
        if data is None:
            none_calls += 1
        else:
            success_calls += 1
            frame_timestamps.append(data.timestamp)
            if first_frame_time is None:
                first_frame_time = data.timestamp

            if show_preview:
                # 显示彩色图 (convert BGR -> RGB for display)
                cv2.imshow("Orbbec DC1 - Color", data.rgb_image)
                # 深度图归一化显示
                depth_norm = cv2.normalize(data.depth_image, None, 0, 255,
                                           cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                depth_colormap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
                cv2.imshow("Orbbec DC1 - Depth", depth_colormap)
                key = cv2.waitKey(1)
                if key == ord('q') or key == 27:
                    break

        # 精确控制调用间隔
        elapsed = time.time() - loop_start
        sleep_time = interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    t_elapsed = time.time() - t_start
    cam.cleanup()

    # ---- 统计报告 ----
    actual_hz = total_calls / t_elapsed
    print(f"\n{'='*60}")
    print(f"测试结果")
    print(f"{'='*60}")
    print(f"实际运行时长:     {t_elapsed:.2f}s")
    print(f"实际调用频率:     {actual_hz:.1f}Hz")
    print(f"总调用次数:       {total_calls}")
    print(f"成功获取帧:       {success_calls}  ({success_calls/total_calls*100:.1f}%)")
    print(f"返回 None:        {none_calls}  ({none_calls/total_calls*100:.1f}%)")

    if success_calls > 0:
        # 取最后一帧展示细节
        data = cam.get_camera_data()
        if data is not None:
            print(f"\n帧信息 (最后一帧成功数据):")
            print(f"  RGB 形状:   {data.rgb_image.shape}")
            print(f"  RGB dtype:  {data.rgb_image.dtype}")
            print(f"  RGB 范围:   [{data.rgb_image.min()}, {data.rgb_image.max()}]")
            print(f"  深度形状:   {data.depth_image.shape}")
            print(f"  深度 dtype:  {data.depth_image.dtype}")
            print(f"  深度范围:   [{data.depth_image.min():.4f}, {data.depth_image.max():.4f}] m")
            print(f"  内参矩阵:\n{data.camera_intrinsics}")

        # 帧率统计
        if len(frame_timestamps) >= 2:
            intervals = np.diff(frame_timestamps)
            print(f"\n帧到达间隔统计:")
            print(f"  平均: {np.mean(intervals)*1000:.1f}ms")
            print(f"  中值: {np.median(intervals)*1000:.1f}ms")
            print(f"  标准差: {np.std(intervals)*1000:.1f}ms")
            print(f"  等效 FPS: {1.0/np.mean(intervals):.1f}")
            print(f"  总帧数: {len(frame_timestamps)}")

        if first_frame_time is not None:
            print(f"\n首帧到达时间:   {(first_frame_time - t_start)*1000:.0f}ms (从采集开始)")
    else:
        print("\n[FAIL] 未成功获取任何帧")

    print(f"{'='*60}\n")
    return success_calls > 0


def main():
    # 使用与数据采集器相同的配置文件
    config_path = os.path.join(
        os.path.dirname(__file__),
        'cfgs',
        'piper_data_collect.yaml'
    )
    if not os.path.exists(config_path):
        print(f"配置文件不存在: {config_path}")
        sys.exit(1)

    config = load_camera_config(config_path)
    print(f"相机配置: {config}")

    # === 测试场景 ===

    # 1. 低频测试 (10Hz — 录制目标频率) 仅 5 秒
    run_test(config, call_hz=10, duration=5, show_preview=False)

    # 2. 高频测试 (50Hz — 实际 arm 控制环频率) 仅 5 秒
    run_test(config, call_hz=50, duration=5, show_preview=False)

    # 3. 带预览测试 (30Hz, 按 q 退出, 不设时间限制)
    print("按 q 或 ESC 退出预览窗口")
    run_test(config, call_hz=30, duration=30, show_preview=True)


if __name__ == '__main__':
    main()

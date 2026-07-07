#!/usr/bin/env python3
"""
手眼标定 - 独立入口脚本

用法:
    python run_calibration.py                        # 使用默认配置
    python run_calibration.py --config path/to/config.yaml  # 使用自定义配置
"""

import os
import sys
import argparse

# 确保 src 目录在 Python 路径中
_src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from visual_servo.calibration.data_collector import HandEyeDataCollector
from visual_servo.calibration.result_manager import CalibrationResultManager


def main():
    parser = argparse.ArgumentParser(description="手眼标定数据采集与计算")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="配置文件路径 (默认: calibration/config/calibration.yaml)",
    )
    args = parser.parse_args()

    if args.config:
        config_path = args.config
    else:
        config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "config",
            "calibration.yaml",
        )

    if not os.path.exists(config_path):
        print(f"配置文件不存在: {config_path}")
        sys.exit(1)

    collector = HandEyeDataCollector(config_path)

    try:
        # 1. 连接机械臂
        if not collector.connect_robot():
            print("机械臂连接失败，退出")
            return

        # 2. 连接相机
        if not collector.connect_camera():
            print("相机连接失败，退出")
            collector.cleanup()
            return

        # 3. 运行数据采集
        collector.run_collection()

        if len(collector.collected_data) < 3:
            print(f"采集数据不足（{len(collector.collected_data)}），退出")
            collector.cleanup()
            return

        # 4. 保存原始数据
        data_dir = collector.save_data()

        # 5. 计算标定
        print("\n" + "=" * 60)
        result = collector.compute_calibration()

        if result is not None:
            # 6. 保存标定结果
            CalibrationResultManager.save(result, data_dir)
            CalibrationResultManager.print_summary(result)
        else:
            print("标定计算失败，原始数据已保留")

    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"运行异常: {e}")
        import traceback

        traceback.print_exc()
    finally:
        collector.cleanup()


if __name__ == "__main__":
    main()

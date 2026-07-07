#!/usr/bin/env python3
"""
手眼标定 - 主入口脚本

一键运行：手动拖拽机械臂 → 按 s 保存点位 → 自动计算标定矩阵

用法:
    python run_calibration.py

配置:
    calibration/config/calibration.yaml
"""

import os
import sys

# 确保 src 目录在 Python 路径中
_src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from visual_servo.calibration.data_collector import main

if __name__ == "__main__":
    main()

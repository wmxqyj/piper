#!/usr/bin/env python3
"""HSV 红色阈值调试工具

实时显示相机画面和 HSV 二值图，拖动滑块调整阈值，
找到能稳定识别红色方块的参数后，把值填入 verification.yaml。

用法:
  cd ~/qyj/program/pyAgxArm/src/visual_servo
  python hsv_tuner.py

滑块说明:
  H_Low1 / H_High1 — 红色区间1 (0~10)
  H_Low2 / H_High2 — 红色区间2 (160~179)
  S_Low / V_Low    — 降低可包容更淡的颜色
"""

import cv2
import numpy as np
import yaml
import os
import sys

_src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from data_collection.camera_interface import OrbbecInterface


def nothing(x):
    pass


def main():
    # 加载当前配置作初始值
    config_path = os.path.join(os.path.dirname(__file__), "verification", "config", "verification.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    cfg = config.get("red_block", config)

    # 连接相机
    camera_cfg = config.get("camera", {})
    camera_cfg["type"] = "orbbec"
    print("连接 Orbbec 相机...")
    camera = OrbbecInterface(camera_cfg)
    if not camera.connect():
        print("相机连接失败")
        return
    print("相机已连接，等待第一帧...")
    for _ in range(30):
        data = camera.get_camera_data()
        if data is not None:
            break
        cv2.waitKey(100)

    # 创建窗口 + 滑块
    cv2.namedWindow("Tune", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Tune", 960, 540)
    cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Mask", 480, 360)

    cv2.createTrackbar("H_Low1", "Mask", cfg["hsv_range1_low"][0], 179, nothing)
    cv2.createTrackbar("H_High1", "Mask", cfg["hsv_range1_high"][0], 179, nothing)
    cv2.createTrackbar("H_Low2", "Mask", cfg["hsv_range2_low"][0], 179, nothing)
    cv2.createTrackbar("H_High2", "Mask", cfg["hsv_range2_high"][0], 179, nothing)
    cv2.createTrackbar("S_Low", "Mask", cfg["hsv_range1_low"][1], 255, nothing)
    cv2.createTrackbar("V_Low", "Mask", cfg["hsv_range1_low"][2], 255, nothing)

    print("\n=== HSV 调参 ===")
    print("调整滑块直到红色方块在 Mask 窗口中变成白色区域")
    print("按 's' 保存当前参数到配置文件")
    print("按 'q' 退出\n")

    while True:
        data = camera.get_camera_data()
        if data is None:
            cv2.waitKey(30)
            continue

        frame = data.rgb_image.copy()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 读取滑块值
        hl1 = cv2.getTrackbarPos("H_Low1", "Mask")
        hh1 = cv2.getTrackbarPos("H_High1", "Mask")
        hl2 = cv2.getTrackbarPos("H_Low2", "Mask")
        hh2 = cv2.getTrackbarPos("H_High2", "Mask")
        s_low = cv2.getTrackbarPos("S_Low", "Mask")
        v_low = cv2.getTrackbarPos("V_Low", "Mask")

        # 生成 mask
        lower1 = np.array([hl1, s_low, v_low])
        upper1 = np.array([hh1, 255, 255])
        lower2 = np.array([hl2, s_low, v_low])
        upper2 = np.array([hh2, 255, 255])

        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)

        # 形态学去噪
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 在 frame 上叠加红色半透明 mask（用于定位）
        overlay = frame.copy()
        overlay[:, :, 2] = cv2.add(overlay[:, :, 2], mask // 2)
        display = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        # 显示
        cv2.imshow("Tune", display)
        cv2.imshow("Mask", mask)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            # 保存参数到配置文件
            new_cfg = {
                "hsv_range1_low": [hl1, s_low, v_low],
                "hsv_range1_high": [hh1, 255, 255],
                "hsv_range2_low": [hl2, s_low, v_low],
                "hsv_range2_high": [hh2, 255, 255],
            }
            import shutil
            shutil.copy2(config_path, config_path + ".bak")
            config["red_block"].update(new_cfg)
            with open(config_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            print(f"[OK] 已保存到 {config_path}")

    camera.cleanup()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

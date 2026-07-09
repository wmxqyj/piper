# ******************************************************************************
#  Copyright (c) 2023 Orbbec 3D Technology, Inc
#  
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.  
#  You may obtain a copy of the License at
#  
#      http:# www.apache.org/licenses/LICENSE-2.0
#  
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ******************************************************************************
import argparse
import sys

import cv2
import numpy as np

from pyorbbecsdk import *
from utils import frame_to_bgr_image

ESC_KEY = 27


def main(argv):
    pipeline = Pipeline()
    device = pipeline.get_device()
    device_info = device.get_device_info()
    device_pid = device_info.get_pid()
    config = Config()
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode",
                        help="align mode, HW=hardware mode,SW=software mode,NONE=disable align",
                        type=str, default='HW')
    parser.add_argument("-s", "--enable_sync", help="enable sync", type=bool, default=True)
    args = parser.parse_args()
    align_mode = args.mode
    enable_sync = args.enable_sync
    try:
        profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        color_profile = profile_list.get_default_video_stream_profile()
        config.enable_stream(color_profile)
        profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        assert profile_list is not None
        depth_profile = profile_list.get_default_video_stream_profile()
        assert depth_profile is not None
        print("color profile : {}x{}@{}_{}".format(color_profile.get_width(),
                                                   color_profile.get_height(),
                                                   color_profile.get_fps(),
                                                   color_profile.get_format()))
        print("depth profile : {}x{}@{}_{}".format(depth_profile.get_width(),
                                                   depth_profile.get_height(),
                                                   depth_profile.get_fps(),
                                                   depth_profile.get_format()))
        config.enable_stream(depth_profile)
    except Exception as e:
        print(e)
        return
    align_filter = None
    if align_mode in ('HW', 'SW'):
        try:
            ob_align_mode = OBAlignMode.HW_MODE if align_mode == 'HW' else OBAlignMode.SW_MODE
            if device_pid == 0x066B:
                ob_align_mode = OBAlignMode.SW_MODE  # Femto Mega 仅支持软件对齐
            config.set_align_mode(ob_align_mode)
            print(f"set_align_mode({align_mode}) 配置成功")
        except Exception as e:
            print(f"set_align_mode({align_mode}) 失败 ({e})，将使用 AlignFilter 软件后处理对齐")
            align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
    else:
        config.set_align_mode(OBAlignMode.DISABLE)
        print("对齐已禁用")

    if enable_sync:
        try:
            pipeline.enable_frame_sync()
        except Exception as e:
            print(f"enable_frame_sync 失败: {e}")

    try:
        pipeline.start(config)
    except Exception as e:
        print(e)
        return

    # ---- debug: 打印相机内参与对齐信息 ----
    try:
        camera_param = pipeline.get_camera_param()
        print("\n===== 相机参数 (Debug) =====")
        print(f"RGB 内参: fx={camera_param.rgb_intrinsic.fx:.4f}, fy={camera_param.rgb_intrinsic.fy:.4f}, "
              f"cx={camera_param.rgb_intrinsic.cx:.4f}, cy={camera_param.rgb_intrinsic.cy:.4f}, "
              f"{camera_param.rgb_intrinsic.width}x{camera_param.rgb_intrinsic.height}")
        print(f"Depth 内参: fx={camera_param.depth_intrinsic.fx:.4f}, fy={camera_param.depth_intrinsic.fy:.4f}, "
              f"cx={camera_param.depth_intrinsic.cx:.4f}, cy={camera_param.depth_intrinsic.cy:.4f}, "
              f"{camera_param.depth_intrinsic.width}x{camera_param.depth_intrinsic.height}")
        # D2C 变换: depth -> color 的外参
        trans = camera_param.transform
        print(f"D2C 外参 (depth->color):")
        print(f"  Rotation:\n{np.array(trans.rot).reshape(3, 3)}")
        print(f"  Translation: {np.array(trans.transform)}")
        print(f"对齐模式: {align_mode}")
        print(f"帧同步: {'已启用' if enable_sync else '未启用'}")
        print("============================\n")
    except Exception as e:
        print(f"[Debug] 无法获取相机参数: {e}")

    frame_count = 0
    while True:
        try:
            frames: FrameSet = pipeline.wait_for_frames(100)
            if frames is None:
                continue

            # ---- 如果 set_align_mode 不支持，用 AlignFilter 做软件后处理对齐 ----
            if align_filter is not None:
                frames = align_filter.process(frames)
                if frames is None:
                    continue

            color_frame = frames.get_color_frame()
            if color_frame is None:
                continue
            # covert to RGB format
            color_image = frame_to_bgr_image(color_frame)
            if color_image is None:
                print("failed to convert frame to image")
                continue
            depth_frame = frames.get_depth_frame()
            if depth_frame is None:
                continue

            width = depth_frame.get_width()
            height = depth_frame.get_height()
            scale = depth_frame.get_depth_scale()

            depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
            depth_data = depth_data.reshape((height, width))
            depth_data = depth_data.astype(np.float32) * scale

            # ---- debug: 每 30 帧打印一次校验信息 ----
            if frame_count % 30 == 0:
                color_h, color_w = color_image.shape[:2]
                print(f"[Debug] 帧 #{frame_count}: "
                      f"color={color_w}x{color_h}, "
                      f"depth={width}x{height}, "
                      f"depth_scale={scale:.6f}, "
                      f"align_mode={align_mode}")
                # 取画面中心像素，验证深度和彩色是否对齐
                cy, cx = color_h // 2, color_w // 2
                # 如果 depth 已对齐到 color，尺寸应一致，可直接索引
                if height == color_h and width == color_w:
                    d_center = depth_data[cy, cx]
                    print(f"  >> 中心像素 ({cx},{cy}): depth={d_center:.1f}mm, color=BGR({color_image[cy, cx]})")
                else:
                    # 未对齐: 分别取各自中心
                    d_center = depth_data[height // 2, width // 2]
                    print(f"  >> depth 未对齐到 color! depth中心=({width//2},{height//2}): {d_center:.1f}mm")
            frame_count += 1

            depth_image = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_image = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)
            # overlay color image on depth image
            depth_image = cv2.addWeighted(color_image, 0.5, depth_image, 0.5, 0)
            cv2.imshow("SyncAlignViewer ", depth_image)
            key = cv2.waitKey(1)
            if key == ord('q') or key == ESC_KEY:
                break
        except KeyboardInterrupt:
            break
    pipeline.stop()


if __name__ == "__main__":
    print("Please NOTE: This example is NOT supported by the Gemini 330 series.")
    print("If you want to see the example on Gemini 330 series, please refer to align_filter_viewer.py")
    main(sys.argv[1:])

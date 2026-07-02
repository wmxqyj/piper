"""
PyRealSense 相机接口封装

提供 RGB-D 相机的数据获取功能
"""

import numpy as np
import time
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

try:
    import pyrealsense2 as rs
except ImportError:
    print("请安装 pyrealsense2: pip install pyrealsense2")
    rs = None


@dataclass
class CameraData:
    """相机数据结构"""
    timestamp: float
    rgb_image: np.ndarray  # RGB图像 (H, W, 3)
    depth_image: np.ndarray  # 深度图像 (H, W)
    camera_intrinsics: np.ndarray  # 相机内参矩阵 (3, 3)
    camera_extrinsics: np.ndarray  # 相机外参矩阵 (4, 4) (需要从TF获取)


class RealSenseInterface:
    """RealSense 相机接口"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化 RealSense 相机接口

        Parameters
        ----------
        config : Dict[str, Any]
            配置字典，包含以下键：
            - resolution: {'width': int, 'height': int}
            - fps: int
            - depth_scale: float
            - align_depth: bool
        """
        self.config = config
        self.pipeline = None
        self.profile = None
        self.align = None
        self.is_initialized = False

        # 相机参数
        self.width = config.get('resolution', {}).get('width', 640)
        self.height = config.get('resolution', {}).get('height', 480)
        self.fps = config.get('fps', 30)
        self.depth_scale = config.get('depth_scale', 0.001)
        self.align_depth = config.get('align_depth', True)

        # 相机内参（将在初始化后获取）
        self.intrinsics = None

    def connect(self) -> bool:
        """
        连接相机

        Returns
        -------
        bool
            连接是否成功
        """
        if rs is None:
            print("pyrealsense2 未安装")
            return False

        try:
            # 创建 pipeline
            self.pipeline = rs.pipeline()

            # 创建配置
            config = rs.config()
            config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
            config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)

            # 启动 pipeline
            print(f"连接 RealSense 相机 ({self.width}x{self.height}, {self.fps}fps)...")
            self.profile = self.pipeline.start(config)

            # 获取深度传感器并设置深度比例
            depth_sensor = self.profile.get_device().first_depth_sensor()
            depth_scale = depth_sensor.get_depth_scale()
            print(f"深度比例: {depth_scale}")

            # 创建对齐对象（将深度对齐到彩色）
            if self.align_depth:
                self.align = rs.align(rs.stream.color)

            # 获取相机内参
            self._get_intrinsics()

            self.is_initialized = True
            print("相机已连接")
            return True

        except Exception as e:
            print(f"相机连接失败: {e}")
            return False

    def _get_intrinsics(self):
        """获取相机内参"""
        try:
            # 获取彩色流配置
            color_stream = self.profile.get_stream(rs.stream.color)
            color_profile = rs.video_frame_profile(color_stream)
            self.intrinsics = color_profile.get_intrinsics()

            print(f"相机内参:")
            print(f"  分辨率: {self.intrinsics.width} x {self.intrinsics.height}")
            print(f"  焦距: fx={self.intrinsics.fx}, fy={self.intrinsics.fy}")
            print(f"  主点: ppx={self.intrinsics.ppx}, ppy={self.intrinsics.ppy}")

        except Exception as e:
            print(f"获取相机内参失败: {e}")
            self.intrinsics = None

    def get_intrinsics_matrix(self) -> np.ndarray:
        """
        获取相机内参矩阵

        Returns
        -------
        np.ndarray
            3x3 内参矩阵 [[fx, 0, ppx], [0, fy, ppy], [0, 0, 1]]
        """
        if self.intrinsics is None:
            return np.eye(3)

        return np.array([
            [self.intrinsics.fx, 0, self.intrinsics.ppx],
            [0, self.intrinsics.fy, self.intrinsics.ppy],
            [0, 0, 1]
        ])

    def get_camera_data(self) -> Optional[CameraData]:
        """
        获取相机数据（RGB + 深度）

        Returns
        -------
        CameraData | None
            相机数据，如果获取失败则返回 None
        """
        if not self.is_initialized:
            return None

        try:
            timestamp = time.time()

            # 等待帧
            frames = self.pipeline.wait_for_frames()

            # 对齐深度到彩色
            if self.align:
                frames = self.align.process(frames)

            # 获取彩色帧
            color_frame = frames.get_color_frame()
            if not color_frame:
                return None

            # 获取深度帧
            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                return None

            # 转换为 numpy 数组
            rgb_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # 获取内参矩阵
            camera_intrinsics = self.get_intrinsics_matrix()

            # 外参矩阵需要从TF系统获取（暂时设为单位矩阵）
            camera_extrinsics = np.eye(4)

            return CameraData(
                timestamp=timestamp,
                rgb_image=rgb_image,
                depth_image=depth_image,
                camera_intrinsics=camera_intrinsics,
                camera_extrinsics=camera_extrinsics,
            )

        except Exception as e:
            print(f"获取相机数据失败: {e}")
            return None

    def cleanup(self):
        """清理资源"""
        try:
            if self.pipeline is not None:
                print("停止相机...")
                self.pipeline.stop()
                print("相机已停止")

        except Exception as e:
            print(f"相机清理异常: {e}")

    def get_depth_at_pixel(self, depth_image: np.ndarray, x: int, y: int) -> float:
        """
        获取指定像素的深度值（单位：米）

        Parameters
        ----------
        depth_image : np.ndarray
            深度图像
        x : int
            像素 x 坐标
        y : int
            像素 y 坐标

        Returns
        -------
        float
            深度值（米）
        """
        depth_value = depth_image[y, x]
        return depth_value * self.depth_scale

    def deproject_pixel_to_point(self, depth_image: np.ndarray, x: int, y: int) -> Tuple[float, float, float]:
        """
        将像素坐标转换为3D点坐标

        Parameters
        ----------
        depth_image : np.ndarray
            深度图像
        x : int
            像素 x 坐标
        y : int
            像素 y 坐标

        Returns
        -------
        Tuple[float, float, float]
            3D点坐标 (x, y, z) 单位：米
        """
        if self.intrinsics is None:
            return (0.0, 0.0, 0.0)

        depth = self.get_depth_at_pixel(depth_image, x, y)
        result = rs.rs2_deproject_pixel_to_point(self.intrinsics, [x, y], depth)

        return (result[0], result[1], result[2])


# 测试代码
if __name__ == '__main__':
    # 示例配置
    config = {
        'resolution': {'width': 640, 'height': 480},
        'fps': 30,
        'depth_scale': 0.001,
        'align_depth': True
    }

    # 创建接口
    camera = RealSenseInterface(config)

    # 连接
    if camera.connect():
        try:
            # 测试获取数据
            for i in range(10):
                data = camera.get_camera_data()
                if data is not None:
                    print(f"帧 {i}:")
                    print(f"  时间: {data.timestamp:.3f}")
                    print(f"  RGB 尺寸: {data.rgb_image.shape}")
                    print(f"  深度尺寸: {data.depth_image.shape}")
                    print(f"  内参矩阵: {data.camera_intrinsics}")

                    # 测试深度值
                    center_x = data.rgb_image.shape[1] // 2
                    center_y = data.rgb_image.shape[0] // 2
                    depth = camera.get_depth_at_pixel(data.depth_image, center_x, center_y)
                    print(f"  中心深度: {depth:.3f} m")

                    time.sleep(0.1)

        except KeyboardInterrupt:
            print("停止测试")

        finally:
            camera.cleanup()
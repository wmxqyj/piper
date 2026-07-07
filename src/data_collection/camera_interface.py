"""
相机接口封装

提供 RGB-D 相机的数据获取功能，支持 RealSense 和 Orbbec 相机
"""

import numpy as np
import time
import threading
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None

try:
    from pyorbbecsdk import (
        Context as OBContext,
        Pipeline as OBPipeline,
        Config as OBConfig,
        OBFormat,
        OBSensorType,
        OBError,
        OBLogLevel,
        AlignFilter,
        OBStreamType,
    )
except ImportError:
    OBContext = None
    OBPipeline = None
    OBConfig = None
    OBFormat = None
    OBSensorType = None
    OBError = None
    OBLogLevel = None
    AlignFilter = None
    OBStreamType = None


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


class OrbbecInterface:
    """Orbbec 相机接口 (DC1 / DaBai 系列)"""

    _LOG_LEVEL_MAP = {
        'DEBUG': OBLogLevel.DEBUG,
        'INFO': OBLogLevel.INFO,
        'WARNING': OBLogLevel.WARNING,
        'ERROR': OBLogLevel.ERROR,
        'FATAL': OBLogLevel.FATAL,
        'NONE': OBLogLevel.NONE,
    }

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pipeline = None
        self.device = None
        self.is_initialized = False

        self.width = config.get('resolution', {}).get('width', 640)
        self.height = config.get('resolution', {}).get('height', 480)
        self.fps = config.get('fps', 15)
        self.depth_scale = config.get('depth_scale', 0.001)
        self.align_depth = config.get('align_depth', True)
        self.log_level_str = config.get('log_level', 'WARNING').upper()

        self.intrinsics = None

        # 后台采集线程 & 帧缓存（避免阻塞调用端）
        self._capture_thread = None
        self._running = False
        self._latest_frame = None
        self._frame_lock = threading.Lock()
        self._align_filter = None

    def connect(self) -> bool:
        if OBPipeline is None:
            print("pyorbbecsdk 未安装: pip install pyorbbecsdk")
            return False

        try:
            log_level = self._LOG_LEVEL_MAP.get(self.log_level_str, OBLogLevel.WARNING)
            OBContext.set_logger_level(log_level)
            OBContext.set_logger_to_file(OBLogLevel.NONE, "")
            OBContext.set_logger_to_console(log_level)

            self.pipeline = OBPipeline()
            self.device = self.pipeline.get_device()
            device_info = self.device.get_device_info()
            print(f"Orbbec 设备: {device_info.get_name()}, PID: {hex(device_info.get_pid())}")

            config = OBConfig()

            try:
                depth_profiles = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
                if depth_profiles is not None:
                    depth_profile = depth_profiles.get_default_video_stream_profile()
                    print(f"深度流: {depth_profile.get_width()}x{depth_profile.get_height()} @ {depth_profile.get_fps()}fps")
                    config.enable_stream(depth_profile)
            except Exception as e:
                print(f"深度流配置失败: {e}")

            try:
                color_profiles = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
                if color_profiles is not None:
                    color_profile = color_profiles.get_default_video_stream_profile()
                    print(f"彩色流: {color_profile.get_width()}x{color_profile.get_height()} @ {color_profile.get_fps()}fps, format: {color_profile.get_format()}")
                    config.enable_stream(color_profile)
            except Exception as e:
                print(f"彩色流配置失败: {e}")

            try:
                self.pipeline.enable_frame_sync()
            except Exception:
                print("[INFO] DC1 不支持硬件帧同步，使用软件同步")

            self.pipeline.start(config)

            self._load_intrinsics()

            # 创建对齐滤波器（DC1/Gemini 330 使用 AlignFilter 而非 set_align_mode）
            self._align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM) if self.align_depth else None

            # 启动后台采集线程
            self._running = True
            self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._capture_thread.start()

            self.is_initialized = True
            print("Orbbec 相机已连接")
            return True

        except Exception as e:
            print(f"Orbbec 相机连接失败: {e}")
            return False

    def _load_intrinsics(self):
        try:
            camera_param = self.pipeline.get_camera_param()
            rgb_intrinsic = camera_param.rgb_intrinsic
            self.intrinsics = rgb_intrinsic
            print(f"相机内参:")
            print(f"  分辨率: {rgb_intrinsic.width} x {rgb_intrinsic.height}")
            print(f"  焦距: fx={rgb_intrinsic.fx}, fy={rgb_intrinsic.fy}")
            print(f"  主点: cx={rgb_intrinsic.cx}, cy={rgb_intrinsic.cy}")
        except Exception as e:
            print(f"获取相机内参失败: {e}")
            self.intrinsics = None

    def _capture_loop(self):
        """后台采集线程：持续拉取帧并缓存最新一帧"""
        import cv2
        while self._running:
            try:
                frames = self.pipeline.wait_for_frames(500)
                if frames is None:
                    continue

                # DC1/Gemini 330 使用 AlignFilter 进行后处理对齐
                if self._align_filter is not None:
                    frames = self._align_filter.process(frames)
                    if frames is None:
                        continue

                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                if color_frame is None or depth_frame is None:
                    continue

                color_image = self._frame_to_bgr_image(color_frame)
                if color_image is None:
                    continue

                depth_width = depth_frame.get_width()
                depth_height = depth_frame.get_height()
                depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
                depth_image = depth_data.reshape((depth_height, depth_width)).astype(np.float32)
                depth_image *= depth_frame.get_depth_scale()

                camera_intrinsics = self.get_intrinsics_matrix()
                camera_extrinsics = np.eye(4)

                frame = CameraData(
                    timestamp=time.time(),
                    rgb_image=color_image,
                    depth_image=depth_image,
                    camera_intrinsics=camera_intrinsics,
                    camera_extrinsics=camera_extrinsics,
                )

                with self._frame_lock:
                    self._latest_frame = frame

            except Exception as e:
                if self._running:
                    print(f"[WARN] 相机采集线程异常: {e}")

    def get_intrinsics_matrix(self) -> np.ndarray:
        if self.intrinsics is None:
            return np.eye(3)
        return np.array([
            [self.intrinsics.fx, 0, self.intrinsics.cx],
            [0, self.intrinsics.fy, self.intrinsics.cy],
            [0, 0, 1]
        ])

    @staticmethod
    def _frame_to_bgr_image(frame) -> Optional[np.ndarray]:
        import cv2
        width = frame.get_width()
        height = frame.get_height()
        fmt = frame.get_format()
        data = np.asanyarray(frame.get_data())

        if fmt == OBFormat.RGB:
            image = data.reshape((height, width, 3))
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif fmt == OBFormat.BGR:
            return data.reshape((height, width, 3))
        elif fmt == OBFormat.MJPG:
            return cv2.imdecode(data, cv2.IMREAD_COLOR)
        elif fmt == OBFormat.YUYV:
            image = data.reshape((height, width, 2))
            return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_YUYV)
        elif fmt == OBFormat.UYVY:
            image = data.reshape((height, width, 2))
            return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_UYVY)
        else:
            print(f"不支持的彩色格式: {fmt}")
            return None

    def get_camera_data(self) -> Optional[CameraData]:
        if not self.is_initialized:
            return None
        with self._frame_lock:
            return self._latest_frame

    def cleanup(self):
        # 停止后台采集线程
        self._running = False
        if self._capture_thread is not None and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2.0)
        try:
            if self.pipeline is not None:
                print("停止 Orbbec 相机...")
                self.pipeline.stop()
                print("Orbbec 相机已停止")
        except Exception as e:
            print(f"相机清理异常: {e}")

    def get_depth_at_pixel(self, depth_image: np.ndarray, x: int, y: int) -> float:
        return float(depth_image[y, x])

    def deproject_pixel_to_point(self, depth_image: np.ndarray, x: int, y: int) -> Tuple[float, float, float]:
        if self.intrinsics is None:
            return (0.0, 0.0, 0.0)
        depth = self.get_depth_at_pixel(depth_image, x, y)
        x_3d = (x - self.intrinsics.cx) * depth / self.intrinsics.fx
        y_3d = (y - self.intrinsics.cy) * depth / self.intrinsics.fy
        return (x_3d, y_3d, depth)


def create_camera_interface(config: Dict[str, Any]):
    """相机接口工厂函数"""
    camera_type = config.get('type', 'realsense')
    if camera_type == 'orbbec':
        return OrbbecInterface(config)
    else:
        return RealSenseInterface(config)
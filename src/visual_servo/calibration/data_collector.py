"""
手眼标定数据采集器

工作流程：
  1. 连接机械臂，设置为重力补偿（零力拖动）模式
  2. 连接 Orbbec 相机，显示实时预览
  3. 用户手动拖拽机械臂到不同位置/姿态
  4. 按 's' 键保存当前点位（拍照 + 检测棋盘格 + 记录机器人位姿）
  5. 按 'q' 键结束采集，保存数据文件
  6. 自动计算手眼标定矩阵
"""

import os
import sys
import cv2
import yaml
import time
import numpy as np
from typing import Optional, List, Dict, Any
from datetime import datetime

from pyAgxArm import create_agx_arm_config, AgxArmFactory

# 确保 src 目录在路径中，以便引入 data_collection.camera_interface
_src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from data_collection.camera_interface import OrbbecInterface
from .board_detector import ChessboardDetector
from .solver import HandEyeSolver
from .result_manager import CalibrationResult, CalibrationResultManager


class HandEyeDataCollector:
    """手眼标定数据采集器"""

    def __init__(self, config_path: str):
        """
        Parameters
        ----------
        config_path : str
            配置文件路径 (calibration.yaml)
        """
        self.config = self._load_config(config_path)
        self.cfg_calib = self.config.get("calibration", self.config)

        # 棋盘格参数
        chessboard_cfg = self.cfg_calib.get("chessboard", {})
        self.pattern_size = tuple(chessboard_cfg.get("pattern_size", [9, 6]))
        self.square_size_mm = float(chessboard_cfg.get("square_size_mm", 25.0))

        # 检测器
        self.detector = ChessboardDetector(self.pattern_size, self.square_size_mm)

        # 求解器
        self.solver = HandEyeSolver()

        # 相机
        self.camera = None
        self.camera_matrix = None
        self.dist_coeffs = None

        # 机械臂
        self.robot = None
        self.arm_cfg = self.cfg_calib.get("robot", {})

        # 采集数据缓存
        self.collected_data = []  # List of dict: {image, gripper_pose, board_in_cam, corners}
        self.save_dir = self.cfg_calib.get("data_collection", {}).get(
            "save_dir", "calibration_data"
        )
        self.num_target = self.cfg_calib.get("data_collection", {}).get("num_target", 15)

    def _load_config(self, config_path: str) -> dict:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def connect_robot(self) -> bool:
        """连接机械臂并设置为重力补偿模式"""
        try:
            robot_cfg = create_agx_arm_config(
                robot=self.arm_cfg.get("arm_model", "piper"),
                firmeware_version=self.arm_cfg.get("firmware_version", "default"),
                channel=self.arm_cfg.get("can_channel", "can0"),
            )
            print(f"连接机械臂 ({self.arm_cfg.get('can_channel', 'can0')})...")
            self.robot = AgxArmFactory.create_arm(robot_cfg)
            self.robot.connect()
            print("机械臂已连接")

            print("使能机械臂...")
            retry = 0
            while not self.robot.enable():
                time.sleep(0.01)
                retry += 1
                if retry % 500 == 0:
                    print(f"  等待使能... ({retry * 0.01:.1f}s)")

            print("设置重力补偿（零力拖动）模式...")
            self.robot.set_leader_mode()
            time.sleep(0.5)
            print("机械臂已进入拖动模式，可以手动拖拽")
            return True

        except Exception as e:
            print(f"机械臂连接失败: {e}")
            return False

    def connect_camera(self) -> bool:
        """连接 Orbbec 相机"""
        try:
            camera_cfg = self.cfg_calib.get("camera", {})
            camera_cfg["type"] = "orbbec"
            print(f"连接 Orbbec 相机...")
            self.camera = OrbbecInterface(camera_cfg)
            if not self.camera.connect():
                print("相机连接失败")
                return False

            # 等待获取第一帧，确保内参已加载
            for _ in range(30):
                data = self.camera.get_camera_data()
                if data is not None:
                    self.camera_matrix = data.camera_intrinsics
                    break
                time.sleep(0.1)

            if self.camera_matrix is None:
                print("无法获取相机内参")
                return False

            print(f"相机内参已加载:")
            print(f"  fx={self.camera_matrix[0,0]:.4f}, fy={self.camera_matrix[1,1]:.4f}")
            print(f"  cx={self.camera_matrix[0,2]:.4f}, cy={self.camera_matrix[1,2]:.4f}")
            return True

        except Exception as e:
            print(f"相机连接失败: {e}")
            return False

    def _get_robot_pose(self) -> Optional[np.ndarray]:
        """
        获取机械臂末端法兰在基坐标系下的位姿

        在 leader（重力补偿拖动）模式下不能使用 get_flange_pose()，
        需要先用 get_leader_joint_angles() 获取实际关节角度，
        再用 FK 正运动学计算末端位姿。
        """
        try:
            # 1. 获取 leader 模式下的实际关节角度
            joint_msg = self.robot.get_leader_joint_angles()
            if joint_msg is None or joint_msg.msg is None:
                print("  [WARN] get_leader_joint_angles() 返回空")
                # 降级尝试 get_flange_pose
                pose_msg = self.robot.get_flange_pose()
                if pose_msg is not None and pose_msg.msg is not None:
                    return np.array(pose_msg.msg)
                return None

            joint_angles = joint_msg.msg  # list[float]

            # 2. 用 FK 正运动学计算法兰位姿
            flange_pose = self.robot.fk(joint_angles)
            if flange_pose is None:
                print("  [WARN] FK 计算返回空")
                return None

            return np.array(flange_pose)  # [x, y, z, roll, pitch, yaw]

        except Exception as e:
            print(f"获取机械臂位姿失败: {e}")
            return None

    def run_collection(self):
        """
        运行数据采集主循环

        按键说明:
          s - 保存当前点位（需检测到棋盘格）
          q - 结束采集
        """
        if self.camera is None or self.robot is None:
            print("请先连接相机和机械臂")
            return

        if self.camera_matrix is None:
            print("相机内参未加载")
            return

        print("\n" + "=" * 60)
        print("手眼标定数据采集")
        print("=" * 60)
        print(f"棋盘格: {self.pattern_size[0]}x{self.pattern_size[1]}, "
              f"方格 {self.square_size_mm}mm")
        print(f"目标采集点数: {self.num_target}")
        print("\n操作说明:")
        print("  1. 拖拽机械臂到不同位置/姿态")
        print("  2. 确保棋盘格在相机视野中清晰可见")
        print("  3. 按 's' 键保存当前点位（绿色=成功, 红色=失败）")
        print("  4. 按 'q' 键结束采集并计算标定")
        print("=" * 60)

        window_name = "手眼标定 - 数据采集"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 960, 720)

        self.collected_data = []
        last_save_time = 0
        save_cooldown = 0.5  # 两次保存最小间隔（秒）

        while True:
            # 获取相机数据
            camera_data = self.camera.get_camera_data()
            if camera_data is None:
                time.sleep(0.03)
                continue

            frame = camera_data.rgb_image.copy()

            # 检测棋盘格
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            success, corners = self.detector.detect_corners(gray)

            # 绘制检测结果
            display = self.detector.draw_corners(frame, corners, self.pattern_size, success)

            # 获取当前机械臂位姿（用于实时显示）
            current_pose = self._get_robot_pose()

            # 显示状态信息
            status_color = (0, 255, 0) if success else (0, 0, 255)
            status_text = "Board detected" if success else "No board"
            cv2.putText(
                display, status_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2,
            )
            cv2.putText(
                display, f"Saved: {len(self.collected_data)}/{self.num_target}",
                (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
            )

            # 实时显示当前机器人位姿
            if current_pose is not None:
                pos_str = f"pos: ({current_pose[0]:.3f}, {current_pose[1]:.3f}, {current_pose[2]:.3f})"
                rpy_str = f"rpy: ({np.degrees(current_pose[3]):.1f}, {np.degrees(current_pose[4]):.1f}, {np.degrees(current_pose[5]):.1f})"
                cv2.putText(
                    display, pos_str, (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2,
                )
                cv2.putText(
                    display, rpy_str, (10, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2,
                )

            if success and corners is not None:
                # 进一步计算棋盘格位姿
                ret, board_in_cam = self.detector.estimate_board_pose(
                    corners, self.camera_matrix, self.dist_coeffs
                )
                if ret:
                    display = self.detector.draw_board_axis(
                        display, board_in_cam, self.camera_matrix, self.dist_coeffs,
                        axis_length=0.04,
                    )

            # 显示
            cv2.imshow(window_name, display)
            key = cv2.waitKey(30) & 0xFF

            now = time.time()

            if key == ord("s"):
                if now - last_save_time < save_cooldown:
                    print("  操作太频繁，请稍后再按")
                    continue

                last_save_time = now

                if not success or corners is None:
                    print("  [失败] 未检测到棋盘格，请调整位置")
                    continue

                # 计算棋盘格位姿
                ret, board_in_cam = self.detector.estimate_board_pose(
                    corners, self.camera_matrix, self.dist_coeffs
                )
                if not ret:
                    print("  [失败] 无法计算棋盘格位姿")
                    continue

                # 获取机械臂位姿
                gripper_pose = self._get_robot_pose()
                if gripper_pose is None:
                    print("  [失败] 无法获取机械臂位姿")
                    continue

                # 计算重投影误差
                reproj_err = self.detector.compute_reprojection_error(
                    corners, board_in_cam, self.camera_matrix, self.dist_coeffs
                )

                data_point = {
                    "index": len(self.collected_data) + 1,
                    "timestamp": now,
                    "image": frame.copy(),
                    "gray": gray.copy(),
                    "corners": corners,
                    "gripper_pose": gripper_pose,  # [x, y, z, roll, pitch, yaw]
                    "board_in_cam": board_in_cam,  # 4x4 matrix
                    "reprojection_error_px": reproj_err,
                }
                self.collected_data.append(data_point)

                pos_str = f"({gripper_pose[0]:.3f}, {gripper_pose[1]:.3f}, {gripper_pose[2]:.3f})"
                print(
                    f"  [{len(self.collected_data)}/{self.num_target}] "
                    f"已保存 | 位置: {pos_str} m | "
                    f"重投影误差: {reproj_err:.2f} px"
                )

                if len(self.collected_data) >= self.num_target:
                    print(f"\n已达到目标采集点数 ({self.num_target})，自动结束采集")
                    break

            elif key == ord("q"):
                print("\n用户手动结束采集")
                break

        cv2.destroyWindow(window_name)

        # 显示采集统计
        print("\n" + "=" * 60)
        print(f"采集完成: {len(self.collected_data)} 个点位")
        print("=" * 60)

    def save_data(self, custom_dir: Optional[str] = None) -> str:
        """
        保存采集的原始数据

        Parameters
        ----------
        custom_dir : str, optional
            自定义保存目录

        Returns
        -------
        str
            保存目录路径
        """
        if len(self.collected_data) == 0:
            print("没有数据可保存")
            return ""

        if custom_dir:
            save_dir = custom_dir
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = os.path.join(self.save_dir, f"calib_data_{timestamp}")

        os.makedirs(save_dir, exist_ok=True)

        # 保存为 npz
        n = len(self.collected_data)
        images = np.stack([d["image"] for d in self.collected_data])
        grays = np.stack([d["gray"] for d in self.collected_data])
        gripper_poses = np.stack([d["gripper_pose"] for d in self.collected_data])
        board_in_cam_list = np.stack([d["board_in_cam"] for d in self.collected_data])

        npz_path = os.path.join(save_dir, "calibration_data.npz")
        np.savez(
            npz_path,
            images=images,
            grays=grays,
            gripper_poses=gripper_poses,
            board_in_cam_list=board_in_cam_list,
            camera_matrix=self.camera_matrix,
            dist_coeffs=self.dist_coeffs if self.dist_coeffs is not None else np.zeros(5),
            pattern_size=np.array(self.pattern_size),
            square_size_mm=self.square_size_mm,
            timestamps=np.array([d["timestamp"] for d in self.collected_data]),
            reprojection_errors=np.array(
                [d["reprojection_error_px"] for d in self.collected_data]
            ),
        )
        print(f"原始数据已保存: {npz_path}")

        # 同时保存可读的文本信息
        info_path = os.path.join(save_dir, "collection_info.txt")
        with open(info_path, "w") as f:
            f.write(f"手眼标定数据采集信息\n")
            f.write(f"采集时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"样本数: {n}\n")
            f.write(f"棋盘格: {self.pattern_size[0]}x{self.pattern_size[1]}, "
                    f"{self.square_size_mm}mm/格\n")
            f.write(f"\n位姿列表:\n")
            for i, d in enumerate(self.collected_data):
                p = d["gripper_pose"]
                f.write(
                    f"  [{i+1:2d}] "
                    f"pos=({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}) "
                    f"rpy=({np.degrees(p[3]):.1f}, {np.degrees(p[4]):.1f}, {np.degrees(p[5]):.1f})° "
                    f"reproj={d['reprojection_error_px']:.2f}px\n"
                )

        print(f"采集信息已保存: {info_path}")
        return save_dir

    def compute_calibration(self) -> Optional[CalibrationResult]:
        """
        使用采集的数据计算手眼标定矩阵

        Returns
        -------
        CalibrationResult or None
        """
        if len(self.collected_data) < 3:
            print(f"数据不足（{len(self.collected_data)} < 3），无法计算标定")
            return None

        algorithm = self.cfg_calib.get("solver", {}).get("algorithm", "tsai_lenz")

        gripper_poses = [d["gripper_pose"] for d in self.collected_data]
        board_in_cam_poses = [d["board_in_cam"] for d in self.collected_data]

        print(f"\n计算手眼标定矩阵... (算法: {algorithm})")
        print(f"使用 {len(gripper_poses)} 组数据")

        try:
            X, info = self.solver.solve_eye_in_hand(
                gripper_poses, board_in_cam_poses, algorithm
            )

            # 验证：棋盘格在基坐标系下的一致性
            verify_info = self.solver.verify_calibration(
                X, gripper_poses, board_in_cam_poses
            )

            result = CalibrationResult(
                gripper_T_cam=X,
                rot_error_deg_mean=info["rot_error_deg_mean"],
                rot_error_deg_std=info["rot_error_deg_std"],
                trans_error_m_mean=info["trans_error_m_mean"],
                trans_error_m_std=info["trans_error_m_std"],
                position_std_mm=verify_info["position_std_mm"],
                num_samples=len(gripper_poses),
                algorithm=algorithm,
            )

            return result

        except Exception as e:
            print(f"标定计算失败: {e}")
            return None

    def cleanup(self):
        """清理资源"""
        print("\n清理资源...")
        if self.camera is not None:
            self.camera.cleanup()
        if self.robot is not None:
            try:
                # 退出重力补偿模式
                self.robot.set_follower_mode()
                self.robot.disable()
                self.robot.disconnect()
            except Exception:
                pass
        print("清理完成")


def main():
    """入口函数"""
    import sys

    # 配置文件路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config", "calibration.yaml")

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

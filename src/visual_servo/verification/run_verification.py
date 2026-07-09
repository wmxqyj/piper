"""
试管架视觉抓取验证 — 平面分割 + 圆形孔检测 + Skills 流水线

流程：
  1. 连接机械臂（CAN 控制模式）、Orbbec 相机、加载手眼标定
  2. RackDetector 实时检测试管架，识别源孔（有试管）和目标孔（空孔）
  3. SkillExecutor 按流水线执行抓取-移载动作
  4. 按 's' → 执行流水线，'q' → 退出

坐标变换链：
  base_T_hole = base_T_gripper * gripper_T_cam * cam_T_hole
"""

import os
import sys
import cv2
import yaml
import time
import numpy as np
from typing import Optional

from pyAgxArm import create_agx_arm_config, AgxArmFactory

_src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from data_collection.camera_interface import OrbbecInterface
from visual_servo.calibration.result_manager import CalibrationResultManager
from visual_servo.calibration.solver import HandEyeSolver
from visual_servo.verification.yolo_rack_detector import YoloRackDetector, RackDetectionResult
from visual_servo.verification.skill_executor import SkillExecutor


class TubeTransferVerification:
    """试管抓取移载验证器"""

    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.robot = None
        self.camera = None
        self.detector = YoloRackDetector(self.config)
        self.executor: Optional[SkillExecutor] = None
        self.gripper_T_cam = None
        self.camera_matrix = None
        self._saved_stderr = None  # 保存的 stderr fd（压制 Orbbec 警告用）

        # 流水线配置
        self.pipeline_config = self.config.get("pipeline", [])

        # 数据集采集
        _base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self._dataset_dir = os.path.join(_base, "dataset", "rack_holes", "images")
        os.makedirs(self._dataset_dir, exist_ok=True)
        self._capture_count = 0

    # ──────────────────────────────────────────────
    # 连接与初始化
    # ──────────────────────────────────────────────

    def load_calibration(self) -> bool:
        """加载手眼标定结果"""
        calib_path = self.config.get("calibration_result_path", "")

        if not calib_path:
            base_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "calibration_data",
            )
            if not os.path.exists(base_dir):
                print(f"标定数据目录不存在: {base_dir}")
                return False
            subdirs = sorted(
                [d for d in os.listdir(base_dir) if d.startswith("calib_data_")],
                reverse=True,
            )
            if not subdirs:
                print("未找到标定结果")
                return False
            calib_path = os.path.join(base_dir, subdirs[0], "calibration_result.yaml")

        if not os.path.exists(calib_path):
            print(f"标定文件不存在: {calib_path}")
            return False

        result = CalibrationResultManager.load(calib_path)
        if result is None:
            return False

        self.gripper_T_cam = result.gripper_T_cam
        print(f"已加载标定结果: {calib_path}")
        print(f"  误差: 旋转={result.rot_error_deg_mean:.3f}°, "
              f"平移={result.trans_error_m_mean*1000:.2f}mm")
        return True

    def connect_robot(self) -> bool:
        """连接机械臂，使能并切换到 CAN 控制模式"""
        robot_cfg = self.config.get("robot", {})
        try:
            cfg = create_agx_arm_config(
                robot=robot_cfg.get("arm_model", "piper"),
                firmeware_version=robot_cfg.get("firmware_version", "default"),
                channel=robot_cfg.get("can_channel", "can0"),
            )
            print(f"连接机械臂 ({robot_cfg.get('can_channel', 'can0')})...")
            self.robot = AgxArmFactory.create_arm(cfg)
            self.robot.connect()
            print("机械臂已连接")

            print("使能机械臂...")
            enable_ok = False
            for retry_attempt in range(2000):
                ret = self.robot.enable()
                if retry_attempt % 100 == 0:
                    print("  enable:", ret, "joints:",
                          self.robot.get_joints_enable_status_list())
                if ret:
                    enable_ok = True
                    break
                time.sleep(0.01)
            if not enable_ok:
                print("使能超时，请检查 CAN 总线")
                return False

            print("使能完成")
            st = self.robot.get_arm_status()
            if st and st.msg:
                print(f"  ctrl_mode={st.msg.ctrl_mode}, "
                      f"arm_status={st.msg.arm_status}")
            print("机械臂就绪")

            # 初始化执行器
            self.executor = SkillExecutor(
                self.config, self.robot, self.gripper_T_cam)
            return True

        except Exception as e:
            print(f"机械臂连接失败: {e}")
            return False

    def connect_camera(self) -> bool:
        """连接相机（压制 Orbbec SDK 底层噪音输出，持续到 cleanup）"""
        camera_cfg = self.config.get("camera", {})
        camera_cfg["type"] = "orbbec"

        # 重定向 stderr → /dev/null，压制 Align.cpp 等 C++ 层 warning
        self._saved_stderr = os.dup(2)
        null_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(null_fd, 2)
        os.close(null_fd)

        try:
            self.camera = OrbbecInterface(camera_cfg)
            if not self.camera.connect():
                os.dup2(self._saved_stderr, 2)
                os.close(self._saved_stderr)
                self._saved_stderr = None
                return False
            for _ in range(30):
                data = self.camera.get_camera_data()
                if data is not None:
                    self.camera_matrix = data.camera_intrinsics
                    break
                time.sleep(0.1)
            if self.camera_matrix is None:
                print("无法获取相机内参")
                os.dup2(self._saved_stderr, 2)
                os.close(self._saved_stderr)
                self._saved_stderr = None
                return False
            # 保持 stderr 压制状态（restore in cleanup）
            print("相机已连接（Orbbec SDK 警告已静音）")
            return True
        except Exception as e:
            os.dup2(self._saved_stderr, 2)
            os.close(self._saved_stderr)
            self._saved_stderr = None
            print(f"相机连接失败: {e}")
            return False

    # ──────────────────────────────────────────────
    # 主循环
    # ──────────────────────────────────────────────

    def run(self):
        """运行验证主循环"""
        if not self._check_ready():
            return

        print("\n" + "=" * 60)
        print("试管架视觉抓取验证")
        print("=" * 60)
        print("操作说明:")
        print("  1. 将试管架放在相机视野内")
        print("  2. 调整机械臂位置使架面可见")
        print("  3. 实时显示检测结果（绿色=有试管, 蓝色=空孔）")
        print("  4. 按 'd' → 锁定当前检测结果（冻结孔位信息）")
        print("  5. 按 's' → 用锁定的结果执行完整抓取-移载流水线")
        print("  6. 按 'v' → 移动到源孔验证精度（先按 d 锁定）")
        print("  7. 按 'x' → 移动到目标孔验证精度（先按 d 锁定）")
        print("  8. 按 'q' 退出")
        print("=" * 60)

        window_name = "Tuberack Visual Servoing"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 960, 720)

        snapshot: Optional[RackDetectionResult] = None  # 锁定的检测快照

        while True:
            camera_data = self.camera.get_camera_data()
            if camera_data is None:
                time.sleep(0.03)
                continue

            frame = camera_data.rgb_image.copy()
            depth = camera_data.depth_image

            # 实时检测试管架（每一帧都重新检测）
            live_result = self.detector.detect(frame, depth, self.camera_matrix)

            # 绘制实时检测结果
            display = self.detector.draw_detection(frame, live_result)

            # 显示机械臂状态
            pose = self._get_robot_pose()
            if pose is not None:
                pt = f"robot: ({pose[0]:.2f}, {pose[1]:.2f}, {pose[2]:.2f})"
                cv2.putText(display, pt, (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # 显示锁定状态
            status_line = ""
            if snapshot is not None:
                src = snapshot.source_hole
                tgt = snapshot.target_hole
                src_str = f"R{src.index[0]}C{src.index[1]}" if src else "?"
                tgt_str = f"R{tgt.index[0]}C{tgt.index[1]}" if tgt else "?"
                status_line = f"LOCKED: {src_str}->{tgt_str}  "

            # 提示（含采集计数）
            cap_info = f"  [c] capture({self._capture_count})"
            hint = f"{status_line}[d] detect  [s] pipeline{cap_info}  v=src  x=tgt  [q] quit"
            cv2.putText(display, hint, (10, display.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

            cv2.imshow(window_name, display)
            key = cv2.waitKey(30) & 0xFF

            if key == ord("d"):
                # 锁定：优先用成功结果，否则有孔位也可锁定
                if live_result.success or (live_result.holes and len(live_result.holes) > 0):
                    snapshot = live_result
                    src = snapshot.source_hole
                    tgt = snapshot.target_hole
                    print(f"\n[检测] 结果已锁定"
                          f"  源孔: R{src.index[0]}C{src.index[1]}" if src else "")
                    print(f"        目标孔: R{tgt.index[0]}C{tgt.index[1]}" if tgt else "")
                    print(f"        按 'v/x' 验证位置，按 's' 执行流水线，按 'd' 重新检测")
                else:
                    print("\n[检测] 当前帧未检测到任何孔位，锁定失败")

            elif key == ord("s"):
                if snapshot is None:
                    print("尚未锁定检测结果，请先按 'd' 检测")
                    continue
                if snapshot.source_hole is None:
                    print("锁定结果中未找到有试管的源孔")
                    continue
                if snapshot.target_hole is None:
                    print("锁定结果中未找到空的目标孔")
                    continue

                self._execute_pipeline(snapshot)

            elif key == ord("v"):
                self._move_to_verify(snapshot, "source", 0.15)

            elif key == ord("x"):
                self._move_to_verify(snapshot, "target", 0.15)

            elif key == ord("c"):
                self._capture_frame(frame, depth)

            elif key == ord("q"):
                print("退出")
                break

        cv2.destroyWindow(window_name)

    # ──────────────────────────────────────────────
    # 点位验证
    # ──────────────────────────────────────────────

    def _move_to_verify(self, snapshot, hole_type: str, offset_z: float):
        """移动到指定孔位验证精度"""
        if snapshot is None:
            print("请先按 'd' 锁定检测结果")
            return

        hole = snapshot.source_hole if hole_type == "source" else snapshot.target_hole
        if hole is None:
            print(f"锁定结果中无{hole_type}孔")
            return

        print(f"\n[验证] 移动至{hole_type}孔 R{hole.index[0]}C{hole.index[1]}"
              f"  @ ({hole.position_3d[0]:.3f}, {hole.position_3d[1]:.3f}, "
              f"{hole.position_3d[2]:.3f})")

        # 复用 executor 内部变换逻辑
        cfg = {hole_type + "_hole": hole, "offset_z": offset_z,
               "speed": 0.1, "hole_ref": hole_type + "_hole"}
        self.executor._move_to_hole(cfg)

    # ──────────────────────────────────────────────
    # 流水线执行
    # ──────────────────────────────────────────────

    def _execute_pipeline(self, result: RackDetectionResult):
        """执行完整抓取-移载流水线"""
        print("\n" + "-" * 50)
        print("执行抓取-移载流水线")
        print(f"  源孔: R{result.source_hole.index[0]}C{result.source_hole.index[1]}"
              f"  @ ({result.source_hole.position_3d[0]:.3f}, "
              f"{result.source_hole.position_3d[1]:.3f}, "
              f"{result.source_hole.position_3d[2]:.3f})")
        print(f"  目标孔: R{result.target_hole.index[0]}C{result.target_hole.index[1]}"
              f"  @ ({result.target_hole.position_3d[0]:.3f}, "
              f"{result.target_hole.position_3d[1]:.3f}, "
              f"{result.target_hole.position_3d[2]:.3f})")
        print("-" * 50)

        ok = self.executor.execute_pipeline(
            self.pipeline_config,
            source_hole=result.source_hole,
            target_hole=result.target_hole,
        )

        if ok:
            print("\n  [OK] 流水线执行成功")
        else:
            print("\n  [FAIL] 流水线执行失败")

    # ──────────────────────────────────────────────
    # 辅助方法
    # ──────────────────────────────────────────────

    def _check_ready(self) -> bool:
        return all([
            self.camera is not None,
            self.robot is not None,
            self.gripper_T_cam is not None,
            self.executor is not None,
        ])

    def _get_robot_pose(self) -> Optional[np.ndarray]:
        msg = self.robot.get_flange_pose()
        if msg is not None and msg.msg is not None:
            return np.array(msg.msg)
        return None

    def _capture_frame(self, frame: np.ndarray, depth: np.ndarray):
        """按 'c' 保存当前帧到数据集目录"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"rack_{timestamp}_{self._capture_count:03d}.png"
        filepath = os.path.join(self._dataset_dir, filename)
        cv2.imwrite(filepath, frame)
        self._capture_count += 1
        print(f"\n[采集] 已保存: {filename}  (共 {self._capture_count} 张)")

    def cleanup(self):
        """清理资源"""
        print("\n清理资源...")

        # 恢复 stderr（在相机清理之前，这样相机报错能正常打印）
        if self._saved_stderr is not None:
            os.dup2(self._saved_stderr, 2)
            os.close(self._saved_stderr)
            self._saved_stderr = None

        if self.camera is not None:
            self.camera.cleanup()
        if self.robot is not None:
            try:
                MsgCls = self.robot._MSG_ModeCtrl
                self.robot._send_msg(MsgCls(ctrl_mode=0x00))
                time.sleep(0.2)
                self.robot.disable()
                self.robot.disconnect()
            except Exception:
                pass
        print("清理完成")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config", "verification.yaml")

    if not os.path.exists(config_path):
        print(f"配置文件不存在: {config_path}")
        sys.exit(1)

    verifier = TubeTransferVerification(config_path)

    try:
        if not verifier.load_calibration():
            print("加载标定结果失败")
            return

        if not verifier.connect_robot():
            print("机械臂连接失败")
            return

        if not verifier.connect_camera():
            print("相机连接失败")
            return

        print("\n系统就绪，按 's' 开始执行流水线，'q' 退出")
        verifier.run()

    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"运行异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        verifier.cleanup()


if __name__ == "__main__":
    main()

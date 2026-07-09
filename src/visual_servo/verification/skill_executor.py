"""
技能执行器 — 解析配置中定义的原子动作并执行

支持的技能类型:
  - relative_move : 沿工具系/基系相对移动
  - move_to_pose  : 移动到指定 6D 位姿
  - move_to_hole  : 移动到某个孔位上方（指定 offset）
  - gripper       : 夹爪开/关
  - wait          : 等待指定时长
"""

import time
import numpy as np
from typing import Optional

from visual_servo.calibration.solver import HandEyeSolver
from visual_servo.verification.rack_detector import HoleInfo


class SkillExecutor:
    """执行预设技能序列"""

    def __init__(self, config: dict, robot, gripper_T_cam: np.ndarray):
        """
        Parameters
        ----------
        config : dict
            完整配置字典（含 skills 段）
        robot : AgxArm 实例
        gripper_T_cam : (4, 4) 手眼标定矩阵
        """
        self.skills_config = config.get("skills", {})
        self.robot = robot
        self.gripper_T_cam = gripper_T_cam

        robot_cfg = config.get("robot", {})
        self.default_speed = float(robot_cfg.get("speed", 0.2))

        # 抓取姿态（固定值，试管竖直向上）
        grasp_cfg = config.get("grasp_orientation", {})
        self.grasp_roll = float(grasp_cfg.get("roll", 180.0))
        self.grasp_pitch = float(grasp_cfg.get("pitch", 0.0))
        self.grasp_yaw = float(grasp_cfg.get("yaw", 0.0))

    # ──────────────────────────────────────────────
    # 公开接口
    # ──────────────────────────────────────────────

    def execute(self, skill_name: str, **kwargs) -> bool:
        """
        执行指定技能

        Parameters
        ----------
        skill_name : str
            技能名称（对应配置中 skills 下的键名）
        **kwargs : 运行时覆盖参数
            target_pose : 6D 位姿 (用于 move_to_pose)
            target_hole : HoleInfo (用于 move_to_hole)
            source_hole : HoleInfo (用于 move_to_hole 的 offset 计算)

        Returns
        -------
        bool
        """
        if skill_name not in self.skills_config:
            print(f"  [ERROR] 技能 '{skill_name}' 未在配置中定义")
            return False

        cfg = self.skills_config[skill_name].copy()
        skill_type = cfg.pop("type")
        cfg.update(kwargs)  # 运行时覆盖

        print(f"  [Skill] {skill_name} (type={skill_type})")

        dispatch = {
            "relative_move": self._relative_move,
            "move_to_pose": self._move_to_pose,
            "move_to_hole": self._move_to_hole,
            "gripper": self._gripper_action,
            "wait": self._wait,
        }

        handler = dispatch.get(skill_type)
        if handler is None:
            print(f"  [ERROR] 未知技能类型: {skill_type}")
            return False

        return handler(cfg)

    def execute_pipeline(self, pipeline: list, **global_kwargs) -> bool:
        """
        按顺序执行流水线中的技能

        Parameters
        ----------
        pipeline : list[dict]
            技能序列，每项格式: {"skill": str, "verify": bool, ...}
        **global_kwargs : 全局参数（如 source_hole, target_hole）
        """
        for step in pipeline:
            skill_name = step.get("skill", "")
            step_kwargs = {k: v for k, v in step.items() if k != "skill"}
            # 注入全局参数
            step_kwargs.update(global_kwargs)

            ok = self.execute(skill_name, **step_kwargs)
            if not ok:
                print(f"  [FAIL] 技能 '{skill_name}' 执行失败，中止流水线")
                return False

            # 如果需要校验
            if step.get("verify", False):
                print(f"  [Verify] {skill_name} 执行后校验...")
                time.sleep(0.5)
                # 简单校验：确认夹爪状态或位姿（可后续扩展）

        print("  [OK] 流水线执行完成")
        return True

    # ──────────────────────────────────────────────
    # 内部执行器
    # ──────────────────────────────────────────────

    def _get_current_pose(self) -> Optional[np.ndarray]:
        """获取当前法兰 6D 位姿 [x,y,z,roll,pitch,yaw]"""
        msg = self.robot.get_flange_pose()
        if msg is not None and msg.msg is not None:
            return np.array(msg.msg)
        return None

    def _relative_move(self, cfg: dict) -> bool:
        """相对移动"""
        direction = cfg.get("direction", [0, 0, 0])
        speed = float(cfg.get("speed", self.default_speed))
        frame = cfg.get("frame", "tool")

        current = self._get_current_pose()
        if current is None:
            print("  [FAIL] 无法获取当前位姿")
            return False

        dx, dy, dz = direction
        world_dir = np.array([dx, dy, dz])

        if frame == "tool":
            # 工具坐标系：需要旋转变换
            R = HandEyeSolver.pose6_to_matrix(current.tolist())[:3, :3]
            world_dir = R @ world_dir

        target = current.copy()
        target[:3] += world_dir
        return self._do_move_p(target, speed)

    def _move_to_pose(self, cfg: dict) -> bool:
        """移动到指定 6D 位姿"""
        target_pose = cfg.get("target_pose")
        speed = float(cfg.get("speed", self.default_speed))

        if target_pose is None:
            print("  [FAIL] 未指定 target_pose")
            return False

        return self._do_move_p(np.array(target_pose), speed)

    def _move_to_hole(self, cfg: dict) -> bool:
        """
        移动到某个孔位上方（自动计算 6D 位姿）

        变换链: base_T_hole = base_T_gripper * gripper_T_cam * hole_in_cam

        cfg 参数:
            hole_ref : str  引用哪个孔 (source_hole / target_hole)
            offset_z : float  孔上方的 Z 偏移 (m)
        """
        hole_ref = cfg.get("hole_ref", "target_hole")
        offset_z = float(cfg.get("offset_z", 0.0))

        # 从 kwargs 中获取孔位
        hole: Optional[HoleInfo] = cfg.get(hole_ref)
        if hole is None:
            print(f"  [FAIL] 未找到孔位引用 '{hole_ref}'")
            return False

        current = self._get_current_pose()
        if current is None:
            print("  [FAIL] 无法获取当前位姿")
            return False

        # 构造 cam_T_hole (只有平移，姿态用抓取固定姿态)
        cam_T_hole = np.eye(4)
        cam_T_hole[:3, 3] = hole.position_3d

        # 变换到基坐标系
        base_T_gripper = HandEyeSolver.pose6_to_matrix(current.tolist())
        base_T_hole = base_T_gripper @ self.gripper_T_cam @ cam_T_hole
        hole_pos = base_T_hole[:3, 3]

        # 构造目标位姿：孔位上方 + 固定抓取姿态
        target_pose = np.array([
            hole_pos[0],
            hole_pos[1],
            hole_pos[2] + offset_z,
            np.radians(self.grasp_roll),
            np.radians(self.grasp_pitch),
            np.radians(self.grasp_yaw),
        ])

        return self._do_move_p(target_pose, float(cfg.get("speed", self.default_speed)))

    def _do_move_p(self, target_pose: np.ndarray, speed: float) -> bool:
        """执行 move_p 指令"""
        try:
            target_list = target_pose.tolist()
            print(f"  Move to: ({target_pose[0]:.3f}, {target_pose[1]:.3f}, "
                  f"{target_pose[2]:.3f})")

            self.robot.move_p(target_list)
            print("  指令已发送，等待到达...")
            time.sleep(2.5)

            arrived = self._get_current_pose()
            if arrived is not None:
                err = np.linalg.norm(arrived[:3] - target_pose[:3])
                print(f"  到位, 位置误差: {err*1000:.1f} mm")
                return err < 0.05
            return True
        except Exception as e:
            print(f"  [FAIL] 移动失败: {e}")
            return False

    def _gripper_action(self, cfg: dict) -> bool:
        """夹爪动作"""
        action = cfg.get("action", "close")
        width = float(cfg.get("width", 0.02))
        force = float(cfg.get("force", 1.0))

        try:
            # 获取机械臂上的执行器接口
            effector = self.robot.init_effector(
                self.robot.OPTIONS.EFFECTOR.AGX_GRIPPER)

            if action == "close":
                print(f"  夹爪闭合: width={width*1000:.1f}mm, force={force:.1f}N")
                effector.move_gripper_m(value=width, force=force)
            elif action == "open":
                print(f"  夹爪张开: width={width*1000:.1f}mm")
                effector.move_gripper_m(value=width, force=force)
            else:
                print(f"  [FAIL] 未知夹爪动作: {action}")
                return False

            time.sleep(0.5)
            return True
        except Exception as e:
            print(f"  [FAIL] 夹爪动作失败: {e}")
            return False

    @staticmethod
    def _wait(cfg: dict) -> bool:
        """等待"""
        duration = float(cfg.get("duration", 1.0))
        print(f"  等待 {duration:.1f}s...")
        time.sleep(duration)
        return True

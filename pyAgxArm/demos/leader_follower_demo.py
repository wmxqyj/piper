#!/usr/bin/env python3
"""
主从（Leader-Follower）联动示例 —— 软件转发版（含夹爪跟随）

适用场景:
  两个机械臂分别连接在不同的 CAN 接口上（如 can_piper_l 和 can_piper_r），
  由于两条 CAN 总线物理隔离，固件级主从无法跨总线通信。
  此脚本通过 Python 层实时读取 Leader 臂关节角度和夹爪状态，并转发给 Follower 臂。

硬件准备:
  - 两个机械臂分别通过两个 USB-CAN 模块连接电脑
  - can_piper_l 接臂 A（设为 Leader，可拖拽）
  - can_piper_r 接臂 B（设为 Follower，自动跟随）
  - 两臂末端均安装 AgxGripper 夹爪

前置条件（Linux）:
  sudo ip link set can_piper_l up type can bitrate 1000000
  sudo ip link set can_piper_r up type can bitrate 1000000

用法:
  python3 leader_follower_demo.py                       # 默认 Piper
  python3 leader_follower_demo.py --arm nero            # Nero
  python3 leader_follower_demo.py --leader-can can0 --follower-can can1
"""

import time
import signal
import sys
import argparse
from platform import system

from pyAgxArm import create_agx_arm_config, AgxArmFactory, ArmModel, PiperFW, NeroFW


# ---------------------------------------------------------------------------
# 配置解析
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Leader-Follower 主从联动（软件转发，含夹爪）")
parser.add_argument("--arm", choices=["piper", "nero"], default="piper",
                    help="机械臂系列（默认 piper）")
parser.add_argument("--leader-can", default="can_piper_l",
                    help="主臂（Leader）的 CAN 通道（默认 can_piper_l）")
parser.add_argument("--follower-can", default="can_piper_r",
                    help="从臂（Follower）的 CAN 通道（默认 can_piper_r）")
parser.add_argument("--hz", type=int, default=50,
                    help="转发频率 Hz（默认 50）")
parser.add_argument("--effector", default="agx_gripper",
                    choices=["agx_gripper", "revo2", "none"],
                    help="末端执行器类型（默认 agx_gripper，设为 none 禁用夹爪跟随）")
args = parser.parse_args()

leader_robot = None
follower_robot = None
leader_effector = None
follower_effector = None


# ---------------------------------------------------------------------------
# 信号处理：Ctrl+C 安全退出
# ---------------------------------------------------------------------------
def signal_handler(sig, frame):
    print("\n\n正在安全退出，请稍候...")
    cleanup()
    sys.exit(0)


def cleanup():
    """退出前恢复并失能两臂。"""
    global leader_robot, follower_robot
    try:
        if leader_robot is not None and leader_robot.is_ok():
            print("退出 Leader 模式...")
            leader_robot.set_follower_mode()  # 退出零力拖动
            time.sleep(0.3)
            leader_robot.disable()

        if follower_robot is not None and follower_robot.is_ok():
            print("失能 Follower 臂...")
            follower_robot.disable()

        if leader_robot is not None:
            leader_robot.disconnect()
        if follower_robot is not None:
            follower_robot.disconnect()
    except Exception as e:
        print(f"清理异常: {e}")


signal.signal(signal.SIGINT, signal_handler)


# ---------------------------------------------------------------------------
# 创建配置
# ---------------------------------------------------------------------------
def create_arm_config(arm_model, firmware_version, channel):
    platform_system = system()
    if platform_system == "Windows":
        return create_agx_arm_config(
            robot=arm_model,
            firmeware_version=firmware_version,
            interface="agx_cando",
            channel=channel,
        )
    elif platform_system == "Linux":
        return create_agx_arm_config(
            robot=arm_model,
            firmeware_version=firmware_version,
            interface="socketcan",
            channel=channel,
        )
    else:
        raise RuntimeError(f"不支持的操作系统: {platform_system}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    global leader_robot, follower_robot, leader_effector, follower_effector

    # 1. 确定型号
    if args.arm == "piper":
        arm_model = ArmModel.PIPER
        fw_version = PiperFW.DEFAULT
        arm_name = "Piper"
        joint_nums = 6
    else:
        arm_model = ArmModel.NERO
        fw_version = NeroFW.DEFAULT
        arm_name = "Nero"
        joint_nums = 7

    print(f"=== {arm_name} 主从联动（软件转发，含夹爪）===")
    print(f"Leader 通道:   {args.leader_can}")
    print(f"Follower 通道: {args.follower_can}")
    print(f"转发频率:       {args.hz} Hz")
    print(f"末端执行器:    {args.effector}")
    print()

    # 2. 创建配置
    leader_cfg = create_arm_config(arm_model, fw_version, args.leader_can)
    follower_cfg = create_arm_config(arm_model, fw_version, args.follower_can)

    # 3. 创建实例并连接
    leader_robot = AgxArmFactory.create_arm(leader_cfg)
    follower_robot = AgxArmFactory.create_arm(follower_cfg)

    print(f"连接 Leader ({args.leader_can})...")
    leader_robot.connect()
    print(f"连接 Follower ({args.follower_can})...")
    follower_robot.connect()
    print("两臂已连接")
    print()

    # 4. 初始化末端执行器（夹爪）
    if args.effector == "agx_gripper":
        effector_option = leader_robot.OPTIONS.EFFECTOR.AGX_GRIPPER
        print("初始化 Leader 夹爪...")
        leader_effector = leader_robot.init_effector(effector_option)
        print("初始化 Follower 夹爪...")
        follower_effector = follower_robot.init_effector(effector_option)
        print("夹爪已初始化")
    elif args.effector == "revo2":
        effector_option = leader_robot.OPTIONS.EFFECTOR.REVO2
        print("初始化 Leader 夹爪 (Revo2)...")
        leader_effector = leader_robot.init_effector(effector_option)
        print("初始化 Follower 夹爪 (Revo2)...")
        follower_effector = follower_robot.init_effector(effector_option)
        print("夹爪已初始化")
    else:
        print("夹爪跟随已禁用")
    print()

    # 5. 使能
    print("使能 Leader...")
    while not leader_robot.enable():
        time.sleep(0.01)
    print("使能 Follower...")
    while not follower_robot.enable():
        time.sleep(0.01)
    print("两臂已使能")
    print()

    # 6. 设置模式
    print("设置 Leader → 零力拖动模式...")
    leader_robot.set_leader_mode()
    time.sleep(0.5)

    print()
    print("=" * 60)
    print("  主从联动已启动！")
    print(f"  Leader ({args.leader_can})   — 零力拖动，手动拖拽")
    print(f"  Follower ({args.follower_can}) — 实时跟随 Leader")
    if leader_effector is not None:
        print("  夹爪跟随已开启")
    print()
    print("  按 Ctrl+C 安全退出")
    print("=" * 60)
    print()

    # 7. 软件转发循环
    period = 1.0 / args.hz
    last_angles = [0.0] * joint_nums
    last_gripper_value = 0.0
    last_gripper_force = 1.0
    no_data_count = 0
    MAX_NO_DATA = args.hz * 2  # 2 秒无数据则认为异常

    while True:
        loop_start = time.monotonic()

        # 7a. 读取 Leader 关节角度并转发
        joint_msg = leader_robot.get_leader_joint_angles()
        if joint_msg is not None and joint_msg.msg is not None:
            last_angles = joint_msg.msg
            no_data_count = 0
            follower_robot.move_js(last_angles)
        else:
            no_data_count += 1
            if no_data_count > MAX_NO_DATA:
                print(f"[WARN] 已 {MAX_NO_DATA/args.hz:.0f} 秒未收到 Leader 关节数据，停止跟随")
                break

        # 7b. 读取 Leader 夹爪控制状态并转发（仅当夹爪已初始化）
        if leader_effector is not None and follower_effector is not None:
            try:
                gcs = leader_effector.get_gripper_ctrl_states()
                if gcs is not None and gcs.msg is not None:
                    cur_val = gcs.msg.value
                    cur_frc = gcs.msg.force
                    if (abs(cur_val - last_gripper_value) > 0.0001 or
                            abs(cur_frc - last_gripper_force) > 0.01):
                        last_gripper_value = cur_val
                        last_gripper_force = cur_frc
                        follower_effector.move_gripper_m(value=cur_val, force=cur_frc)
            except Exception:
                pass

        # 保持稳定的循环频率
        elapsed = time.monotonic() - loop_start
        time.sleep(max(0, period - elapsed))


if __name__ == "__main__":
    main()

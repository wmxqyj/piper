#!/usr/bin/env python3
"""查看机械臂型号和固件版本"""

import sys
import time
from pyAgxArm import create_agx_arm_config, AgxArmFactory
from pyAgxArm.protocols.can_protocol.msgs.piper.default.transmit.arm_leader_follower_config import (
    ArmMsgLeaderFollowerModeConfig,
)
from pyAgxArm.protocols.can_protocol.msgs.piper.default.transmit.arm_mode_ctrl import (
    ArmMsgModeCtrl,
)
from pyAgxArm.protocols.can_protocol.msgs.piper.default.transmit.arm_motor_enable_disable import (
    ArmMsgMotorEnableDisableConfig,
)

# === 从 verification.yaml 读取参数，或直接写在这里改 ===
CAN_CHANNEL = "can_piper_r"
ARM_MODEL = "piper"
FW_VERSION = "default"  # 先填 default，连接上后再用 get_firmware 确认

# 连接
print(f"连接机械臂: model={ARM_MODEL}, channel={CAN_CHANNEL}")
cfg = create_agx_arm_config(
    robot=ARM_MODEL,
    firmeware_version=FW_VERSION,
    channel=CAN_CHANNEL,
)
robot = AgxArmFactory.create_arm(cfg)
robot.connect()

# 等待反馈帧到达（需要一点时间让 CAN 接收线程积累数据）
time.sleep(0.5)


def _try_recover_from_leader_mode():
    """检测并尝试恢复 leader 模式。

    判断依据：
      1. arm_status（0x2A1）不存在 → 可能处于 leader 模式
      2. get_firmware() 能通信 → CAN 总线正常，确认是模式问题

    恢复策略（已验证的手动恢复序列）：
      0x470#FC → 设为 Follower，退出 leader 模式
      0x471#07 02 → 使能所有关节
      0x151#01 → 切换到 CAN 控制模式

    返回 True 表示恢复成功，False 表示需要断电重启。
    """
    has_arm_status = getattr(robot._parser, "arm_status", None) is not None
    if has_arm_status:
        return True  # 已有状态帧，无需恢复

    print("\n[WARN] 未收到 0x2A1 状态反馈帧")

    # 用 get_firmware() 确认 CAN 通信是否正常
    print("[INFO] 查询固件确认 CAN 通信状态...")
    fw = robot.get_firmware(timeout=1.0)
    if fw is None:
        print("[WARN] 固件查询无回应，请确认 CAN 总线连接和供电")
        return False

    print(f"[INFO] CAN 通信正常 (fw={fw.get('software_version', '?')})，尝试恢复...\n")

    # === 恢复序列（与手动 cansend 验证通过的指令完全一致）===
    steps = [
        ("设为 Follower 退出 leader 模式 (0x470#FC)",
         lambda: robot._send_msg(
             ArmMsgLeaderFollowerModeConfig(linkage_config=0xFC))),
        ("等待 0.2s", lambda: time.sleep(0.2)),
        ("使能所有关节 (0x471#07 02)",
         lambda: robot._send_msg(
             ArmMsgMotorEnableDisableConfig(joint_index=7, enable_flag=2))),
        ("等待 0.3s", lambda: time.sleep(0.3)),
        ("切换到 CAN 控制模式 (0x151#01)",
         lambda: robot._send_msg(
             ArmMsgModeCtrl(ctrl_mode=0x01, move_mode=0x00, move_spd_rate_ctrl=0, mit_mode=0x00))),
        ("等待 0.5s", lambda: time.sleep(0.5)),
    ]

    for desc, fn in steps:
        print(f"  {desc} ...", end=" ", flush=True)
        fn()
        print("done")

    # 检查恢复结果
    has_arm_status = getattr(robot._parser, "arm_status", None) is not None
    if has_arm_status:
        print("\n[OK] 恢复成功！已收到 0x2A1 状态帧")
        return True

    print("\n[FAIL] 恢复失败，机械臂需要断电重启才能退出 Leader 模式")
    print("      请按以下步骤操作：")
    print("        1. 关闭机械臂电源")
    print("        2. 等待 10 秒")
    print("        3. 重新上电")
    print("        4. 重新运行本程序")
    return False


recovered = _try_recover_from_leader_mode()
if not recovered and getattr(robot._parser, "arm_status", None) is None:
    print("[INFO] 0x2A1 不可用，跳过状态依赖操作，继续尝试使能和固件查询...")

# 使能
print("\n使能...")
for _ in range(500):
    if robot.enable():
        print("已使能")
        break
    time.sleep(0.01)
else:
    print("使能失败，继续尝试查询...")

# 查询固件信息
fw = robot.get_firmware(timeout=2.0)
if fw:
    print("\n=== 机械臂信息 ===")
    for key, val in fw.items():
        print(f"  {key}: {val}")
else:
    print("未能获取固件信息")

# 断开
robot.disable()
robot.disconnect()
print("\n已断开连接")

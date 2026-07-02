# Piper 双臂主从数据采集系统

基于 Piper SDK 和 PyRealSense 的主从操作数据采集系统，支持将轨迹数据保存为 RLBench 格式（6关节版本）。

## 系统概述

本系统用于采集 Piper 双臂主从操作的演示数据，适用于机器人学习训练。主要特点：

- **主从控制**：主臂零力拖动示教，从臂实时跟随执行
- **数据采集**：同步采集机械臂状态和相机数据
- **格式适配**：保存为 RLBench 格式（适配6关节机械臂）
- **数据清洗**：支持帧删除和关键帧标记

## 文件结构

```
src/data_collection/
├── README.md                       # 本说明文档
├── cfgs/
│   └── piper_data_collect.yaml    # 配置文件
├── piper_interface.py              # Piper SDK 接口封装
├── camera_interface.py             # PyRealSense 相机接口
├── data_sync.py                    # 数据同步机制
├── rlbench_adapter.py              # RLBench 格式适配器
├── data_collect_piper.py           # 主录制脚本
└── data_cleaner.py                 # 数据清洗工具
```

## 系统架构

```
┌─────────────────────────────────────────────────┐
│          主臂 (Leader)                           │
│          零力拖动示教                             │
│          CAN: can_piper_l                       │
└──────────────┬──────────────────────────────────┘
               │ Piper SDK
               ▼
┌─────────────────────────────────────────────────┐
│          主从控制循环                             │
│          - 读取主臂关节角度                       │
│          - 发送命令给从臂                         │
│          - 同步夹爪状态                           │
└──────────────┬──────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────┐
│          从臂 (Follower)                         │
│          实时跟随执行                             │
│          CAN: can_piper_r                       │
└──────────────┬──────────────────────────────────┘
               │ 状态数据采集
               ▼
┌─────────────────────────────────────────────────┐
│          RGB-D 相机                              │
│          (RealSense)                             │
└──────────────┬──────────────────────────────────┘
               │ 图像数据采集
               ▼
┌─────────────────────────────────────────────────┐
│          数据同步                                 │
│          时间戳同步 (<50ms)                       │
└──────────────┬──────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────┐
│          RLBench 格式转换                         │
│          (6关节版本)                              │
└──────────────┬──────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────┐
│          数据保存                                 │
│          - RGB/Depth 图像                         │
│          - 低维状态 (pickle)                      │
│          - 语言目标                               │
└─────────────────────────────────────────────────┘
```

## 模块说明

### 1. 配置文件 (`cfgs/piper_data_collect.yaml`)

包含所有系统参数：
- 机械臂配置（CAN通道、控制频率）
- 相机配置（分辨率、帧率）
- 录制参数（频率、缓冲区）
- 保存路径和格式

### 2. Piper 接口 (`piper_interface.py`)

**主要功能**：
- 连接主臂和从臂
- 设置主臂为零力拖动模式
- 运行主从控制循环（50Hz）
- 获取从臂完整状态数据

**核心 API**：
```python
# 连接
interface.connect()

# 运行主从循环
interface.run_leader_follower_loop(callback_func=my_callback)

# 获取从臂状态
state = interface.get_follower_state()
# 返回: PiperArmState(joint_positions, joint_velocities, joint_forces,
#                     gripper_pose, gripper_positions, gripper_open)

# 清理资源
interface.cleanup()
```

### 3. 相机接口 (`camera_interface.py`)

**主要功能**：
- 连接 RealSense 相机
- 获取 RGB 和深度图像
- 提取相机内参矩阵
- 深度对齐到彩色图像

**核心 API**：
```python
# 连接
camera.connect()

# 获取数据
data = camera.get_camera_data()
# 返回: CameraData(rgb_image, depth_image, camera_intrinsics, camera_extrinsics)

# 获取内参矩阵
intrinsics = camera.get_intrinsics_matrix()  # 3x3

# 清理资源
camera.cleanup()
```

### 4. 数据同步 (`data_sync.py`)

**主要功能**：
- 同步机械臂和相机数据（基于时间戳）
- 管理帧缓冲区
- 统计录制信息

**核心 API**：
```python
# 创建同步管理器
sync_manager = DataSyncManager(config)

# 同步数据
frame = sync_manager.sync_data(arm_state, camera_data)
# 返回: SyncedFrame(timestamp, joint_positions, ..., front_rgb, front_depth)

# 添加到缓冲区
sync_manager.add_frame_to_buffer(frame)

# 获取统计信息
sync_manager.print_stats()
```

### 5. RLBench 适配器 (`rlbench_adapter.py`)

**主要功能**：
- 转换为 RLBench 格式（6关节版本）
- 欧拉角 → 四元数转换
- 位姿 → 4x4 矩阵转换
- 保存为 pickle 文件

**核心 API**：
```python
# 创建适配器
adapter = RLBenchAdapter(config)

# 转换帧为观测
obs = adapter.convert_frame_to_observation(frame)

# 转换帧列表为 Demo
demo = adapter.convert_frames_to_demo(frames)

# 保存 Demo
adapter.save_demo(demo, save_path, episode_idx)
```

### 6. 主录制脚本 (`data_collect_piper.py`)

**主要功能**：
- 集成所有模块
- 交互式录制控制
- 数据保存

**使用方法**：
```bash
python data_collect_piper.py
```

**交互流程**：
1. 初始化系统
2. 输入任务名称、episode 编号
3. 主从控制启动
4. 输入命令：
   - `s` - 开始录制
   - `q` - 停止录制
   - `y` - 保存数据
   - `n` - 丢弃数据
   - `e` - 退出系统

### 7. 数据清洗工具 (`data_cleaner.py`)

**主要功能**：
- 删除前/后 N 帧
- 交互式标记关键帧
- 自动检测关键帧
- 清理多余图像文件

**使用方法**：
```bash
python data_cleaner.py
```

**清洗流程**：
1. 输入 Episode 路径
2. 选择操作：
   - `1` - 删除帧
   - `2` - 交互式标记关键帧
   - `3` - 自动检测关键帧
   - `4` - 保存清洗数据
   - `0` - 退出

## 数据格式

### Observation 结构（6关节版本）

```python
Observation(
    # 视觉数据
    front_rgb: np.ndarray,          # shape: (H, W, 3), dtype: uint8
    front_depth: np.ndarray,        # shape: (H, W), dtype: uint16
    
    # 机械臂状态（6关节）
    joint_positions: np.ndarray,    # shape: (6,), 单位: rad
    joint_velocities: np.ndarray,   # shape: (6,), 单位: rad/s
    joint_forces: np.ndarray,       # shape: (6,), 单位: N·m
    
    # 末端状态
    gripper_pose: np.ndarray,       # shape: (7,), [x,y,z,qx,qy,qz,qw]
    gripper_matrix: np.ndarray,     # shape: (4, 4)
    gripper_open: float,            # 0.0 或 1.0
    gripper_joint_positions: np.ndarray,  # shape: (2,)
    
    # 相机参数（在 misc 中）
    misc: {
        'front_camera_intrinsics': np.ndarray,  # shape: (3, 3)
        'front_camera_extrinsics': np.ndarray,  # shape: (4, 4)
        'front_camera_near': float,             # 0.5
        'front_camera_far': float,              # 4.5
        'keypoint_idxs': np.ndarray             # 关键帧索引
    }
)
```

### 保存路径结构

```
/home/siat/data/piper_demos/
└── task_name/
    └── all_variations/
        └── episodes/
            └── episode0/
                ├── front_rgb/
                │   ├── 0.png
                │   ├── 1.png
                │   └── ...
                ├── front_depth/
                │   ├── 0.png
                │   ├── 1.png
                │   └── ...
                ├── low_dim_obs.pkl           # Demo 对象
                ├── variation_number.pkl      # variation 编号
                └── variation_descriptions.pkl # 语言目标
```

## 使用示例

### 1. 基本录制流程

```bash
# 1. 启动录制系统
cd src/data_collection
python data_collect_piper.py

# 2. 系统初始化
[1/4] 初始化 Piper 双臂接口...
[2/4] 初始化 RealSense 相机...
[3/4] 初始化数据同步管理器...
[4/4] 初始化 RLBench 格式适配器...

# 3. 选择任务（从配置文件）
可用任务列表:
  [0] pick_place: ['pick up the red block and place it in the box']
  [1] push_button: ['push the blue button']
  [2] slide_block: ['slide the block to the left']
  [3] open_drawer: ['open the drawer']

选择任务:
输入任务编号 (0-3): 0

任务配置:
任务名称: pick_place
语言指令: ['pick up the red block and place it in the box']
起始Episode: 0
Variation: 0

# 4. 开始录制
输入命令: s
录制已开始，拖动主臂进行示教...
当前任务: pick_place | Episode: 0
已录制 50 帧
已录制 100 帧
...

# 5. 停止录制
输入命令: q
停止录制...
缓冲区帧数: 150
录制时长: 3.0s
平均帧率: 50.0fps

# 6. 保存数据（语言指令已从配置文件读取）
保存数据？(y/n): y
保存 Episode...
任务名称: pick_place
Episode: 0
语言指令: ['pick up the red block and place it in the box']
已保存到 /home/siat/data/piper_demos/pick_place/all_variations/episodes/episode0

下一个 Episode 编号: 1

# 7. 继续录制下一个 episode
输入命令: s
录制已开始，拖动主臂进行示教...
当前任务: pick_place | Episode: 1
...
```

### 2. 数据清洗流程

```bash
# 1. 启动清洗工具
cd src/data_collection
python data_cleaner.py

# 2. 输入 Episode 路径
输入 Episode 路径: /home/siat/data/piper_demos/pick_place/all_variations/episodes/episode0

# 3. 显示信息
Episode 信息
总帧数: 150
关键帧索引: []

# 4. 删除帧
选择操作: 1
删除前N帧 (默认0): 10
删除后N帧 (默认0): 5
删除了前 10 帧
删除了后 5 帧
剩余 135 帧

# 5. 自动检测关键帧
选择操作: 3
关节角度变化阈值 (默认0.1 rad): 0.15
自动检测到 5 个关键帧: [20, 45, 72, 100, 128]
是否使用自动检测的关键帧？(y/n): y

# 6. 交互式标记
选择操作: 2
输入命令: m 60
已标记帧 60 为关键帧
输入命令: u 72
已取消帧 72 的关键帧标记
输入命令: q
完成关键帧标记

# 7. 保存
选择操作: 4
输出路径 (默认覆盖原文件): 
已保存，共 135 帧
关键帧索引: [20, 45, 60, 100, 128]

# 8. 退出
选择操作: 0
```

## 配置说明

### 修改配置文件

编辑 `cfgs/piper_data_collect.yaml`：

```yaml
# 机械臂配置
piper:
  leader_can: "can_piper_l"      # 修改为实际的 CAN 通道
  follower_can: "can_piper_r"    # 修改为实际的 CAN 通道
  control_frequency: 50          # 控制频率（Hz）

# 相机配置
camera:
  resolution:
    width: 640                   # 图像宽度
    height: 480                  # 图像高度
  fps: 30                        # 帧率

# 任务配置列表（重点：包含任务名称和语言指令）
tasks:
  - name: "pick_place"           # 任务名称
    descriptions: ["pick up the red block and place it in the box"]  # 语言指令
    episode_start: 0             # 起始 episode 编号
    
  - name: "push_button"
    descriptions: ["push the blue button"]
    episode_start: 0
    
  # 添加更多任务...
  - name: "custom_task"
    descriptions: ["your custom language instruction"]
    episode_start: 10            # 可以从任意编号开始

# 数据保存配置
demo:
  save_path: "/home/siat/data/piper_demos"  # 保存路径
  variation: 0                              # variation 编号
```

**任务配置说明**：
- `name`: 任务名称，用于创建保存目录
- `descriptions`: 语言指令列表，保存时会自动使用
- `episode_start`: 起始 episode 编号，每次保存后自动递增

**添加新任务**：
只需在 `tasks` 列表中添加新的配置项即可，无需修改代码。

### 环境要求

**硬件**：
- 两个 Piper 机械臂（通过 CAN 总线连接）
- AgxGripper 夹爪（可选）
- RealSense RGB-D 相机

**软件**：
- Python 3.8+
- pyAgxArm SDK
- pyrealsense2
- numpy
- opencv-python
- pyyaml

**CAN 总线配置**（Linux）：
```bash
sudo ip link set can_piper_l up type can bitrate 1000000
sudo ip link set can_piper_r up type can bitrate 1000000
```

## 常见问题

### Q1: 如何添加新的任务？

A: 在配置文件 `cfgs/piper_data_collect.yaml` 中的 `tasks` 列表添加新配置：

```yaml
tasks:
  - name: "new_task"
    descriptions: ["your new language instruction"]
    episode_start: 0
```

保存后重启录制系统，新任务就会出现在任务列表中。

### Q2: 相机连接失败怎么办？

A: 检查相机是否正确连接，系统会继续运行但不会记录相机数据。也可以修改配置禁用相机：

```yaml
camera:
  type: "none"  # 禁用相机
```

### Q3: 主臂无法拖动？

A: 确认主臂已进入零力拖动模式，检查 CAN 总线是否正确配置。

### Q4: 数据同步失败？

A: 检查时间戳差异，默认容忍 50ms。可以修改配置：

```yaml
recording:
  max_time_diff: 0.1  # 改为 100ms
```

### Q5: 如何修改关节数量？

A: 修改配置文件中的 `joint_nums`，但需要同步修改 RLBench 适配器代码。

### Q6: 如何从特定的 episode 编号开始录制？

A: 在任务配置中设置 `episode_start`：

```yaml
tasks:
  - name: "pick_place"
    episode_start: 10  # 从 episode10 开始录制
```

### Q7: 关键帧检测不准确？

A: 调整自动检测的阈值：
```python
keypoints = cleaner.auto_detect_keypoints(threshold=0.2)  # 更大的阈值
```

## 性能优化

### 提高控制频率

```yaml
piper:
  control_frequency: 100  # 提高到 100Hz
```

### 降低图像分辨率

```yaml
camera:
  resolution:
    width: 320
    height: 240
  fps: 15
```

### 使用多线程

录制脚本已经使用多线程：
- 主线程：用户交互
- 子线程：主从控制循环

## 测试方法

### 测试 Piper 接口

```bash
python piper_interface.py
```

### 测试相机接口

```bash
python camera_interface.py
```

### 测试完整系统

```bash
python data_collect_piper.py
```

建议在安全环境下先测试基本功能，确认无误后再进行正式录制。

## 安全注意事项

1. **启动前检查**：确保机械臂周围无障碍物
2. **运动范围限制**：注意不要超出机械臂的工作空间
3. **夹爪操作**：小心夹爪的开合动作
4. **紧急停止**：随时准备按 Ctrl+C 安全退出
5. **数据备份**：定期备份录制的数据

## 开发说明

### 扩展功能

**添加新的相机类型**：
```python
# 在 camera_interface.py 中添加新的相机类
class CustomCameraInterface:
    def connect(self): ...
    def get_camera_data(self): ...
```

**修改数据格式**：
```python
# 在 rlbench_adapter.py 中修改 Observation 结构
@dataclass
class CustomObservation:
    # 添加新的字段
    ...
```

### 调试技巧

**打印详细日志**：
```python
# 在配置文件中启用详细日志
piper:
  log_level: "DEBUG"
```

**单步调试**：
```python
# 在主脚本中设置断点
import pdb; pdb.set_trace()
```

## 参考资料

- [pyAgxArm SDK 文档](../../pyAgxArm/)
- [RLBench 论文](https://arxiv.org/abs/1909.12271)
- [RealSense SDK](https://github.com/IntelRealSense/librealsense)

## 更新日志

### v1.0.0 (2026-07-02)
- 完整的主从数据采集系统
- RLBench 格式适配（6关节版本）
- 数据清洗工具
- 交互式录制控制

## 联系方式

如有问题或建议，请联系开发团队。

---

**Happy Data Collection! 🎉**
# Neural-Fly 控制系统复现框架

本仓库基于 ROS + MuJoCo 构建了复现 Neural-Fly / DAIML 思路的整体控制系统框架。目标是完成从离线表征到在线适配的闭环验证，当前仓库重点实现了可在 MuJoCo 仿真中运行的基础控制与调试工具。

## 仓库结构

```
.
├─ sim/                         # MuJoCo 场景与 ROS 接口
│  ├─ quad_ros.py               # 仿真主进程：接入 ROS 话题、执行器控制
│  └─ scene/                    # MuJoCo XML 场景（含四旋翼机械臂、风场配置等）
├─ uam_controller/              # ROS 控制器功能包
│  ├─ cfg/ControllerConfig.cfg  # 动态参数配置（PD/PID、补偿开关等）
│  ├─ config/
│  │  ├─ controller_params.yaml # 控制器默认参数、惯量、旋翼方向等
│  │  └─ uam_config.yaml        # 从仿真场景同步的结构化参数
│  ├─ include/uam_controller/
│  │  ├─ controller.hpp         # 控制节点类声明
│  │  ├─ debug_macros.hpp       # 调试宏、节流与话题发布工具
│  │  └─ log_publisher.hpp      # 日志话题发布器封装
│  ├─ scripts/control_monitor.py# Matplotlib 实时监控台
│  ├─ src/
│  │  ├─ controller_node.cpp    # 位置 PD + 姿态 PID + 扰动前馈核心闭环
│  │  └─ setpoint_publisher.cpp # 简易参考轨迹发布节点
│  └─ launch/uam_controller.launch
│                             # 启动控制器、参考轨迹、rqt_reconfigure、监控台
└─ readme.md                   # 当前说明文档
```

## 功能概述

- **仿真桥接**：`sim/quad_ros.py` 将 MuJoCo 场景与 ROS 话题互通，接收电机指令、发布里程计、关节状态、轨迹等信息。
- **基础控制器**：`uam_controller` 包实现了位置 PD + 姿态 PID 的双层控制，同时保留扰动前馈接口，为后续接入 Neural-Fly 的线性适配头做准备。
- **调试体系**：通过自定义调试宏与日志发布器，将姿态误差、角加速度指令、扭矩指令、混合后推力、油门等信息发布到 `/uam_controller/debug/*` 话题，可直接在 rqt_plot 或监控台查看。
- **图形监控台**：`scripts/control_monitor.py` 订阅调试话题并以 Matplotlib 实时绘制位置、速度、误差、电机油门等曲线。
- **数据采集/训练模块**：`neural-fly/ros_nodes/data_collector.py`、`neural-fly/train_phi.py`、`neural-fly/eval_phi.py` 提供从仿真话题采数 → 训练 φ_net → 评估的完整流水线。
- **在线扰动估计**：`uam_controller/scripts/rls_adapter.py` 加载训练好的 φ_net，执行 RLS 更新并实时发布扰动补偿量，控制器读取后进行前馈抵消。
- **动态调参**：集成 `dynamic_reconfigure`，可在运行中调整位置/姿态 PID、扰动补偿开关、油门上下限等参数。

## 快速开始

1. **依赖环境**
   - ROS Noetic（或同等 ROS1 环境）
   - MuJoCo 2.x 及 Python API（`mujoco` 与 `mujoco-python-viewer`）
   - Python 依赖：`numpy`、`matplotlib`、`rospy`、`tf2_ros` 等（catkin 自动处理大部分 ROS 依赖）

2. **构建与编译**
   ```bash
   cd ~/Neural_Fly_ws
   catkin build uam_controller
   source devel/setup.bash
   ```

3. **启动仿真与控制器**
   ```bash
   roslaunch uam_controller uam_controller.launch
   ```
   - 默认会启动：控制器节点、参考轨迹发布节点、rqt_reconfigure、控制监控台。
   - 如无图形界面，可通过 `use_monitor:=false` 禁用监控台。

4. **调试与可视化**
   - 运行过程中，可在 rqt_reconfigure 调整 PID、扰动补偿等参数。
   - 调试话题示例：
     - `/uam_controller/debug/position`
     - `/uam_controller/debug/attitude_error`
     - `/uam_controller/debug/ang_acc_cmd`
     - `/uam_controller/debug/torque_cmd`
     - `/uam_controller/debug/throttle_0~3`
   - `control_monitor.py` 将上述话题绘制为六个实时子图。

## 数据采集 → 训练 → 在线适配

1. **采集数据**
   ```bash
   # 先 source ROS 工作空间，再激活 conda 环境 (含 torch 等)
   conda activate aloha
   python neural-fly/ros_nodes/data_collector.py \
       --log-directory $(pwd)/neural-fly/data/sim_log \
       --vehicle sim --trajectory random3 \
       --method NF-sim --condition 40wind
   # 另开终端启动采集
   rosservice call /neural_fly_data_collector/start "data: true"
   # 完成后停止并保存
   rosservice call /neural_fly_data_collector/stop_and_save
   ```
   - 节点自动根据里程计、motor_cmd 计算 `fa`，输出 csv 文件可直接喂入训练脚本。

2. **训练 φ_net**
   ```bash
   conda activate aloha
   cd neural-fly
   python train_phi.py --data-folder data/sim_log --model-name phi_sim --epochs 80 \
       --support-size 32 --query-size 128 --latent-dim 32
   ```
   训练结束模型存放于 `neural-fly/models/phi_sim.pth`。

3. **评估**
   ```bash
   conda activate aloha
   python eval_phi.py --model-name phi_sim --data-folder data/sim_log
   ```

4. **在线扰动补偿**
   ```bash
   conda activate aloha
   python neural-fly/ros_nodes/rls_adapter.py \
       --ros-args --param model_name:=phi_sim --param model_folder:=neural-fly/models
   ```
   节点订阅 `/quad/odometry` 与 `/quad/motor_cmd`，在 `/quad/disturbance_estimate` 发布残差力，控制器会自动读取并做前馈补偿。

## 常见参数说明

- **旋翼配置**：默认从 `config/controller_params.yaml` 和 `/uam_config/rotor_configuration` 读取，保证与 MuJoCo 场景一致。若修改 XML 中电机臂长/方向，请同步更新 YAML 或 `uam_config.yaml`。
- **惯量参数**：在 `controller_params.yaml` 中配置 `inertia_xx/yy/zz`，控制器将 PID 输出视为角加速度指令，并自动乘以惯量得到扭矩。
- **调试等级**：通过编译选项 `-DUAM_DEBUG_LEVEL={0,1,2}` 控制日志与话题输出范围；LEVEL ≥2 时才会发布调试话题与采集/在线节点所需的关键数据。

## 后续规划

- 接入 Neural-Fly 离线表征模块，加载 φ(x) 并实现 RLS/复合自适应在线头。
- 在仿真中构建多风场、多轨迹采集流程，实现真实数据驱动的离线训练。
- 扩展监控台，支持更丰富的状态、误差与域间对比分析。

## 参考

- Neural-Fly: https://arxiv.org/abs/2203.09452
- DAIML 原官方代码：请参阅仓库 `neural-fly`（需自行克隆）

如有问题或建议，欢迎提交 issue，共同完善该控制系统复现框架。

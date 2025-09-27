# 🛠️ Imitation Learning for Flying Gripper in MuJoCo (dm_control)

本仓库基于 ACT 和自定义的 MPC 控制器，训练带机械臂的飞行器完成抓取任务。训练后的策略可用于实时控制仿真器或实物系统。
⚠️ 使用前请确保已编译并运行 `ocs2_uam_ros` 功能包。

---

## 📁 项目结构说明

```
.
├── ckpts/                    # 模型权重和归一化统计数据
├── data/                     # 采集的训练数据集
├── scene/                    # 仿真所用的 MuJoCo 场景 XML 文件
├── __pycache__/              # Python 缓存目录
├── constants.py              # 常量定义
├── detr/                     # 模型结构定义模块（含DETR、Transformer等）
├── ee_sim.py                 # 单独运行的 dm_control 模拟器入口（用于调试环境）
├── ee_sim_env.py             # 包装环境的类（用于数据采集/训练）
├── imitate_episodes.py       # 模仿学习训练与评估入口
├── mpc_server.py             # 启动与 MPC 控制器通信的服务端
├── policy.py                 # ACT / CNNMLP 等策略网络定义
├── quad_ros.py               # 原生 MuJoCo 搭建的无人机抓取器仿真环境（非 dm_control）
├── readme.md                 # 本说明文档
├── record_sim_episodes.py    # 执行数据采集脚本
├── run_proxy_controller.py   # 多线程部署推理控制器（仿真推步 + 网络推理）
├── theta_draw.py             # 可视化机械臂角度随时间变化的辅助工具
├── utils.py                  # 工具函数（如图像处理、观测处理、数据读取）
├── visualize_episodes.py     # 可视化采集数据集中某一集的图像和动作序列
```

---

## 🧠 模仿学习训练流程

以下流程涵盖从仿真环境部署、数据采集，到策略训练与评估。

> 💡 **提示**：运行任意 Python 脚本前，请先执行 `conda activate aloha`，以确保依赖正确加载。

### 1. 进入 Python 环境

```bash
conda create -n aloha python=3.8.10
conda activate aloha
pip install torchvision
pip install torch
pip install pyquaternion
pip install pyyaml
pip install rospkg
pip install pexpect
pip install mujoco==3.2.1
pip install dm_control==1.0.22
pip install opencv-python
pip install matplotlib
pip install einops
pip install packaging
pip install h5py
pip install ipython
cd /detr && pip install -e .
```

---

### 2. 启动 MPC 控制器（用于数据采集）

```bash
# 启动 MPC 服务端
python mpc_server.py

# 启动 ROS 控制器发布器
rosrun ocs2_uam_ros uam_mpc_publisher_seq_point
```

---

### 3. 采集模仿数据（由 MPC 控制器驱动）

```bash
python record_sim_episodes.py --task_name sim_uam_grasp_cube_scripted --onscreen_render
```

执行完后可关闭MPC服务器与控制发布ROS节点

---

### 4. 查看采集数据

```bash
python3 visualize_episodes.py --dataset_dir ./data --episode_idx 0
```

---

### 5. 执行策略训练与评估（支持 ACT / CNNMLP）

```bash
# 仅训练
python3 imitate_episodes.py \
  --task_name sim_uam_grasp_cube_scripted \
  --ckpt_dir ./ckpts \
  --policy_class ACT \
  --batch_size 4 \
  --seed 0 \
  --num_epochs 400 \
  --lr 1e-5 \
  --kl_weight 6 \
  --chunk_size 50 \
  --hidden_dim 128 \
  --dim_feedforward 800

# 仅评估
python3 imitate_episodes.py \
  --task_name sim_uam_grasp_cube_scripted \
  --ckpt_dir ./ckpts \
  --policy_class ACT \
  --batch_size 4 \
  --seed 0 \
  --num_epochs 500 \
  --lr 1e-5 \
  --kl_weight 10 \
  --chunk_size 50 \
  --hidden_dim 128 \
  --dim_feedforward 800 \
  --onscreen_render \
  --eval

```

---

### 7. 推理控制（部署训练好的策略）

```bash
# 启动 MPC 服务端
python mpc_server.py

# 启动 ROS 控制器接收器
rosrun ocs2_uam_ros uam_ee_sub


```

```bash
python3 run_proxy_controller.py \
  --policy_class ACT \
  --ckpt_dir ./ckpts \
  --task_name sim_uam_grasp_cube_scripted \
  --seed 0 \
  --num_epochs 100 \
  --onscreen_render
```

---

## 📌 其他脚本说明

- `quad_ros.py`基于原生 MuJoCo Python 接口搭建的无人机 + 机械臂仿真器，非 dm_control 框架，适用于其他实验需求。
- `theta_draw.py`用于可视化机械手的关节角度（如通过 MPC 控制器执行后记录的数据），帮助分析策略行为和控制曲线。
- `ee_sim.py`
  基于dm_control的仿真器，用于调试数据采集仿真环境。

```bash
python ee_sim.py ./scene/scene_quad_with_gripper.xml --camera_id 0
```

---

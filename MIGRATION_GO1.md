# Go1 / Isaac Sim 6.1 / Isaac Lab 3.0 迁移记录

## 环境与安装

2026-10-21 在本机验证，开发分支 `feat/go1-isaacsim61`，起点 `c3fbd01`。
开始时 master 与 origin/master 一致，工作区干净；没有重置历史或推送远程。

| 组件 | 实际版本 / 位置 |
| --- | --- |
| Conda | `/home/sjc/anaconda3/envs/himloco_lab` |
| Python | 3.12.15 |
| PyTorch | 2.11.0+cu128，CUDA runtime 12.8 |
| NVIDIA | RTX 5070 Ti 16GB，驱动 595.91.07 |
| Isaac Sim | pip 包 6.1.0.0，位于上述环境的 `lib/python3.12/site-packages/isaacsim` |
| Isaac Lab | 源码 VERSION 为 3.0.0；distribution metadata 为 17.0.2 |
| Lab 安装方式 | editable，`/home/sjc/RL_Research/IsaacLab-3.0/source/isaaclab` |
| 物理 | `isaaclab_physx.physics.PhysxCfg`，Isaac Sim PhysX GPU |
| 导出验证 | ONNX Runtime 1.31.0，CPUExecutionProvider |

AppLauncher 已在该环境成功初始化并实际运行 PhysX 场景，未更换仿真环境、未修改官方源码。
终端初始没有激活 Conda，验证命令使用现有环境的 Python 绝对路径。
沙箱内无法访问 GPU；需要正常本机终端或允许 GPU 访问的执行环境。

```bash
source /home/sjc/anaconda3/etc/profile.d/conda.sh
conda activate himloco_lab
cd /home/sjc/himloco_lab
python -m pip install -e 'source/himloco_lab[validation]' --no-build-isolation
```

核心额外依赖为 psutil、tensorboard、onnx；validation extra 增加 onnxruntime。
Isaac Sim/Lab 及其既有依赖继续使用当前 Conda 环境。W&B 为第二轮新增的可选依赖，默认仍使用 TensorBoard。
首次启动会从 NVIDIA 官方资源服务获取 USD、mesh 和 Actuator Net，并利用本机 `/tmp/https` 缓存。

## 修改文件

下表路径省略 `source/himloco_lab/himloco_lab/` 前缀（scripts/tests/根目录文件除外）。

| 文件 | 修改内容 |
| --- | --- |
| `tasks/locomotion/velocity_env_cfg.py` | 提取原 Go2 的共享 manager 配置、reward 和粗糙地形；迁移 PhysX、UniformNoiseCfg；开启 final_obs |
| `tasks/locomotion/robots/go2/velocity_env_cfg.py` | 基于共享配置装配原 Go2 自定义 URDF、执行器和 Play 设置 |
| `tasks/locomotion/robots/go1/__init__.py` | 注册 Unitree-Go1-Velocity / Unitree-Go1-Velocity-Play |
| `tasks/locomotion/robots/go1/velocity_env_cfg.py` | 官方 Go1 配置、关节顺序、trunk、传感器、随机化与地形适配 |
| `tasks/locomotion/agents/himloco_rsl_rl_cfg.py` | 新增 Go1 Runner 默认配置，10000 iterations |
| `tasks/locomotion/mdp/__init__.py` | 迁移至 `isaaclab_tasks.core.velocity.mdp` |
| `tasks/locomotion/mdp/observations.py` | ProxyArray 的 `.torch` 与新 wrench composer 接口 |
| `tasks/locomotion/mdp/rewards.py` | articulation、ray、contact 数据的 `.torch`；新版 contact normal force 名称 |
| `tasks/locomotion/mdp/events.py` | 使用新 wrench 的 index 接口 |
| `tasks/locomotion/mdp/commands/commands.py` | heading 数据取 `.torch` |
| `envs/him_manager_based_rl_env.py` | 委托原生 step；仅维护 smoothness 所需的两步前动作和 UI reset 的 terminal obs |
| `rsl_rl/wrappers/history.py` | 当前帧优先的无原地修改历史函数 |
| `rsl_rl/wrappers/himloco_vec_env_wrapper.py` | 5→7 返回值、final_obs、history reset、有限时域处理和 shape/finite 检查 |
| `rsl_rl/modules/him_estimator.py` | 输入维度、监督切片、有限值检查和注释 |
| `rsl_rl/algorithms/him_ppo.py` | terminal value bootstrap、平均 estimation/swap loss，修正 test_mode |
| `rsl_rl/runners/him_on_policy_runner.py` | 完成迭代计数、两个 optimizer 恢复、恢复学习率、TensorBoard flush |
| `utils/export_policy.py` | 从网络读取单帧维度；兼容 Torch 2.11 的显式 legacy ONNX exporter |
| `scripts/himloco_rsl_rl/train.py` | headless 参数、地形 seed、checkpoint 路径、按 robot metadata 生成 Go2 deploy cfg |
| `scripts/himloco_rsl_rl/play.py` | 默认 Kit GUI、camera 跟随机身、最大步数、速度指令、原生录像配置 |
| `scripts/himloco_rsl_rl/play_interactive.py` | GUI/headless、环境数量与新版录像配置 |
| `scripts/himloco_rsl_rl/play_onnx.py` | GUI/headless 与新版录像配置 |
| `scripts/himloco_rsl_rl/validate_go1.py` | 可重复的算法和真实 PhysX 集成验证 |
| `tests/test_him_contracts.py` | 历史、四元数、目标切片、梯度隔离、超时、loss 平均、混合终止回归 |
| `source/himloco_lab/setup.py` | 依赖和 Python/Sim 版本声明，validation extra |
| `.gitignore` | 允许跟踪迁移测试 |
| `README.md` / `README_CN.md` | 链接 Go1 迁移与启动说明 |

## 机器人、动作与场景

直接复用 `isaaclab_assets.robots.unitree.UNITREE_GO1_CFG`，包括官方
`Robots/Unitree/Go1/go1.usd` 和 `ActuatorNets/Unitree/unitree_go1.pt`。
实际执行器类型验证为 `ActuatorNetMLP`，并在真实 PPO rollout 中执行。
`sim.use_newton_actuators=False` 明确使用 Isaac Lab 的显式 Actuator Net 路径。
官方网络的 pos_scale=-1.0、vel_scale=1.0、torque_scale=1.0；不要额外反转关节符号。
髋/大腿力矩上限为 23.7 Nm，小腿 35.55 Nm；速度上限分别为 30.1/20.06 rad/s，沿用本地官方配置。

动作和关节观测使用以下公开顺序，并在配置中显式固定；这是 PhysX/USD 顺序，不能直接当作 Unitree SDK 顺序：

```text
FL_hip_joint, FR_hip_joint, RL_hip_joint, RR_hip_joint,
FL_thigh_joint, FR_thigh_joint, RL_thigh_joint, RR_thigh_joint,
FL_calf_joint, FR_calf_joint, RL_calf_joint, RR_calf_joint
```

默认关节位置依次为 `[0.1,-0.1,0.1,-0.1, 0.8,0.8,1.0,1.0, -1.5,-1.5,-1.5,-1.5]`。
动作是默认站姿上的关节位置 offset，scale=0.25；没有额外符号转换。
初始根高度为官方的 0.4m，站立机身高度奖励目标为 0.30m。
官方模型关闭 self collisions，开启重力与接触传感器。

实际刚体：`trunk`，四个 hip、thigh、calf、foot，共 17 个。
两个 ray caster、基座质量/COM/扰动力/推力及接触终止均引用 `trunk`。
Contact Sensor 每 0.005s 更新，保留 3 帧 history 和 air time；两个 height scanner 每 0.02s 更新。
187 点 height scanner 为 17×11，使用 yaw alignment；20m 的偏移用于射线起点，`data.pos_w` 仍为附着刚体位置，已检查。
地面摩擦 static/dynamic=1；启动时各刚体随机摩擦 0.2–1.25，restitution=0。
质量随机化 -1~+2kg，COM 范围改为各轴 ±0.03m；周期扰动力和 16s 推扰保留。

保留坡面、带噪坡面、正/反楼梯和离散障碍物；Go1 楼梯高度 0.04–0.18m，带噪坡面振幅 0.01–0.05m，障碍 0.03–0.12m，初始 level≤3。
保留原 rough reward，包括速度跟踪、机身速度/姿态/高度、关节加速度、能量、foot clearance、action rate 和 smoothness。
新版 ContactSensor 报告量使用 `net_normal_forces_w`；原法向接触判定保持。未新增研究奖励。

## 算法与数据对照

对照本机 `/home/sjc/HIMLoco-Go1`，HEAD `7bc9762a4bb56426f4fee8bfcdb2db58b40211a5`，及
[原 HIMLoco](https://github.com/InternRobotics/HIMLoco)、[上游 HIMLocoLab](https://github.com/IsaacZH/himloco_lab)。

- Estimator 原主要源码与 Go1 baseline 相同；新增检查以外保持 encoder、target、prototype、Sinkhorn、estimation loss、swap loss 与独立 Adam 更新。
- Actor 由 45D 当前帧 + 3D 速度估计 + 16D latent 输入；Actor/Critic 隐藏层为 [512,256,128]。
- Actor PPO 路径在 no_grad 下取得 Estimator 输出；PPO 反传不会更新 Estimator，已用梯度测试确认。
- RolloutStorage 的核心逻辑与 baseline 相同（原差别仅一个未使用 import），本次没有替换 storage。
- PPO 保留 clipped surrogate、clipped value loss、entropy、KL 自适应学习率及梯度裁剪。
- 返回本次更新所有 minibatch 的平均 estimation/swap loss。
- 时间截断使用 **终止前 next critic observation 的 value**，对齐目前 Go1 baseline 的工程修复。物理失败不 bootstrap；同时 terminated/truncated 时失败优先；有限时域不 bootstrap。
- Go1 baseline 目前额外包含可选 estimator freeze、学习率上下限参数及更多诊断统计；本项目保持原 Lab 参数集和 1e-5~1e-2 自适应范围，Estimator 始终独立更新。
- Lab 原有 Actor NaN 检查保留。使用 Manager-Based rough 物理、官方 Actuator Net，区别于旧 Gym PD baseline；不宣称动力学或已有 checkpoint 可直接互换。

单帧观测布局：command[0:3]、ang_vel×0.25[3:6]、projected_gravity[6:9]、joint_pos_rel[9:21]、joint_vel×0.05[21:33]、last_action[33:45]。
Critic 为 `[policy结构45D, base_lin_vel×2 的3D, disturbance3D, height_scan×5 的187D]`，共 238D。
保留 Lab 的 command 原始缩放；旧 Gym baseline 对 command 使用其 commands_scale。
训练 actor 保留噪声，Go1 critic 使用无噪声监督，因此前45维语义和顺序相同，数值可能因 actor 噪声不同；禁用噪声时已验证两者相等。
Estimator velocity 监督为 critic[45:48]，target 网络输入为 critic[3:48]（跳过 command，包含下一时刻速度），保持原算法设计。

history_length=5 表示当前帧+5个过去帧，共270D；最新帧在最前。
首次/reset 时只有当前帧有效，其余为零；终止时每个环境清空历史，不跨 episode。
原生 `compute_final_obs=True` 只在需要 reset 时额外计算终止前观测。
wrapper 依据本步 done IDs 读取 final_obs，不会把 reset 状态作为 Estimator 或 timeout 的目标；环境每步清理残留 extras。
物理步进、事件、manager reset 均由本地 Isaac Lab 执行。
所有自定义四元数运算直接遵循本地 XYZW；没有人工重排 root_quat_w，测试覆盖 identity、pitch/yaw 90° 和真实 projected_gravity。

## 训练、恢复、Play、导出

完整训练（4096 环境，100 步/rollout，5 epochs，4 minibatches，10000 次更新）：

```bash
python scripts/himloco_rsl_rl/train.py \
  --task Unitree-Go1-Velocity --headless --num_envs 4096 \
  --seed 1 --max_iterations 10000 --run_name go1_baseline
```

sim.dt=0.005，decimation=4，策略50Hz；learning_rate=1e-3，gamma=.99，lam=.95，entropy=.01，clip=.2，schedule=adaptive，desired_kl=.01。
默认 TensorBoard；每100次迭代及训练结束保存 checkpoint，文件名为**已完成更新数**。
checkpoint 包含完整 Actor/Critic/Estimator、PPO optimizer、Estimator optimizer、迭代号。
恢复时优化器和学习率均恢复；环境从新 episode 开始，不声称恢复物理状态/随机数流的逐步复现。

```bash
python scripts/himloco_rsl_rl/train.py \
  --task Unitree-Go1-Velocity --headless --num_envs 4096 --seed 1 \
  --resume --checkpoint logs/himloco_rsl_rl/go1_rough/2026-10-21_23-36-56_smoke4096/model_3.pt \
  --max_iterations 10000 --run_name go1_resume
```

`--max_iterations` 表示本次额外执行的更新数；恢复时根据剩余目标填写。
也可用 `--load_run RUN_DIRECTORY --checkpoint 'model_.*.pt'` 选择最新 checkpoint。

```bash
# GUI；不加 --headless。支持任意固定 [vx,vy,wz] 指令。
python scripts/himloco_rsl_rl/play.py \
  --task Unitree-Go1-Velocity-Play --num_envs 4 \
  --checkpoint logs/himloco_rsl_rl/go1_rough/2026-10-21_23-36-56_smoke4096/model_3.pt \
  --velocity 0.7 0 0.3 --real-time

# 自动选择最新训练目录的最新 checkpoint；限定步数可用于自动检查。
python scripts/himloco_rsl_rl/play.py \
  --task Unitree-Go1-Velocity-Play --headless --num_envs 4 --max_steps 200

# 原键盘交互入口（GUI）。
python scripts/himloco_rsl_rl/play_interactive.py \
  --task Unitree-Go1-Velocity-Play --num_envs 1

tensorboard --logdir logs/himloco_rsl_rl/go1_rough
```

Actor 推理只接受270D history，不接收 privileged observation。
第二轮将Play改为默认不导出，可通过`--export jit|onnx|all`显式启用；导出到checkpoint同目录下的`exported/`：`policy.pt` 为 Encoder+Actor，ONNX 分为 `encoder.onnx`（270→19，latent已归一化）和 `policy.onnx`（64→12）。
第二轮ONNX已支持动态batch，独立验证覆盖batch1/7/64及真实rollout。
Go1 不生成依赖 `joint_sdk_names` 的 Go2 实机部署配置；Go2 的部署生成入口保留。

## 实际测试与结果

完整数据/算法/导出验证：

```bash
python scripts/himloco_rsl_rl/validate_go1.py --headless --num_envs 32
```

7 项回归测试及完整集成验证已通过，结果保存在 `logs/go1_validation/validation.json`，包括关节/刚体列表、实际参数变化、JIT/ONNX 误差。
验证强制超时前 last_action=0.137、reset 后 last_action=0，确认 terminal critic 来自 reset 前；检查270/238/12形状、真实关节/姿态/传感器、历史清零、PPO和Estimator更新、两个优化器保存/恢复及逐值相同推理。

本次32环境验证的 Actor/Estimator首层权重最大变化分别为0.0044611/0.0045707；checkpoint重载推理逐值相同。JIT与ONNX最大误差分别为4.47e-8/5.96e-8。

实际命令与训练结果：

| 测试 | 参数 | 结果 |
| --- | --- | --- |
| 32 smoke | `train.py --task Unitree-Go1-Velocity --headless --num_envs 32 --max_iterations 1 --run_name smoke32` | 100步rollout+1次update，成功保存 |
| 4096短训 | `train.py --task Unitree-Go1-Velocity --headless --num_envs 4096 --seed 1 --max_iterations 3 --run_name smoke4096` | 1,228,800 transitions，3次update，无OOM/NaN/Inf |
| 4096恢复 | 上述恢复命令的 `--max_iterations 1 --run_name resume4096` | model_3→model_4，两个optimizer成功恢复后更新 |
| GUI Play | 上述GUI命令加 `--max_steps 200` | 加载model_3，完成200步，正常退出并导出 |
| headless Play | `play.py --task Unitree-Go1-Velocity-Play --headless --num_envs 4 --load_run 2026-10-21_23-38-08_resume4096 --max_steps 100 --velocity 0.5 0 0.2` | 自动选中model_4，完成100步并导出 |

4096 三次更新的 `(value,surrogate,estimation,swap)`：

```text
0: (0.3483,  0.0033, 0.2211, 0.1079)
1: (0.0604, -0.0055, 0.0877, 0.1055)
2: (0.0319, -0.0060, 0.0705, 0.1029)
```

rollout+更新耗时分别约6.37、5.60、5.69秒（不含场景启动），64k~73k transitions/s。
TensorBoard event 文件已实际读取，含 Reward、Value/Surrogate/Estimation/Swap Loss 和 Episode Length。
恢复后的 checkpoint 对比确认 Actor、Critic、Estimator encoder/target/prototype 权重均发生变化；PPO optimizer含17组参数状态，Estimator optimizer含13组参数状态。

本机原始日志：`/tmp/himloco_smoke32.log`、`/tmp/himloco_smoke4096.log`、`/tmp/himloco_resume4096.log`、`/tmp/himloco_validation.log`、`/tmp/himloco_play_gui.log`、`/tmp/himloco_play_headless.log`。
checkpoint与TensorBoard保存在 `logs/himloco_rsl_rl/go1_rough/2026-10-21_23-36-56_smoke4096/` 和 `2026-10-21_23-38-08_resume4096/`。这些训练产物由gitignore排除。

## 边界与后续工作

短程训练验证的是工程和数据链路，**没有证明运动策略已收敛或具备高质量速度跟踪**。
下一步运行完整训练并评估不同地形、速度、随机 seed 的长期 reward、跌倒率及跟踪误差。
未进行Go1 MuJoCo、SDK、真机或新算法开发。W&B离线日志已在第二轮完成验证。
Go2 配置保留并检查可实例化；其 URDF 路径仍需使用者在 `assets/unitree.py` 指向本机模型，未在本次完成 Go2 GPU训练回归。
录像配置已迁移至原生 VideoRecorderCfg；键盘/录像/独立ONNX Play 不在主要验收路径，需在使用前单独验证。
Isaac Sim 启动会报告 CPU package、MaterialX/USD 等 warning；真实训练和 Play 已完成。未对第三方核心库作屏蔽修复。

未来 sim2sim/sim2real 应特别核对：公开关节顺序到 SDK/URDF 的映射、rad与力矩单位、Actuator Net 的误差符号及其3个物理步历史、部署50Hz策略与200Hz执行器调用、270D history 顺序与reset、观测缩放、latent L2归一化、默认站姿和0.25 action scale、传感器/坐标系/XYZW约定（本项目为 XYZW，标量在末位）、力矩/速度/碰撞限制，以及部署端是否具备同等执行器模型。
不得直接把旧 Isaac Gym checkpoint 当作该 Lab 场景的可互换策略。


## 第二轮：旧权重、独立导出与W&B（2026-10-22）

- [LEGACY_POLICY_EVAL.md](LEGACY_POLICY_EVAL.md)：旧model_7500严格加载、真实Gym DOF查询、direct/adapted接口、20秒平地/rough与GUI结果，区分网络/语义/物理差异。
- [EXPORT_POLICY.md](EXPORT_POLICY.md)：新model_4与旧model_7500的独立CPU JIT/split ONNX、多batch和真实rollout验收，Play不再强制导出。
- [WANDB_LOGGING.md](WANDB_LOGGING.md)：完整指标名、reward归一化、histogram限制、离线初训/恢复计数、正式4096环境命令。
- 新增报告纳入`artifacts/round2/*.json`及其子目录；原始checkpoint、二进制导出/观测/日志保持本地，不覆盖旧权重。
- 新logger只改记录与计数，不修改HIM-PPO/Estimator损失或梯度路径。训练默认仍为TensorBoard。
- 当前可选UI示例按需导入，tensor-only网络检查无需Kit；正常任务注册与Go2部署函数入口保留。

第二轮还以原model_4执行正常headless Play 50步（默认不导出）并成功；日志`/tmp/himloco_round2_play.log`。

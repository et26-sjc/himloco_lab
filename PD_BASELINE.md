# Go1 显式 PD Baseline

2026-10-22，本地分支 `feat/go1-pd-baseline`，基于成功训练的 Actuator Net 版本。

## 1. 稳定版与实验保护

- 稳定提交：`7c48d8a410946599473255763d04c25d3ae88f96`，分支
  `feat/go1-isaacsim61` 已推送到 `origin/feat/go1-isaacsim61`，没有修改远程 master。
- 成功训练目录：`logs/himloco_rsl_rl/go1_rough/2026-10-22_01-09-29_go1_him_seed1/`。
- 完整训练权重：该目录的 `model_10000.pt`，SHA256：
  `6dd33fbb05ecdb79759a8cb14c74626e15fa572a64968765947a3db2ce61ad79`。
- W&B project：`himloco-go1-baseline_lab`，run ID：`r20l2lro`，run name：
  `go1_him_seed1`，group：`go1-baseline`。
- 完整环境/训练配置、训练时的 Go1 配置源码、版本与文件校验值已保存到
  [actuator_net_reference](artifacts/pd_baseline/actuator_net_reference/)。
  checkpoint 继续保存在原始本地路径，没有改写或加入 Git。
- 原有 `artifacts/round2/legacy_plane/results.json` 修改与 `artifacts/round3/` 文件保留，
  未纳入本次提交；结束时核对其 SHA256 未变化。

继续使用 Conda `himloco_lab`：Python 3.12.15、PyTorch 2.11.0+cu128、Isaac Sim 6.1.0.0、
Isaac Lab 源码 3.0.0（editable，`/home/sjc/RL_Research/IsaacLab-3.0`）、PhysX GPU。
Lab distribution metadata 仍为 17.0.2，以本地源码 VERSION 和实际 API 为准。
本次未安装额外依赖，未更换仿真后端，未修改 Isaac Lab 核心源码。

## 2. 独立任务与配置差异

| 用途 | Actuator Net | PD |
| --- | --- | --- |
| 训练任务 | `Unitree-Go1-Velocity` | `Unitree-Go1-Velocity-PD` |
| Play 任务 | `Unitree-Go1-Velocity-Play` | `Unitree-Go1-Velocity-PD-Play` |
| experiment name / 日志子目录 | `go1_rough` | `go1_rough_pd` |
| 执行器 | 官方 `ActuatorNetMLP` | 官方 `IdealPDActuator` |
| 力矩模型 | 官方位置误差/速度历史网络 | `40*(q_target-q)-dq` |
| 饱和 | 继承 DCMotor 的速度相关力矩包络 | 固定幅值裁剪 |
| Kp / Kd | 网络模型，无固定 PD 增益 | 所有关节 `40 / 1` |
| 网络参数 | `unitree_go1.pt`；pos/vel/torque scale `-1/1/1`，input_order `pos_vel`，input_idx `[0,1,2]` | 不使用网络及其历史/缩放配置 |
| 默认初始化 | 随机策略，不 resume | 相同，不加载 Actuator Net 策略 |

PD 训练配置继承 `Go1EnvCfg`，PD Play 继承 `Go1PlayEnvCfg`，仅替换
`scene.robot.actuators`；Runner 仅覆盖 `experiment_name`。
没有复制 HIM-PPO、Estimator、wrapper、reward 或训练/Play 实现。

4 项 PD 回归测试比较完整配置字典：训练/Play 环境除 actuator 子树外逐项相同；
Runner 除 experiment name 外逐项相同。另比较了成功 Actuator Net 长训练实际保存的
`env.yaml` 与本次 4096 PD 训练 `env.yaml`：执行器之外只有 `log_dir` 不同。
短训的迭代数、logger、run name、W&B 标签等属于本次 CLI 的测试覆盖项，不是默认 PPO 配置变化。

下列设置全部继承：

- 同一官方 Go1 USD、初始根高度 0.4 m、同名关节默认角、关节顺序与 action offset。
- 左/右 hip `+0.1/-0.1`，前/后 thigh `0.8/1.0`，calf `-1.5 rad`。
- action scale `0.25`、sim.dt `0.005 s`、decimation `4`、策略 `50 Hz`。
- 默认 4096 环境、seed 1、rough terrain、地形/指令课程、全部 11 项 reward 及权重。
- 45D 单帧、270D 历史、238D critic、12D action；相同 HIMEstimator/HIM-PPO 和训练超参数。
- 摩擦、质量、COM、reset、推力与外部力随机化全部保留。
  当前启用的随机化均不依赖 Actuator Net；无需新增、移除或替换任何训练随机化项。

## 3. 力矩与速度限制

实际运行读取到以下值，两套任务使用相同官方 USD 和额定限制：

| 关节类型 | PD 固定力矩裁剪 | 模型额定速度 | USD/PhysX effort limit | USD/PhysX velocity limit |
| --- | --- | --- | --- | --- |
| hip / thigh | 23.7 Nm | 30.1 rad/s | 23.7 Nm | 30.1 rad/s |
| calf | 35.55 Nm | 20.06 rad/s | 35.55 Nm | 20.06 rad/s |

采用 Lab 3.0 的 `actuator_effort_limit` / `actuator_velocity_limit` 字段。
后者是模型额定速度和 soft-limit telemetry；`IdealPDActuator` 不使用它生成速度相关力矩曲线。
solver limits 沿用 USD，没有额外设置 `joint_effort_limit` / `joint_velocity_limit` 覆盖值。
armature、摩擦等仍沿用官方 USD。
Actuator Net 的 `saturation_effort` 也是 hip/thigh 23.7、calf 35.55 Nm；PD没有这个DCMotor字段。
两份实际配置的全部10项字段差异见验证报告的 `actuator_config_diff`；其余actuator字段相同。

与旧 Isaac Gym Go1 相同：Kp40、Kd1、默认角、action scale .25、每物理步重算 PD、目标速度0、
无前馈力矩，`hip_reduction=1` 的动作语义相同。
仍有区别：当前采用官方 USD/PhysX、上述分关节限制；旧 Gym 训练 URDF 的 effort limit 为
33.5 Nm，且旧工程可能启用增益/延迟随机化。PD 任务继续使用当前 Lab baseline 的观测缩放、
随机化、地形和终止机制，不声称严格复刻旧 Gym 动力学。

## 4. 每个物理子步的执行路径

依据本机源码：

```text
HimlocoManagerBasedRLEnv.step
  → 原生 ManagerBasedRLEnv.step
    → action_manager.process_action（每20 ms一次）
    → 重复4次：
        action_manager.apply_action（写入相同 q_target）
        scene.write_data_to_sim
          → PhysX Articulation.write_data_to_sim
            → ActuatorCollection.compute（读取当前 q、dq）
              → IdealPDActuator.compute
                 computed = 40*(q_target-q) + 1*(0-dq) + 0
                 applied = clip(computed, ±effort_limit)
            → submit_commands（仅显式 effort）
        sim.step(dt=0.005)
        scene.update(dt=0.005)
```

本地源码位置：`isaaclab/envs/manager_based_rl_env.py`、
`isaaclab_physx/assets/articulation/articulation.py`、
`isaaclab/actuators/actuator_collection.py`、`actuator_pd.py`、`actuator_control.py`。

Lab 为显式执行器将 solver stiffness/damping 清零，PD 返回的 position/velocity command 为 None。
验证同时读取了 `robot.data` 和 `robot.root_view.get_dof_stiffnesses/get_dof_dampings`，
12 个关节的后端内部驱动增益全部为 0，不会重复施加内部 position drive。

控制审计对全部32环境逐子步核对公式；CSV 有界记录 env0 的全部12关节，含 target position、
position、velocity、computed torque、applied torque 和理论值。
位置/速度是计算力矩前的值，避免拿下一子步状态与上一子步力矩错误比较。
`applied torque` 指裁剪后提交的执行器命令，不是接触/反力传感器测量。

## 5. 实际通过的测试

| 验证 | 结果 |
| --- | --- |
| PD 配置、注册、实例隔离与 CPU 公式 | 4 项测试全部通过 |
| 200 Hz PD / 50 Hz 策略 | 103个策略步、412次实际物理步 PD 调用；每次策略步恰好4次 |
| 零动作站姿 | 32环境、2秒、0终止；最终平均根高度0.30216m；首子步力矩0 |
| 固定动作与正/负饱和 | 目标角不变而 q/dq/torque 随子步变化；公式最大误差0，裁剪正确 |
| 内部驱动 | 直接读取后端，stiffness/damping均0 |
| HIM 回归 | 7项历史、四元数、梯度隔离、目标切片、平均loss、timeout和final_obs测试通过 |
| PD rough 集成 | 实际100步rollout+1次PPO更新；Actor/Estimator权重最大变化0.004459/0.004570 |
| checkpoint 重载 | 推理逐值相同，两个optimizer均有状态 |
| 32环境 CLI | 随机初始化，W&B offline、2 iterations，6400 transitions |
| 4096环境 CLI | 随机初始化，TensorBoard、3 iterations，1,228,800 transitions，无NaN/Inf或OOM |
| 4096恢复 | model_3→model_4，累计1,638,400；两个optimizer step均60→80 |
| W&B恢复 | 32环境model_2→model_3，iteration3、累计9600；两个optimizer step均40→60 |
| GUI Play | Kit GUI，4环境、100步，加载PD model_4，指令(0.7,0,0.3)，确定性270D输入 |
| Headless Play | 4环境、50步，由PD日志目录自动选择model_4，指令(0.5,0,0.2) |
| Actuator Net 回归 | 原任务实际rollout/PPO/保存重载/导出均通过，仍为ActuatorNetMLP |
| CPU独立导出 | PD model_1及model_4，batch1/7/64与32条真实rollout；隔离子进程重载JIT/ONNX通过 |

4096新训的 `(Value, Surrogate, Estimation, Swap)` 平均loss：

```text
iteration 1: (0.461335,  0.000051, 0.308454, 0.107492)
iteration 2: (0.246129, -0.005623, 0.204648, 0.104587)
iteration 3: (0.159551, -0.003605, 0.177359, 0.101124)
```

三个iteration分别约6.74/5.59/5.55秒，约61k–74k transitions/s，不含场景启动。
32环境集成验证恢复推理相同；4096和32环境的实际恢复训练均确认 Actor、Critic、
Estimator encoder/target/prototype继续更新。恢复后不声称物理状态与随机数流逐步复现。

model_4独立CPU导出：重载PyTorch和JIT相对原PyTorch最大误差均0；
ONNX所有测试最大误差 `1.79e-7`；真实rollout最大/平均误差 `5.96e-8 / 1.34e-8`。
JIT为完整270→12的Estimator+Actor；split ONNX为encoder270→19、actor64→12。
隔离进程仅加载Torch/ONNX Runtime，不导入项目Runner、Critic或Isaac Lab。

### 日志与可复现结果

W&B offline二进制history和TensorBoard event文件均实际读取验证：
32环境iteration为1/2、恢复后3；timesteps为3200/6400、恢复后9600。
每个run有50个TensorBoard标量，包括：

- trainer/iteration、trainer/timesteps、Metrics/total_timesteps。
- Value/Surrogate/Estimation/Swap Loss、learning_rate、mean_noise_std。
- mean_reward、mean_episode_length及其时间曲线。
- FPS、collection/learning/iteration/total time。
- 全部11项启用reward的 `Episode/rew_*` 和原生 `Episode_Reward/*`。
- terrain level、max command x、termination分类、指令误差和原有success metric。
- histogram开关实测：每iteration仅250个action/value样本（配置上限256），W&B和TB均有记录。

保留原logger实现，未重复初始化session；checkpoint artifact默认不上传。
完整逐项结果见 [training_validation.json](artifacts/pd_baseline/training_validation.json)，
控制/HIM/导出及Actuator Net回归JSON也位于该目录。
逐子步原始CSV：`logs/go1_pd_validation/control/substeps.csv`。

本次运行目录：

```text
logs/himloco_rsl_rl/go1_rough_pd/
  2026-10-22_20-55-30_pd_smoke32/          model_2.pt
  2026-10-22_20-56-42_pd_smoke4096/        model_3.pt
  2026-10-22_20-58-42_pd_resume4096/       model_4.pt
  2026-10-22_21-01-49_pd_wandb_resume32/   model_3.pt
logs/go1_pd_validation/
  control/                  逐子步审计
  him/                      PD HIM集成、model_1、真实观测
  standalone_export/        model_1独立导出
  model4_export/            model_4独立导出
  actuator_net_regression/  原任务回归
```

`validate_pd.py`的平地、无噪声/随机化、nominal reset是专门的控制诊断覆盖；
`validate_go1.py`为检验actor/critic前45维而关闭actor噪声。
上述CLI32/4096训练均使用原rough任务和完整训练随机化，没有应用这些诊断覆盖。

## 6. 启动命令

```bash
source /home/sjc/anaconda3/etc/profile.d/conda.sh
conda activate himloco_lab
cd /home/sjc/himloco_lab
```

正式训练，从随机初始化开始，不添加resume/checkpoint参数：

```bash
WANDB_PROJECT=himloco-go1-baseline_lab WANDB_MODE=online \
python scripts/himloco_rsl_rl/train.py \
  --task Unitree-Go1-Velocity-PD --headless \
  --num_envs 4096 --seed 1 --max_iterations 10000 \
  --logger wandb --run_name go1_pd_him_seed1 \
  --wandb_group go1-pd-baseline --wandb_tags go1 pd isaaclab3 physx
```

TensorBoard默认命令：

```bash
python scripts/himloco_rsl_rl/train.py \
  --task Unitree-Go1-Velocity-PD --headless \
  --num_envs 4096 --seed 1 --max_iterations 10000 --run_name go1_pd_him_seed1
tensorboard --logdir logs/himloco_rsl_rl/go1_rough_pd
```

GUI Play自动选择正式训练run中的最新checkpoint：

```bash
python scripts/himloco_rsl_rl/play.py \
  --task Unitree-Go1-Velocity-PD-Play --num_envs 4 \
  --load_run '.*_go1_pd_him_seed1' --velocity 0.7 0 0.3 --real-time
```

已实际通过的短训checkpoint Play与4096恢复命令：

```bash
python scripts/himloco_rsl_rl/play.py \
  --task Unitree-Go1-Velocity-PD-Play --num_envs 4 --seed 1 \
  --checkpoint logs/himloco_rsl_rl/go1_rough_pd/2026-10-22_20-58-42_pd_resume4096/model_4.pt \
  --velocity 0.7 0 0.3 --max_steps 100 --real-time

python scripts/himloco_rsl_rl/train.py \
  --task Unitree-Go1-Velocity-PD --headless --num_envs 4096 --seed 1 \
  --resume --checkpoint logs/himloco_rsl_rl/go1_rough_pd/2026-10-22_20-56-42_pd_smoke4096/model_3.pt \
  --max_iterations 1 --run_name pd_resume4096
```

`--max_iterations`是本次额外更新数：例如从3000恢复到10000应填写7000。
恢复命令使用PD自己的checkpoint；正式新训不会默认加载其他任务权重。

复测控制、HIM与独立导出：

```bash
python scripts/himloco_rsl_rl/validate_pd.py --headless --num_envs 32
python scripts/himloco_rsl_rl/validate_go1.py \
  --task Unitree-Go1-Velocity-PD --headless --num_envs 32 --output logs/go1_pd_validation/him
python scripts/himloco_rsl_rl/validate_export.py \
  --checkpoint logs/himloco_rsl_rl/go1_rough_pd/2026-10-22_20-58-42_pd_resume4096/model_4.pt \
  --observations logs/go1_pd_validation/him/rollout_observations.pt \
  --output logs/go1_pd_validation/model4_export
```

独立W&B smoke / histogram测试命令：

```bash
WANDB_PROJECT=himloco-go1-baseline_lab WANDB_MODE=offline \
python scripts/himloco_rsl_rl/train.py \
  --task Unitree-Go1-Velocity-PD --headless --num_envs 32 --seed 1 --max_iterations 2 \
  --logger wandb --run_name pd_smoke32 --wandb_group go1-pd-baseline-smoke \
  --wandb_tags go1 pd isaaclab3 physx \
  --wandb_histograms --wandb_histogram_interval 1 --wandb_histogram_samples 256
```

## 7. 文件、提交与验证边界

- `robots/go1/pd_env_cfg.py`：独立PD训练/Play配置。
- `robots/go1/__init__.py`：注册两个PD任务，原任务注册保持。
- `agents/himloco_rsl_rl_cfg.py`：独立experiment name。
- `validate_pd.py`、`tests/test_pd_baseline.py`：配置公平性和实际子步力矩审计。
- `validate_go1.py`：可选择原任务/PD任务，复用完整HIM验证并保存真实观测用于独立导出。
- `artifacts/pd_baseline/`：稳定版记录及本轮验证JSON。
- `PD_BASELINE.md`、迁移记录及README：启动与复现说明。

提交：稳定版 `7c48d8a`；PD代码与验证器 `736ed7a`；本文件和验证报告的提交可用
`git log -2 --oneline`查看。PD分支仅在本地提交，未推送。

本轮证明了PD控制、训练更新、恢复、GUI推理、日志与导出的工程链路。
**尚未运行完整10000轮PD训练，也未证明PD运动表现或sim2real性能优于/等于Actuator Net。**
后续应使用相同seed和训练预算完成PD训练，再固定地形/指令做物理状态指标评估；
多seed结论需要额外实验。

额外试用GUI录像选项时发现当前环境缺少MoviePy；去掉可选`--video`后GUI测试通过。
本次未验证录像功能，也未为它新增依赖。未开发MuJoCo、未修改旧Gym权重或实机控制器。

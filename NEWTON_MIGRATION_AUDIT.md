# Go1 HIMLoco PD → Newton MJWarp 迁移审计

审计日期：2026-10-29。项目：`/home/sjc/himloco_lab`。

## 1. 结论与本轮边界

**可行，建议新增独立的 Newton + MJWarp 任务，复用现有 Manager-Based 环境和 HIM 算法。**
本地已具备相匹配的 Newton/Warp/MuJoCo Warp 软件栈；官方 Go1 USD 已通过本轮的
Newton 导入器内存解析。未发现需要替换 HIM-PPO、Estimator、Actor、history 或 Runner 的理由。
但当前项目尚不能仅增加一个后端命令行参数就视为完成迁移。

需要实施的重点是：配置驱动的启动、Newton 物理/碰撞配置、关节和刚体公共顺序、
高度扫描范围、PD 力矩传递、随机化语义与运行时验证。

本轮只新增本报告；没有修改训练/Play/部署代码，没有安装或升级依赖，没有切换分支、
改写 checkpoint、初始化 SimulationApp/SimulationContext、创建求解器或运行 Newton 仿真/训练。
实际执行了包导入、配置构造、USD 读取与解析、checkpoint CPU 推理和现有 CPU 单元测试。
下文明确区分已执行检查、源码结论和待实现/待验证事项。

“回到 PD baseline”在本轮指以原始 `Unitree-Go1-Velocity-PD` 为对照，
不回滚 Git，不带入 DeployV1 的 reward/command 调整。

## 2. Git、版本与依赖

### 2.1 工作区与版本锚点

| 项目 | 实际结果 |
| --- | --- |
| 当前分支 | `feat/go1-pd-baseline` |
| 当前 HEAD | `c6a214e3e74868c50e8a76ed692c9a354fa72703` |
| HEAD 提交 | `feat: inspect Go1 heading drift across saved checkpoints` |
| 相对本地远程跟踪引用 | 比 `origin/feat/go1-pd-baseline` 多 3 个提交；本轮没有 fetch/push |
| 本地 Lab 仓库 | `/home/sjc/RL_Research/IsaacLab-3.0`，工作区干净 |
| Lab SHA / tag | `ae37b028ea415c91ea2bc32609efcd759ed2b974` / `v3.0.0-EA` |
| Lab VERSION | `3.0.0`，不能据此将其等同于后续 GA 源码 |

审计开始已有以下用户工作区内容，本轮保留：

```text
 M artifacts/round2/legacy_plane/results.json
?? artifacts/legacy_pd_comparison/pd_visual/
?? artifacts/round3/
?? artifacts/sim2sim/deploy_v1_backward_view/
?? artifacts/sim2sim/go1_damping0_visual/
?? artifacts/sim2sim/go1_run/
```

### 2.2 实际安装

使用既有解释器 `/home/sjc/anaconda3/envs/himloco_lab/bin/python`；审计终端本身没有激活 Conda。

| 组件 | 实际版本 / 位置 |
| --- | --- |
| Python | `3.12.15` |
| PyTorch / CUDA runtime | `2.11.0+cu128` / `12.8` |
| GPU | RTX 5070 Ti，Torch 检测可用，总显存约 15.45 GiB |
| Isaac Sim | `6.1.0.0`，既有 Conda 的 site-packages |
| Isaac Lab | distribution `17.0.2`；editable `.../IsaacLab-3.0/source/isaaclab` |
| isaaclab-newton | distribution `5.4.1`；editable `.../IsaacLab-3.0/source/isaaclab_newton` |
| isaaclab-physx | distribution `6.0.0`；editable，同一 Lab 仓库 |
| Newton | `1.5.2`，Conda site-packages/newton |
| Warp | `warp-lang 1.16.0`，Conda site-packages/warp |
| MuJoCo / MuJoCo Warp | `3.11.0` / `3.11.0` |
| Newton USD schemas | `0.4.1` |
| USD Exchange | `2.3.0` |

本地 [pyproject.toml](/home/sjc/RL_Research/IsaacLab-3.0/pyproject.toml:399)
明确覆盖为 Newton 1.5.2、MuJoCo/MuJoCo Warp 3.11.x、Torch 2.11.0、schemas >=0.4.1；
Warp 固定为 1.16.0，Linux x86_64 的 Torch 源为 cu128。本机核心组合符合这些约束。
官方 [3.0 EA 发布说明](https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0-EA)
也明确对应 Sim 6.1、Python 3.12、Torch 2.11、Warp 1.16、Newton 1.5.2。
**不应为了套用最新 GA 文档而升级现有 Sim、Lab、CUDA 或 Newton。**

`torch`、`warp`、`newton`、`mujoco`、`mujoco_warp` 实际导入成功；
`NewtonCfg(solver_cfg=MJWarpSolverCfg())` 实际构造成功，解析的 manager 为
`isaaclab_newton.physics.mjwarp_manager:NewtonMJWarpManager`。
这些检查不等同于 Newton GPU 求解器已启动成功。

### 2.3 pip check 并非全绿

本轮 `python -m pip check` 报出 7 项既有依赖问题：

| 包 | 报告的问题 | 本轮判断 |
| --- | --- | --- |
| isaacsim-core | 缺少 torchaudio | 当前 CPU 算法/配置检查未受阻；不是 Newton 启动成功的证明 |
| isaacsim-kernel | click 要求 8.1.7，实际 8.5.0 | 保留记录，未擅自降级 |
| tensordict 0.15.1 | 要求 torch >=2.13.0a0，实际 2.11 | 当前自定义 HIM Runner 不以 TensorDict 为数据接口；仍属于环境元数据冲突 |
| torch | 要求 setuptools <82，实际 84.0.0 | 主要需关注后续构建/安装路径；本轮未安装软件 |
| ipython | 要求 psutil >=7，实际 5.9.8 | 与本轮训练接口审计分开记录 |
| isaacsim-core | llvmlite 要求 0.46.0，实际 0.50.0 | 尚未验证全部相关扩展路径 |
| isaacsim-core | Newton pin 1.5.0，实际 1.5.2 | 本地 Lab 的 override 明确选择 1.5.2；不应为消除这条元数据冲突降级 Newton |

结论是“核心版本与该 EA 源码预期一致，环境存在额外依赖冲突”，不是“全部依赖已经验证兼容”。
下一阶段先复现具体功能错误，再最小化处理相关依赖；不要执行整套环境的批量升级。

## 3. 选定的原始 PD 对照

源码入口：

- [Go1 PD 配置](source/himloco_lab/himloco_lab/tasks/locomotion/robots/go1/pd_env_cfg.py)。
- [Go1 配置](source/himloco_lab/himloco_lab/tasks/locomotion/robots/go1/velocity_env_cfg.py)。
- [共享场景与 MDP 配置](source/himloco_lab/himloco_lab/tasks/locomotion/velocity_env_cfg.py)。
- [PD 记录](PD_BASELINE.md)。

本轮重新读取完整训练保存的 `params/env.yaml`、`params/agent.yaml` 和最终 checkpoint：

```text
/home/sjc/himloco_lab/logs/himloco_rsl_rl/go1_rough_pd/
  2026-10-22_23-51-36_go1_pd_him_seed1/
    params/env.yaml
    params/agent.yaml
    model_10000.pt
    exported/policy.pt
    exported/policy.manifest.json
    wandb_session.json
```

| 项目 | 实际原始 PD 配置 |
| --- | --- |
| Task / experiment | `Unitree-Go1-Velocity-PD` / `go1_rough_pd` |
| 执行器 | `IdealPDActuator`，所有关节 Kp=40、Kd=1 |
| 动作 | scale=0.25，默认关节位置 offset，无 hip 缩减 |
| 时间 | dt=0.005 s，decimation=4，200 Hz PD / 50 Hz policy |
| 环境 / seed | 4096 / 1；完整训练 10000 iterations |
| 观测 | policy 45D，6 帧 270D，critic 238D，action 12D |
| 高度奖励 | target=0.30 m，weight=-1.0 |
| 指令 | min_command_norm=0.20，heading_command=True，rel_heading_envs=1.0 |
| DR | 摩擦、trunk 质量/COM、周期外力、推扰、reset 随机化保留 |
| 训练初始化 | resume=False，随机策略初始化 |
| W&B | project `himloco-go1-baseline_lab`，run `vzm57xqb` |
| 训练 Git | `PD_BASELINE.md` 记录为 `73fc02a`；不是用当前 HEAD 冒充历史训练 SHA |

最终 checkpoint SHA256：
`3e2ec748f7657ac502334ce69e1275780ecf4cf63251c8afad7b25f5615b29f3`。
JIT SHA256：`82d69e89c1a5759d003da86d056e1152e4c5317a3dc19b49cbb11c45b661c336`。
本轮重新核对两者，并确认 iter=10000、累计 4,096,000,000 transitions、权重有限。
它是最终模型；现有流程没有独立 best 模型选择机制。

Newton 首版应继承这些条件。DeployV1 的 0.32 m/-2.0 高度奖励、腿碰撞惩罚、
0.10 死区、50% heading 环境不纳入本轮迁移基准。

## 4. Go1 USD 对 Newton 的适用性

### 4.1 本轮实际检查

复用 `isaaclab_assets.robots.unitree.UNITREE_GO1_CFG` 指向的官方资产，缓存路径：

```text
/tmp/https/omniverse-content-production.s3-us-west-2.amazonaws.com/
Assets/Isaac/6.1/Isaac/IsaacLab/Robots/Unitree/Go1/go1.usd
```

| 层 | SHA256 |
| --- | --- |
| go1.usd | `156eaea84566ddd6f5f1eb7db2ec0a70fc32c2a5b6cd592f879e1f63dfed6b31` |
| Props/instanceable_meshes.usd | `dccc3f09108c8d3ea6d58dfb869c87a53e76c82a4cd845f4074fba8a8046fe58` |

通过 `pxr.Usd` 读取并遍历 instance proxies 后，确认：

- articulation root 为 `/go1/trunk`，17 个刚体、12 个 revolute joints、4 个固定足端关节。
- 21 个启用的碰撞形状，包含 4 个独立足端球体；普通不展开实例的遍历会漏掉碰撞形状。
- USD 中 12 个关节 drive 的 stiffness/damping 均为 0，maxForce 按关节为 23.7/35.55 Nm。
- 未发现专用 NewtonActuator prim 或 Newton/MuJoCo schemas；这不妨碍标准物理属性的导入。

随后直接使用本地 `NewtonMJWarpManager.create_builder()` 及其 Newton/PhysX/MJC schema resolvers，
调用 `builder.add_usd()`，仅在内存解析、关闭视觉几何加载；**没有 finalize model、创建 solver 或 step**。
结果：

```text
body_count=17
joint_count=17                 # free root + 12 revolute + 4 fixed
joint_dof_count=18             # free root 6 + driven joints 12
shape_count=21
mass_sum=13.100449070334435 kg
all_body_masses_positive=True
min_body_inertia_eigenvalue=9.60000033956021e-06 kg*m^2
```

### 4.2 是否要转换资产

**首选直接复用现有 USD，不必先转 URDF/MJCF，也不应换成 Menagerie/Go2 模型。**
本地 Newton 导入路径已能读取其标准 USD Physics 和受支持的 PhysX 属性；
官方 `contrib/velocity/config/go1/rough_env_cfg.py` 同样复用 `UNITREE_GO1_CFG` 并提供 Newton 参数。

需要的是独立的 Newton task/asset override，核对最终导入的质量、惯量、关节限位、
自碰撞过滤、碰撞近似、drive 与被动阻尼。仅成功读取 USD 不能证明这些参数全部被 MJWarp 消费。

新 URDF/MJCF 的多物理转换是另一条资产准备路径；本项目已有 USD 不应无故重新转换。
只有发现具体缺失字段时，才在项目的独立配置或新 USD overlay 中补齐，保留官方缓存资产原件。
背景见官方 [资产准备说明](https://isaac-sim.github.io/IsaacLab/main/source/how-to/prepare_asset_for_newton.html)；
具体字段以本机 EA 源码为准。

## 5. Joint/body ordering 与 270D history 契约

### 5.1 本轮解析得到的实际顺序

当前 Lab policy 的公共关节顺序由 `GO1_JOINT_NAMES` 显式指定：

```text
FL_hip, FR_hip, RL_hip, RR_hip,
FL_thigh, FR_thigh, RL_thigh, RR_thigh,
FL_calf, FR_calf, RL_calf, RR_calf
```

以上实际名称均带 `_joint` 后缀。Newton 导入器本轮解析的 12 个驱动关节顺序为：

```text
FL_hip, FL_thigh, FL_calf,
FR_hip, FR_thigh, FR_calf,
RL_hip, RL_thigh, RL_calf,
RR_hip, RR_thigh, RR_calf
```

这里的结论来自 Newton parser 输出，不是根据 USD 文本出现顺序推定求解器顺序。
运行时仍必须读取 `backend_joint_names` 再断言。

若 `native12` 是上述 Newton 驱动轴顺序，则本次资产的映射为：

```python
# public[i] = native12[public_from_native[i]]
public_from_native = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]
# native12[j] = public[native_from_public[j]]
native_from_public = [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
```

这些数组是检查结果，实施时应按名称构建，验证名称唯一、完整且映射互逆。
不能直接拿它们索引含 free-root 自由度的 18D solver 数组。

原 PhysX 的刚体顺序已在 [实际 PD 集成报告](artifacts/pd_baseline/him_validation.json) 保存：
`trunk, 4 hips, 4 thighs, 4 calves, 4 feet`，每组 FL/FR/RL/RR。
本次 Newton parser 则为 `trunk` 后按每条腿的 hip/thigh/calf/foot 排列。
该资产的公共 body 从 native body gather 为：

```text
[0, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 4, 8, 12, 16]
```

### 5.2 推荐在资产 API 层完成排列

Newton 子类保留现有 `robot.joint_ordering = GO1_JOINT_NAMES`，
再显式指定完整公共 body 名称序列，或经核对后使用 `body_ordering="physx"`。
建议冻结报告中的完整名称表，以减少未来排序 emulation 变化的影响。
本地 `BaseArticulation` 与 Newton 的 data/writer 均已实现公共↔backend 映射。

- `robot.data`、`find_joints/find_bodies`、高层状态/目标写入使用公共顺序。
- `robot.root_view` 和直接 Newton/MuJoCo 数组使用 backend 顺序，不能混用公共 ID。
- ContactSensor 有自己的 sensing/body 顺序，必须通过 sensor 名称独立解析；
  不能将 articulation 的 body ID 直接拿来索引 contact buffer。
- 外力、质量和 COM 继续通过高层 API 与 `SceneEntityCfg(..., body_names="trunk")` 选择。

这样 ObservationManager、ActionManager、history、Actor 和 Estimator 都保持原接口，
不需要每步在 270D history 上重复转换，也避免动作与 last_action 排列不一致。

### 5.3 保留完整的观测语义

| 单帧切片 | 保持的含义 |
| --- | --- |
| 0:3 | 物理速度指令，Lab 原始 command scale；不引入 Gym 的 [2,2,0.25] |
| 3:6 | root 机体系角速度 ×0.25 |
| 6:9 | root 投影重力 |
| 9:21 | 公共关节顺序的 q-q0 |
| 21:33 | 同顺序的 (dq-dq0)×0.05，当前 dq0=0 |
| 33:45 | 上一 policy step 的原始归一化 action，同一公共顺序 |

q0 按上述公共顺序是
`[0.1,-0.1,0.1,-0.1, 0.8,0.8,1.0,1.0, -1.5,-1.5,-1.5,-1.5]` rad。
维持当前每个 observation term 的 noise→clip→scale 顺序和 critic 无噪声配置。

history 仍为 `[current, previous1, ..., previous5]`，45×6=270D。
reset 仅保留新当前帧，过去五帧填零；不把末帧旧动作带入新 episode。
若以后在 raw solver 边界使用 adapter，必须先逐帧映射 q/dq/last_action，再组 history；
既有带缓存数据需 reshape 为 `[N,6,45]` 后对全部六帧转换，不能只转换当前帧。
优先的公共排序方案无需此额外 adapter。

critic 保持 `[45D, true_base_velocity×2 的3D, disturbance3D, height_scan×5 的187D]`。
Estimator 的 velocity target 为 critic[45:48]，target encoder 输入 critic[3:48]；
不得调整 height scan、删除 disturbance 或改变切片而保持旧监督代码不变。

### 5.4 四元数与机体坐标

本地 Lab 的 root quaternion/math API 使用 XYZW；Newton/Warp 公共状态路径也使用 XYZW，
不应像 SDK/MuJoCo 独立接口那样再次交换 WXYZ。
`root_pos_w/root_quat_w` 是 root link pose；`root_lin_vel_b/root_ang_vel_b` 是
`root_com_*_vel_b` 的别名。继续沿用当前高层语义，不能无意换为 link-origin 线速度。
COM 与 link 速度的区别在转动和 COM 随机化时尤其重要。
foot reward 的位置/速度与 root 数据也应保留当前含义，先用同姿态 FK、已知 root twist 和重力测试核实。

## 6. Newton MJWarp 的 PD 适配

### 6.1 首版保留当前显式执行器

建议 Newton 首版仍用 `GO1_PD_ACTUATOR_CFG`，并保持
`sim.use_newton_actuators=False`。这个 flag 控制执行器执行路径，**不是关闭 Newton 物理后端**。

本地已实现以下调用链：

```text
HIM action (50 Hz)
  → JointPositionAction：q_target = clip(q0 + 0.25*action, -100, 100)
  → 原生 ManagerBasedRLEnv 的 4 次循环
  → action_manager.apply_action()
  → scene.write_data_to_sim()
  → Newton Articulation.write_data_to_sim()
  → ActuatorCollection.compute(0.005)
  → IdealPDActuator.compute(q_target, q, dq)
  → NewtonActuatorControl.submit_commands() 映射并写 effort
  → Newton MJWarp step
```

目标速度和前馈力矩为零时，保持：

```text
tau_computed = 40*(q_target-q) - dq
tau_applied = clip(tau_computed, -joint_motor_limit, joint_motor_limit)
```

clip 是当前 position action 的绝对目标裁剪，不能偷换成先 clip raw action。
本地 `write_resolved_joint_properties()` 为显式执行器将后端 drive stiffness/damping 写零，
Newton builder 也将这些关节设为 EFFORT mode；应实际读取求解器确认，防止重复 PD。

| 项目 | hip/thigh | calf |
| --- | --- | --- |
| 显式执行器力矩限幅 | 23.7 Nm | 35.55 Nm |
| actuator 额定速度 | 30.1 rad/s | 20.06 rad/s |
| 目标位置 scale | 0.25 | 0.25 |
| Kp/Kd | 40/1 | 40/1 |

`IdealPDActuator` 不实现 DCMotor 速度相关力矩包络。MJWarp 不执行 Newton 的
`joint_velocity_limit` hard clamp；不能将保留了 30.1/20.06 的配置数值等同于求解器限速生效。
首版保留额定速度和当前 PD 公式，记录实际超速率，不私自增加速度制动力或换控制器。
这属于相对 PhysX 必须披露的物理行为差异。

裸 USD 本轮导入的 joint armature/friction 均为零；原 PD config 对 armature/friction 没有覆盖。
官方 Go1 Newton rough task 为其 Actuator Net 设置 armature=0.02，可作敏感性实验参考，
不能未经测量就把该值当作当前 PD baseline 的固有参数。
首版审计保持原值，若数值稳定性要求非零 armature，只在 Newton 配置显式增加，
记录为模型差异并做响应对比；不叠加以前独立 MuJoCo 模型中的额外 passive damping。

### 6.2 时间与 CUDA graph

初始设 `NewtonCfg.num_substeps=1`，保持每 0.005 s 反馈一次 PD，policy 每 0.02 s 更新。
若之后使用 `num_substeps=2`，solver 内部为 0.0025 s，但上述显式 PD 仍每个
0.005 s physics tick 更新一次，不会自动变为 400 Hz 控制器。

该版本仍支持 Lab 显式执行路径，但已标记 deprecated；在锁定 EA 版本中可作为语义对齐参考，
长期维护可再迁到原生 actuator。保留这条路径时仍可使用求解器级 CUDA graph，
但不具备完整 policy decimation 融合的性能收益。

### 6.3 后续可选的 Newton 原生 PD

`use_newton_actuators=True` 支持同一 IdealPD config，spawn 会生成
NewtonPDControlAPI + NewtonMaxEffortClampingAPI。它也是显式反馈 PD，
不等于改用 solver 隐式 position drive，更不代表 WTW 的固定 Kp20/Kd0.5。

必须先将其与参考路径做逐 tick 力矩对照，再启用性能优化。原生路径下
`robot.actuators[...]` 不一定返回 Python `IdealPDActuator`，现有验证器的类型和 `.compute()` patch 要适配。
telemetry 应优先读取公共 `actuators.computed_effort/applied_effort` 或 `robot.data`。

**需重点验证接触采样频率：** 本地 `_simulate_full()` 在 decimation 循环结束后才调用
`_update_sensors()`；不能假定融合后仍保留四个 physics tick 的 contact history。
原 baseline 的 contact update=0.005、history=3、track_air_time=True 必须有实测语义对照。
本轮不建议直接启用原生融合路径作为第一版。

## 7. 场景、传感器、事件与 reward 审计

| 范围与实际文件 | 发现 | 修改/验证要求 |
| --- | --- | --- |
| `tasks/locomotion/velocity_env_cfg.py:RobotEnvCfg.__post_init__` | 固定创建 PhysxCfg，设置 TGS、迭代次数、bounce threshold、GPU 容量 | Newton 子类在 super 后替换整个 physics config；不向 NewtonCfg 写 PhysX 字段 |
| `scripts/himloco_rsl_rl/train.py` | Hydra 配置解析前无条件启动 AppLauncher | 新增配置先行、launch_simulation 包裹的 Newton 启动入口；复用 HIM training body |
| `scripts/himloco_rsl_rl/play.py` | 默认强制 KitVisualizerCfg，循环依赖 simulation_app.is_running | Newton Play 选择 newton_gl 等已支持 visualizer；退出/清理用原生 simulation 生命周期，无 Kit app 假设 |
| `play_interactive.py` / `play_onnx.py` | 同样绑定 AppLauncher/Kit 生命周期 | 分阶段适配；首版 Newton 标准 Play 完成后再扩展交互/ONNX Play |
| `robots/go1/velocity_env_cfg.py` | joint_ordering 已明确；body_ordering=None；trunk 已正确引用 | Newton 配置冻结 body_ordering，并保留完整 joint/action 公共顺序 |
| `robots/go1/pd_env_cfg.py` | 官方显式 PD，无直接 PhysX API 调用 | 控制公式可复用；验证 torque 提交、clip、drive=0 和 200 Hz |
| `mdp/observations.py` | 已用 ProxyArray `.torch`、公共 wrench composer、RayCaster 数据 | 预期复用；验证 source frame、force 3D、height 187D 和有效值 |
| `mdp/rewards.py` | energy、foot clearance 等从公共 data 获取，smoothness 从 action history 获取 | 保留 reward 数学式/权重；核验 applied torque 采样、body frame/FK、height hit 与 dt |
| `mdp/events.py` | 周期外力通过 permanent_wrench_composer 高层接口 | 预期可复用；测试 trunk 力的坐标系、只作用目标环境、reset 清零与 critic force 一致 |
| Lab `envs/mdp/events.py` | 材质、mass、COM 已有多后端实现 | 材质语义不同；mass/COM 需核验 model-change 与 MJWarp 状态同步 |
| ContactSensor / illegal_contact | Newton 有 net_normal_forces_w、history、air time 实现 | 保留 trunk 1 N 终止条件，测试真实碰撞、history/采样周期、非 contact impulse 单位 |
| 两个 RayCaster | 当前 global_world_only=False | Newton 配置必须设 True，排除自己机器人，保留 yaw pattern、trunk 和 20 m offset |
| TerrainImporter / 本项目自定义 terrain | 使用共享 generator；没有直接 PhysX tensor 操作 | 地形生成/课程可复用，非凸 mesh 碰撞需要正确管线，验证 ray/碰撞几何一致 |
| reset / root、joint 写入 | 使用原生事件与高层 articulation writers | 不复制 PhysX stepping/reset，测试多环境局部 reset、warm-start 清理、关节与 root 映射 |
| `mdp/commands` / `mdp/curriculums.py` | 读取公共状态和 Manager reward/episode 数据 | 算法可复用，保留原 heading 与 0.2 指令死区、地形/指令课程 |
| `envs/him_manager_based_rl_env.py` | 委托原生 step/reset，仅维护两步前 action 和 episode log | 预期原样复用，实际验证 native final_obs、manual reset 与 smoothness |
| `rsl_rl/*` / `utils/export_policy.py` | 未发现 PhysX/omni 求解器接口依赖 | 保留 HIM 数学方法、history、checkpoint schema 与导出链路 |
| `ui_extension_example.py` | Kit 专用扩展示例 | 不在本轮已验证的纯 Python配置/算法导入链路中；不是 Newton 训练的必要模块 |
| `assets/unitree.py` | 自定义 Go2 资产/部署配置 | Go1 使用官方 UNITREE_GO1_CFG；不将 Go2 URDF/SDK 字段带入 Newton Go1 |

### 7.1 Height scanner 是必须修正的语义差异

PhysX RayCaster 按 `mesh_prim_paths=["/World/ground"]` 查询指定 mesh；
Newton 的默认 RayCaster 使用 live scene BVH。其 `global_world_only=False` 时也命中本环境机器人。
当前射线从机身上方 20 m 向下，可能首先击中 trunk/腿，把机体误当作地面。
后果包括 critic 地形输入错误、base_height reward 错误；网络 shape 仍然完全正常。

Newton 两个 scanner 均应 `global_world_only=True`，地形须位于共享 global world。
保持 187 点网格顺序、base-height 小网格、yaw alignment 和 update=0.02 s。
应在 plane、坡面、台阶分别比较 ray_hits_w 与已知几何；确认 data.pos_w 的含义不被 20 m offset 改变。

### 7.2 Rough terrain 不能直接使用默认碰撞设置

实际读取当前 generator，六类 subterrain 的 `convert_to_heightfield` 为：

```text
hf_pyramid_slope=True
hf_pyramid_slope_inv=True
hf_slope_with_noise=True
pyramid_stairs=False
pyramid_stairs_inv=False
discrete_obstacles=True
```

TerrainImporter 只有在所有子地形都可转 heightfield 时才标记整张地图；
因此当前 atlas 仍是混合非凸三角 mesh。不能默认把它当作已导入的 MuJoCo heightfield，
也不能用凸包接触冒充原楼梯/坡面。

建议 plane 控制诊断可先使用 `MJWarpSolverCfg(use_mujoco_contacts=True)`，
不同时设置 `collision_cfg`；rough 任务优先保留原 atlas 并使用
`use_mujoco_contacts=False` + `NewtonCollisionPipelineCfg`，让 Newton 处理非凸三角碰撞，
MJWarp 仍负责刚体约束求解。这依然是 Newton + MJWarp。

本地官方 `RoughPhysicsCfg` 已提供此组合，以及 njmax=1000、nconmax=300、
max_triangle_pairs=2,500,000、implicitfast、num_substeps=2 的参考。
这些是相似任务的起点，不是已证明适合本项目/16GB 显卡的最佳值。
首版先固定 num_substeps=1 复现 PD 频率，再对比 2；接触参数、margin/gap、
容量和 solver iterations 按最繁忙环境实测，而不是照搬 PhysX 的 2**23 缓冲参数。

### 7.3 Domain randomization 不能只比较配置字典

材质：本地 Newton 实现从 static_friction_range 连续采样单一 mu；
`dynamic_friction_range` 和 `num_buckets` 被忽略。
虽然静/动范围在 baseline 中同为 [0.2,1.25]，原 PhysX 的两次采样与 64 桶分配仍不等价。
Newton 配置应明确记录单一 mu 的 [0.2,1.25]、连续采样、restitution=0，以及求解器混合规则差异。
不能将“参数没改”表述为“随机化分布完全一样”。

质量：官方事件通过 `set_masses_index/set_inertias_index` 写入公共顺序，
Newton 发出 BODY_INERTIAL_PROPERTIES 变更通知。原 trunk [-1,+2] kg 与重算 inertia 行为可望保留，
需确认每环境求解器 mass/inertia 及动力响应实际改变。

COM：Lab 的 `randomize_rigid_body_com` 文档仍警告 MJWarp 运行时 COM 更新可能不完整，
且 Newton setter 只接收 COM position，不设置 COM quaternion。
但是继续追踪本机 Newton 1.5.2 发现：

```text
BODY_INERTIAL_PROPERTIES
  → _update_model_inertial_properties() 写 body_ipos/mass/inertia
  → set_const_fixed()
  → set_const_0()
```

其模型编译还包含为惯性更新预留 qM 存储布局的处理。
因此不能仅凭 Lab 的警告认定本机仍存在该 bug，也不能凭更新代码存在就宣称 DR 已验证正确。
原事件是 startup，在 solver 初始化后执行，不等于 build-time COM 赋值。

下一阶段须专测每环境 ±0.03 m startup COM：模型/求解器读取、质量矩阵/动力响应、
首步冲量、reset、多次事件和 CUDA graph 路径。若通过则保留原事件。
若确有复现问题，才研究项目侧 build-time COM 配置/分组资产构建或受支持的初始化 hook；
不得直接依赖未承诺稳定的 `_per_world_builder_hooks` 作为默认实现，更不能修改官方核心源码。
暂时关闭 COM 只能作为有记录的诊断模式，不算完成 baseline 全功能迁移。

本轮核对已安装 solver_mujoco.py 与 wheel RECORD 一致，SHA256 为
`cd4e7dae4b37d58539621b4d03bb9f5d4f6fd7f4a8822f0fac7e5eeb9cbaabfa`，
上述追踪不是基于另一个版本的源码。

## 8. HIM、环境接口与 checkpoint 的复用判断

| 部分 | 判断 | 必须保持的合同 |
| --- | --- | --- |
| Actor / Estimator 网络 | 可原样复用 | 270→19 的 estimator，45+19→12 的 actor；无 privileged Actor 输入 |
| HIMEstimator 更新 | 可原样复用 | 独立 optimizer、Sinkhorn/prototype、原 estimation/swap loss、既有 target 切片 |
| HIM-PPO / storage | 可原样复用 | clipped PPO、value、entropy、KL adaptive、terminal target 和 timeout bootstrap |
| history | 可原样复用 | 当前帧在前、5 个过去帧、每环境 reset 清零、不原地覆盖 rollout 引用 |
| VecEnvWrapper | 预期无需功能改动 | 原生 5 项→HIM 7 项；terminated/truncated 优先级及 final_obs |
| 自定义 ManagerBased 环境 | 预期无需功能改动 | 原生 stepping/reset；pre_pre_action 正确维护 |
| Runner | 可原样复用 | 同步 Torch tensor 环境接口、两个 optimizer 保存恢复、原日志计数 |
| W&B / TensorBoard | 可复用 | 独立 experiment/run，记录 Newton backend 和其物理差异；无重复 session |
| JIT / ONNX | 可复用 | 使用公共 Lab 观测/动作顺序，JIT 包含 estimator+actor |

原生 `ManagerBasedRLEnv.step()` 与 `compute_final_obs=True` 不限定 PhysX；
Newton asset/sensor 实现通过同一 manager 流程接入。
应保留现有 wrapper 从 extras["final_obs"]["critic"] 读取终止前监督目标的方式，
不将 reset 后 critic 当作 Estimator next target 或 timeout value。

本轮 CPU 测试验证的是算法与 mock 接口合同，不覆盖真实 Newton episode 终止。
下一阶段必须构造纯 timeout、trunk 接触、二者同时触发、部分环境 reset 和 UI reset 的真实集成用例。

### checkpoint 恢复与跨后端加载要分清

checkpoint 没有 PhysX 私有句柄；相同网络结构可直接严格加载。
现有 Runner 保存/恢复 Actor/Critic/Estimator、PPO optimizer、Estimator optimizer、
iteration/timesteps/time 与 W&B run ID。Newton→Newton 的同任务恢复可复用该机制。
环境本来就不保存物理状态和完整 RNG，不应声称可逐步重放原训练轨迹。

PhysX checkpoint 在 Newton 中加载是跨后端评估/微调，不是延续同一动力学实验。
新 Newton 正式 baseline 默认从随机初始化训练，使用 `go1_rough_pd_newton` 独立目录。

特别注意：当前 `runner.load(..., load_optimizer=False)` 仍恢复 iteration、timesteps 和 W&B ID，
它只跳过两个 optimizer；不能拿它直接实现全新的 weights-only 实验初始化。
如果后续另做迁移微调，需要单独严格加载 model_state_dict，并明确重置日志/optimizer/计数来源。
此功能不必为首版 Newton baseline 修改 Runner 核心。

## 9. 建议的文件改动清单

以下均为下一阶段计划，本轮尚未实现。

| 文件 | 建议变更 |
| --- | --- |
| 新增 `tasks/locomotion/robots/go1/newton_env_cfg.py` | 继承原 Go1PDEnvCfg/Go1PDPlayEnvCfg，替换 physics、设置 body order、两个 global-only scanner、明确 Newton DR/碰撞语义；不复制 reward/observation |
| `tasks/locomotion/robots/go1/__init__.py` | 新增 `Unitree-Go1-Velocity-PD-Newton` 和 `Unitree-Go1-Velocity-PD-Newton-Play`，原三个任务族保留 |
| `tasks/locomotion/agents/himloco_rsl_rl_cfg.py` | 新增仅覆盖 experiment_name 的 Go1PDNewtonPPORunnerCfg；超参数/seed/初始化不变 |
| 新增 `scripts/himloco_rsl_rl/train_newton.py` | 配置先行，`launch_simulation(env_cfg,args_cli)` 启动，复用现有 HIM Runner、CLI logger、保存路径/恢复机制 |
| 新增 `scripts/himloco_rsl_rl/play_newton.py` | 对应 Newton 可视化/生命周期，保留显式 velocity、checkpoint、环境数量、限步/实时选项 |
| 可选新增共享启动辅助文件 | 提取必要的训练/加载/日志函数供新旧入口共用；避免复制完整 train.py；每次提取都需原 PhysX 回归 |
| 可选新增 `mdp/newton_events.py` | 只有实际事件测试发现需额外适配才添加；不重写已工作的原生 mass/material/wrench 事件 |
| 新增 `validate_newton_go1.py` / `tests/test_newton_go1_contracts.py` | 分离 CPU mapping/config 合同与真实 Newton PD/传感器/终止/训练集成验证 |
| `validate_pd.py` / `validate_go1.py` | 保留现有 PhysX 验证；抽出可复用 assertions；不对 Newton 调用 get_dof_stiffnesses 等 PhysX raw view 方法 |
| Newton 文档与验证 artifacts | 保存完整 resolved 配置、版本、资产 hash、公共/native 名称、容量/DR 差异、固定条件指标 |

项目内上述路径的 Python task/module 均位于 `source/himloco_lab/himloco_lab/` 下；
scripts/tests 位于仓库根目录。算法文件 `him_ppo.py`、`him_estimator.py`、
`him_actor_critic.py`、`him_rollout_storage.py`、history 与 Runner 不列入必改清单。

配置示意，仅说明本机实际类型，**不是本轮已交付的可运行任务**：

```python
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg

# rough atlas：Newton 三角碰撞 + MJWarp 约束求解
cfg.sim.physics = NewtonCfg(
    solver_cfg=MJWarpSolverCfg(
        integrator="implicitfast",
        use_mujoco_contacts=False,
        # njmax/nconmax 按接触与约束峰值实测配置
    ),
    collision_cfg=NewtonCollisionPipelineCfg(),
    num_substeps=1,
)
cfg.sim.use_newton_actuators = False
cfg.scene.robot.joint_ordering = GO1_JOINT_NAMES
cfg.scene.robot.body_ordering = GO1_PUBLIC_BODY_NAMES
cfg.scene.height_scanner.global_world_only = True
cfg.scene.base_height_scanner.global_world_only = True
```

当前 train.py 尚无该后端预设/启动合同。不能把官方通用 CLI 的
`physics=newton_mjwarp` 直接加到旧 HIM 命令，就声称 Newton 已生效。
是否提供统一 backend preset CLI 可在新任务通过后再做，先避免扩大稳定 PhysX 入口的修改面。

## 10. 下一阶段实施顺序与验收门槛

| 阶段 | 工作与通过条件 |
| --- | --- |
| A：独立配置与启动 | 添加任务和配置驱动 launcher；打印实际 manager=NewtonMJWarpManager；Kitless headless 创建 1/4 环境；原 PhysX resolved config 不变 |
| B：控制与映射 | 名称唯一/全排列断言；q0/FK/符号、零动作与正负固定动作、饱和；每 0.005 s 记录 q_target/q/dq/computed/applied/backend effort；验证每 policy step 4 次 PD、内部 drive=0 |
| C：传感器与 rough | plane 扫描 z、187 点顺序、trunk/root frame、接触支撑力约 mg、真实 trunk 终止；再验证坡/噪坡/正反楼梯/障碍碰撞与 ray 一致、课程与越界终止，无退化平地替代 |
| D：完整 DR | 逐项打开材质、mass/inertia、COM、周期外力和推扰，验证求解器实际值与动力响应；审查首步/局部 reset、CUDA graph 和失效环境恢复；明确所有有意差异 |
| E：HIM 集成 | 32 环境真实 rollout+PPO update，45/270/238/12 维与 finite；Actor/Estimator 均实际更新，PPO 不反传 Estimator；真实 final_obs/timeout/history/smoothness 用例 |
| F：扩展与工作流 | 128 后 4096 环境至少 3 iterations；无 OOM/NaN/容量溢出；保存/恢复两个 optimizer 和日志步数；W&B offline/TB；Newton GUI Play、JIT/ONNX 与独立推理回归 |
| G：性能优化 | 单独评估 native PD、decimation capture、solver/碰撞容量和 substeps；逐 tick 力矩、接触采样/history/终止一致后再采用 |

控制测试先使用固定状态、固定指令、禁用噪声/随机化的诊断场景；完整 baseline 验收再恢复原 rough 和 DR。
禁止用关闭 termination、删除 height scan、删 reward、改变 HIM target 来绕过后端错误。

跨后端行为比较应固定实际状态和动作序列，不能仅依赖相同 seed；
使用实际世界状态统计速度跟踪、roll/pitch、存活、接触、足滑、机身/膝部高度、
能耗、joint error 与 saturation，而不只比较 reward 总和。
先在原 checkpoint 上做 PhysX→Newton Play 可辅助定位语义；运动好坏不是网络加载正确性的充分证据。

4096 环境在 PhysX 已成立，但 Newton 的接触/约束/三角配对和图捕获内存不同，
本轮没有测量 Newton 的显存、FPS 或学习收敛，不能预先保证相同规模或更高速度。
保持 PyTorch Manager-Based 是首选；不为追求性能重写为 Warp Env 或替换 RSL-RL 算法。

## 11. 本轮已执行的验证与复现方式

| 检查 | 本轮结果 | 证据范围 |
| --- | --- | --- |
| Git/环境/metadata/源码 SHA | 已读取，分支未切换 | 静态环境审计 |
| pip check | 7 项问题，详见第 2 节 | 依赖元数据；不是绿色通过 |
| Torch/Newton/Warp/MuJoCo Warp 导入 | 成功，GPU 被 Torch 检测到 | 不等于 Newton GPU kernel/solver 运行 |
| Go1PDEnvCfg / NewtonCfg 构造 | 成功；原 baseline 仍为 PhysxCfg | 无环境实例 |
| launcher scan | 内存 cfg 选择 Newton 后 needs_kit=False | 仅 scan，无 runtime 启动 |
| Go1 USD 展开读取 | 17 body、12 revolute、21 collider | 静态资产 |
| Newton USD parser | 成功，17 body/18 总 DOF/21 shape，惯量正定 | 无 finalize/solver/step |
| 原 PD checkpoint strict=True | 全部 keys 匹配 | 网络结构兼容 |
| CPU PyTorch vs 已有 JIT | batch 1/7，max abs error 均 0，输出有限 | 现有策略推理；不是 Newton rollout |
| HIM CPU 单元测试 | 7/7 通过 | history、quaternion、监督切片、梯度隔离、真实 CPU PPO/Estimator update、平均 loss、mock 终止合同 |
| PD CPU 单元测试 | 4/4 通过 | 原配置隔离、公平性、注册、CPU 力矩公式/裁剪 |
| Newton GUI/4096训练/真实 contact/reset/DR | 本轮未运行 | 下一阶段验收项 |

本轮实际执行的常用复核命令：

```bash
cd /home/sjc/himloco_lab
PY=/home/sjc/anaconda3/envs/himloco_lab/bin/python

git status --short
git branch --show-current
git log -6 --oneline
git -C /home/sjc/RL_Research/IsaacLab-3.0 describe --tags --always --dirty
"$PY" -m pip check
PYTHONDONTWRITEBYTECODE=1 "$PY" -m unittest discover -s tests -p 'test_him_contracts.py' -v
PYTHONDONTWRITEBYTECODE=1 "$PY" -m unittest discover -s tests -p 'test_pd_baseline.py' -v
```

USD parser 检查的等效复现代码如下，路径必须保持本机已缓存资产；此段不运行仿真：

```python
from pxr import Usd
from isaaclab_newton.physics.mjwarp_manager import NewtonMJWarpManager

usd = "/tmp/https/omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.1/Isaac/IsaacLab/Robots/Unitree/Go1/go1.usd"
stage = Usd.Stage.Open(usd)
builder = NewtonMJWarpManager.create_builder(up_axis="Z")
builder.add_usd(
    stage, root_path="/go1", load_visual_shapes=False, hide_collision_shapes=True,
    schema_resolvers=NewtonMJWarpManager._get_usd_import_schema_resolvers(),
)
print(builder.body_count, builder.joint_dof_count, builder.shape_count)
# 本机输出：17 18 21。不要把 18D 全部当成 policy joints。
```

上段使用本地 manager 内部 helper 来复现审计，不建议将 private helper 作为项目长期运行 API。

## 12. 风险排序与实施决策

1. **优先处理：观测看似 shape 正确但语义错误。** 包括射线命中自己、q/dq/action 排列、
   contact body ID 混用、root COM/link frame、终止前监督被 reset 观测替代。
2. **优先验证：rough 的真实碰撞几何。** 当前 atlas 不能自动视为 heightfield；
   容量不足、凸包替代或扫描/碰撞不一致都会影响算法监督与训练。
3. **重点验证：mass/COM 随机化实际生效与数值稳定。** 本机 COM 更新源码有处理，
   Lab 注释仍有警告；需要真实集成证据解决这个不确定性。
4. **控制差异：速度限位、armature、摩擦与接触求解。** 相同 PD 增益不意味着同一闭环动力学；
   所有必要的 Newton 专用差异独立记录，不改原 PhysX baseline。
5. **性能优化风险：原生 actuator 融合。** 力矩公式、遥测 ownership 和 sensor sampling 均需复核，
   不将训练更快作为数据正确性的替代。
6. **依赖维护风险：EA 固定版本与既有 pip 冲突。** 固定当前版本/SHA，按具体失败排错，
   不用 GA 环境要求覆盖已完成训练和部署的环境。

建议实施下一阶段；首版目标是“相同 HIM 算法与观测、正确显式 PD、完整 rough/DR 的 Newton baseline”，
保持原始 PhysX PD、Actuator Net、DeployV1、MuJoCo Sim2Sim 和实机部署版本可继续使用。

## 13. 主要本地依据

所有 Lab 源码路径相对于 `/home/sjc/RL_Research/IsaacLab-3.0/`，均为上述固定 EA SHA：

- `source/isaaclab/isaaclab/app/sim_launcher.py`：配置扫描与 launch_simulation。
- `source/isaaclab/isaaclab/envs/manager_based_rl_env.py`：原生 5 项返回值、decimation、final_obs。
- `source/isaaclab/isaaclab/assets/articulation/base_articulation.py`、`ordering_resolvers.py`：公共/native 排序。
- `source/isaaclab/isaaclab/assets/articulation/base_articulation_data.py`：COM/link 与 quaternion 合同。
- `source/isaaclab/isaaclab/actuators/actuator_pd.py`、`actuator_control.py`、`actuator_collection.py`：PD、限幅、drive 清零。
- `source/isaaclab/isaaclab/envs/mdp/events.py`：Newton 材质语义、质量/COM 更新。
- `source/isaaclab/isaaclab/terrains/terrain_importer.py`：混合地形的 heightfield 标记条件。
- `source/isaaclab_newton/isaaclab_newton/physics/newton_manager_cfg.py`、`mjwarp_manager_cfg.py`：实际配置字段。
- `source/isaaclab_newton/isaaclab_newton/physics/newton_manager.py`、`mjwarp_manager.py`：step、graph、sensor、solver reset。
- `source/isaaclab_newton/isaaclab_newton/assets/articulation/articulation.py`、`actuator_control.py`：Newton 状态/目标/wrench 提交。
- `source/isaaclab_newton/isaaclab_newton/sensors/ray_caster/newton_raycast_sensor.py`：BVH 与 global-only 查询。
- `source/isaaclab_newton/isaaclab_newton/sensors/contact_sensor/contact_sensor.py`：法向力/history/air time。
- `source/isaaclab_tasks/isaaclab_tasks/core/velocity/velocity_env_cfg.py`：Newton rough 参考 preset。
- `source/isaaclab_tasks/isaaclab_tasks/contrib/velocity/config/go1/rough_env_cfg.py`：官方 Go1 Newton 参数。
- `docs/source/how-to/prepare_asset_for_newton.rst`、`transfer_policies_between_physx_and_newton.rst`、
  `docs/source/concepts/joint_and_body_ordering.rst`、`solver-tuning/tune_mjwarp.rst`：与本地版本同 SHA 的文档。

另直接读取 Conda site-packages 的
`newton/_src/solvers/mujoco/solver_mujoco.py` 与
`newton/_src/actuators/controllers/controller_pd.py`，前者的惯性更新实现已校验 wheel RECORD。

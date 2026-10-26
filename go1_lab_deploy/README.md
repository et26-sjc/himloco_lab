# HIMLoco Go1 Lab PD 策略部署

本目录是独立发布包：以已经在同一台 Go1 验证过的 Gym 部署代码为基础，改为**直接生成 Lab 原生观测**，使用 Lab PD 最终策略。正常部署不导入 Isaac Lab、Isaac Gym、MuJoCo、训练 Runner 或 Critic。源 Gym 包和训练 checkpoint 未修改。

新策略已通过真实 Lab 状态离线回放与本机 CPU 推理验证；**尚未在 Go1 上运行本版本，也未在 Jetson 原镜像中加载本次新 JIT**。配置中的 IMU、物理通道及阈值记录继承自原机器人 Gym 部署，不能当作新策略的真机运动验证。

## 1. 使用的模型和目录

- 训练 run：`go1_rough_pd/2026-10-22_23-51-36_go1_pd_him_seed1`。
- checkpoint：`model_10000.pt`，训练 Git `73fc02aa59482bcd80d1b1d81f16ea1483d17571`。
- 选择依据：完整训练最终模型；没有另行进行最佳模型筛选。
- 本包 `models/policy.pt` 为原导出 JIT 的逐字节副本，内含 HIMEstimator + Actor，float32 `[1,270] → [1,12]`。
- JIT SHA256：`82d69e89c1a5759d003da86d056e1152e4c5317a3dc19b49cbb11c45b661c336`。
- checkpoint SHA256：`3e2ec748f7657ac502334ce69e1275780ecf4cf63251c8afad7b25f5615b29f3`。

```text
go1_lab_deploy/                 复制整个目录到 Go1
├── README.md
├── himloco.json                唯一默认运行配置，schema 2
├── deployment_contract.json    训练接口、来源、训练/部署差异
├── manifest.json               本发布全部静态文件摘要
├── models/policy.pt            完整 Estimator+Actor JIT
├── start_bridge.sh             终端 A：原 SDK bridge
├── start_policy.sh             终端 B：新 Lab 策略
├── scripts/                    环境检查、bridge 准备、公共路径
├── go1_lab_deploy/
│   ├── himloco/                观测、名称映射、历史、推理、状态机
│   ├── lcm_types/              与原包相同的四种协议
│   ├── unitree_legged_sdk_bin/  与原包相同的 C++ bridge/guard
│   ├── scripts/                离线检查、回放、打包工具
│   └── tests/himloco/          接口和控制回归测试
├── provenance/                 原包清单、实际训练 env/agent 快照
├── configs/evidence/           原机器人硬件接口记录，仅作来源保留
└── docs/                       验证结果与差异说明
```

## 2. 实际转换的接口

部署直接从 SDK 名称映射到 Lab 顺序，不使用 Gym 接口包装器，因此**不能再在外部重复重排或除一次 command scale**。

| 字段 | 原 Gym 部署 | 新 Lab 部署 |
| --- | --- | --- |
| 策略顺序 | FL/FR/RL/RR，每腿 hip/thigh/calf | hip/thigh/calf，每类 FL/FR/RL/RR |
| SDK 顺序 | FR/FL/RR/RL，每腿 hip/thigh/calf | 保留；通过真实名称双向映射 |
| 物理命令进观测 | `[vx*2, vy*2, wz*.25]` | `[vx,vy,wz]` |
| gyro / gravity | body gyro×.25 / body gravity | 相同尺度；SDK quaternion 继续按 WXYZ 解释 |
| 关节位置 | q−q0，Gym顺序 | q−q0，Lab顺序，q0按名称取得 |
| 关节速度 | dq×.05 | 相同；默认关节速度为0 |
| 观测裁剪 | 缩放后整体 clip | 每项 raw clip±100 后 scale |
| 上一动作 | clipped/applied | 正常运行是 raw policy output，观测时再clip |
| 目标处理 | raw action先clip±100再scale+q0 | **先q0+.25×raw，再clip绝对目标±100rad** |
| 硬件目标 | SDK绝对关节角 | 相同；不能再次乘action scale或加q0 |

Lab顺序：

```text
FL_hip FR_hip RL_hip RR_hip
FL_thigh FR_thigh RL_thigh RR_thigh
FL_calf FR_calf RL_calf RR_calf
```

实际名称均有 `_joint` 后缀。q0为左hip+.1、右hip−.1、前thigh .8、后thigh1、calf−1.5rad。

单帧为 `[commands3, gyro3, gravity3, q_rel12, dq12, previous_action12]`，六帧当前在前。reset清空过去帧；不把线速度真值或privileged observation送给Actor。逐项raw clip意味着gyro观测最终界为±25，dq观测最终界为±5。

## 3. 原样保留的控制层

- Kp40/Kd1、action scale .25、hip reduction1，Python 50Hz；bridge 500Hz转发位置与增益，MCU执行PD。
- `qd_des=0`、`tau_ff=0`，不在Python叠加一份外部PD力矩。
- 状态按同SDK tick和时间戳配对；重复包不能刷新watchdog。
- R2校准/接管、L2停止、过期、重复tick、倾角、跟踪、调度及锁存停止行为保留。
- 标定后6个实测帧热身、1秒动作渐入保留。它是部署接管流程，不等同于仿真reset立即全动作执行。
- 仅渐入alpha<1时，历史上一动作使用实际混合位置目标反算的无量纲动作；正常RUNNING回填raw。
- SDK bridge及C++ guard源码、四份LCM定义逐字节不改，可复用经摘要验证的原bridge二进制。
- 原附加目标步长停止仍为`null`；机械范围、跟踪、tilt、freshness及按钮保护不变。

**训练/执行差异**：Lab在每个.005s子步计算显式PD并限力矩。真机发送位置/Kp/Kd，由MCU与SDK执行，不能假定力矩/速度限制与PhysX逐值相同。训练hip/thigh effort23.7Nm、calf35.55Nm和速度30.1/20.06rad/s作为来源记录，不宣称SDK位置接口已经逐值强制这些界限。MuJoCo的额外passive damping参数不发送到真机，MCU的Kd仍是1。

## 4. 机械范围和已知起步差异

沿用原部署范围，不因Lab USD训练范围较宽而悄悄放宽：

| 关节 | Lab训练范围rad | 继承部署范围rad |
| --- | --- | --- |
| hip | [-.863,.863] | [-.802851,.802851] |
| thigh | [-.686,4.501] | [-1.047198,4.188790] |
| calf | [-2.818,-.888] | [-2.696534,-.916298] |

原生Lab的冷启动参考中，RL calf曾出现超出旧部署上界的**目标**：

- .7m/s：1024样本中12个，集中在step5～7，最大越界.143115rad；此指令高于默认硬件vx上限.3。
- .3m/s：2048样本中4个，集中在step5，目标约−.884155rad，超上界约.032143rad。
- 两组之后的样本没有该目标越界。它不等于实际关节已经运动到该角度。

回放的`passed`表示观测/JIT/目标转换一致，不表示机械目标全部可发送。真实部署按原六帧热身和渐入启动，运行状态与仿真冷启动不同，仍不能保证未来不会触发保护。Python会拒绝越界候选并先发送停止；新日志保存候选target、raw action、ramp、边界及history，便于定位。这里没有放宽限位、静默裁剪或修改策略。

## 5. 本机离线验证与重打包

在本仓库根目录、已安装的`himloco_lab`环境中：

```bash
python go1_lab_deploy/go1_lab_deploy/scripts/deploy_policy.py \
  --config go1_lab_deploy/himloco.json --check
python go1_lab_deploy/go1_lab_deploy/scripts/build_deployment_bundle.py \
  --verify go1_lab_deploy
PYTHONPATH=go1_lab_deploy python -m unittest discover \
  -s go1_lab_deploy/go1_lab_deploy/tests/himloco -v
```

实际完成79项Python测试、1项独立C++ guard测试；不创建真实LCM对象。原生Lab采集4env×256steps和4env×512steps，共3072个同步样本，部署代码独立CPU回放通过：

| 项目 | .7m/s 1024样本最大误差 | .3m/s 2048样本最大误差 |
| --- | --- | --- |
| 完整45D/270D（包括本工具上一输出闭环） | 4.65e−6 | 3.82e−6 |
| JIT action | 4.65e−6 | 3.82e−6 |
| Lab/SDK绝对target | 1.19e−6rad | 9.54e−7rad |
| SDK关节双向映射 | 0 | 0 |

command、gyro、q_rel、dq字段误差为0；gravity误差不超过5.97e−7。history中微小差值来自重力/前次推理浮点误差传播。CPU推理p95约.14～.20ms，仅代表本机x86 CPU，不代表Jetson速度。详细报告在`docs/validation/`。

主仓库可复现采集和回放：

```bash
python scripts/himloco_rsl_rl/record_deploy_reference.py --headless \
  --policy go1_lab_deploy/models/policy.pt --steps 512 --num_envs 4 --velocity .3 0 0 \
  --output artifacts/go1_lab_deploy/new_reference
python go1_lab_deploy/go1_lab_deploy/scripts/validate_replay.py \
  --config go1_lab_deploy/himloco.json \
  --reference artifacts/go1_lab_deploy/new_reference/reference.npz \
  --output artifacts/go1_lab_deploy/new_replay_report.json
```

采集需要主机Isaac Lab；回放和部署不需要。输出拒绝覆盖旧文件。若发布源码或配置有意更新，在开发机显式`--refresh go1_lab_deploy`后重新验证完整清单；正常启动只验证，不能自动刷新清单来掩盖改动。

生成另一份可移植发布目录：

```bash
python go1_lab_deploy/go1_lab_deploy/scripts/build_deployment_bundle.py \
  --config go1_lab_deploy/himloco.json --model go1_lab_deploy/models/policy.pt \
  --output /tmp/go1_lab_deploy_release
```

### 部署状态机的 MuJoCo 闭环检查

主机额外执行了原Controller的15秒接管测试：使用MuJoCo作为SDK反馈源，逐项生成真实q/dq、WXYZ和gyro，目标由同一部署Controller生成；不使用LCM或Robot SDK。

模拟标定阶段的人工支撑后，WAIT_R2→CALIBRATE→SETTLE→READY→RAMP→RUNNING全部通过，READY时3.16s释放夹具，RUNNING持续10.8s；命令.3m/s、平均vx约.239m/s，最大跟踪误差.394rad、最大tilt .0845rad，无保护触发/跌倒/NaN。夹具只固定WAIT/CALIBRATE/SETTLE期间的floating root，12关节始终真实动力学；这是理想化夹具测试，不是真机或严格无约束性能验收。

默认不模拟支撑的对照在.20s校准期触发原tracking保护，此时尚未执行JIT；报告一并保留，不能解释成策略运动失败。没有为通过测试放宽保护或修改PD参数。

```bash
python scripts/himloco_rsl_rl/validate_deploy_handover.py \
  --support-until-ready --velocity .3 0 0 --duration 15 \
  --output artifacts/go1_lab_deploy/handover_new.json
```

## 6. 复制到原Go1并先检查环境

复制整个本目录到新的`/home/unitree/go1_lab_deploy`，与原Gym包分开。目录已存在时另选新目录，保留原版本与日志。使用原已验证`jetson-model-deployment`镜像，不依据WTW旧Dockerfile重新搭环境。运行仅需要兼容的CPU Torch、NumPy和LCM；不安装训练框架。

在Go1上复用原bridge（将下列路径替换为本机实际已验证程序路径）：

```bash
bash /home/unitree/go1_lab_deploy/scripts/prepare_bridge.sh --reuse \
  /home/unitree/him_go1_gym_deploy/go1_gym_deploy/unitree_legged_sdk_bin/build/lcm_position_him
sudo bash /home/unitree/go1_lab_deploy/scripts/check_environment.sh
```

`prepare_bridge`只校验摘要/执行`--version`，不会打开机器人通信。若无法复用，在Jetson使用原SDK编译：

```bash
bash /home/unitree/go1_lab_deploy/scripts/prepare_bridge.sh --build \
  /home/unitree/unitree_legged_sdk-go1
```

`check_environment.sh`在原镜像中使用`--network none`离线加载新JIT，应显示`configuration_valid=true`、`model_loaded=true`、`motor_commands_published=false`。`deployment_ready=true`仅表示元数据门控通过，不是实机运动合格证明。本次JIT由Torch2.11导出，**不能用旧Gym JIT曾经能加载来替代本次检查**。若失败，保留原镜像并根据实际错误处理兼容导出，不直接重装仿真或升级Jetson全套环境。

## 7. 运行方式保持原两终端流程

完成原机器人相同的低层模式准备、支撑与原控制器退出后，在Go1的两个SSH终端执行：

```bash
# 终端A：保留bridge原手动Enter
sudo bash /home/unitree/go1_lab_deploy/start_bridge.sh
# 终端B：原镜像中新建临时容器him_go1_lab_policy，载入Lab JIT
sudo bash /home/unitree/go1_lab_deploy/start_policy.sh
```

沿用原按钮顺序：摇杆居中，第一次R2按下松开进入3秒标定；等待SETTLE→READY后，第二次R2进入1秒RAMP及RUNNING。默认速度上限仍为vx±.30m/s、vy±.10m/s、yaw±.15rad/s，yaw摇杆符号仍为−1。

L2各阶段请求停止，RAMP/RUNNING中的R2停止。先结束策略，再结束bridge；停止是零增益锁存请求，不保证保持站姿。没有添加自动启动、远程下发或绕过保护的参数。

新日志为`logs/him_lab_*.jsonl`，记录Lab顺序的q/dq、45D观测、raw/处理后/实际动作、raw反馈类型、渐入前目标及SDK绝对目标。默认不逐周期重复写全部270D；需要时可由逐帧观测重建，fault保留完整history。

源码来源、原硬件记录与新策略验证结论分开保存。本包没有连接Go1、发送LCM/UDP、运行MCU控制或重新训练策略。

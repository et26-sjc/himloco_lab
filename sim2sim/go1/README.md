# Go1 MuJoCo Sim2Sim：可调显式PD

2026-10-25，在既有`feat/go1-pd-baseline`分支实现。运行过程仅需要MuJoCo、NumPy、PyTorch、
PyYAML；Matplotlib用于绘图。**run/evaluate不导入Isaac Lab、HIM Runner、Critic或Unitree SDK。**
本轮没有新建分支、重新训练、修改原checkpoint或更改原训练/Play/导出/W&B代码。

## 1. 使用哪个策略

已训练PD run：

```text
logs/himloco_rsl_rl/go1_rough_pd/2026-10-22_23-51-36_go1_pd_him_seed1/
  model_10000.pt
  exported/policy.pt
  exported/policy.manifest.json
  exported/encoder.onnx
  exported/policy.onnx
```

- 训练提交：`73fc02aa59482bcd80d1b1d81f16ea1483d17571`。
- 10000轮、4,096,000,000 transitions；seed1，4096环境，100步/rollout。
- W&B project：`himloco-go1-baseline_lab`，run ID：`vzm57xqb`。
- checkpoint SHA256：`3e2ec748f7657ac502334ce69e1275780ecf4cf63251c8afad7b25f5615b29f3`。
- 实测JIT SHA256：`82d69e89c1a5759d003da86d056e1152e4c5317a3dc19b49cbb11c45b661c336`。
- 选择依据：**完整训练最终模型，不声称是最佳评估模型**。现有Runner按间隔/结束保存，
  没有独立评估并选best checkpoint的机制。
- 导出是本轮从原checkpoint单独生成并严格校验的，没有改写checkpoint。

完整记录：[pd_policy.json](../../artifacts/sim2sim/reference/pd_policy.json)。

现有Conda已安装Python3.12.15、Torch2.11.0+cu128、MuJoCo3.11.0、NumPy2.3.1、
PyYAML6.0.3、Matplotlib3.10.8，本次未安装新依赖。
其他机器可仅复制`sim2sim/`、JIT及相邻manifest，并准备`requirements.txt`中的依赖；
不需要原始checkpoint、Isaac Sim、Isaac Lab、CUDA GPU或整个训练项目。
`prepare_policy.py`和`audit_lab_contract.py`是可选开发工具，前者使用原HIM导出器，后者需要Lab。

## 2. 直接运行

在仓库根目录和既有`himloco_lab`环境执行：

```bash
source /home/sjc/anaconda3/etc/profile.d/conda.sh
conda activate himloco_lab
cd /home/sjc/himloco_lab
GO1_POLICY=logs/himloco_rsl_rl/go1_rough_pd/2026-10-22_23-51-36_go1_pd_him_seed1/exported/policy.pt

# GUI：默认PD40/1，真实策略闭环、跟随机身相机。
python sim2sim/go1/run.py --policy "$GO1_POLICY" \
  --velocity 0.7 0 0.3 --duration 30 --real-time

# 用户指定浮点增益；配置和有效12维gain都会写入结果。
python sim2sim/go1/run.py --policy "$GO1_POLICY" \
  --kp 20 --kd 0.5 --velocity 0.7 0 0 --duration 20 --real-time
python sim2sim/go1/run.py --policy "$GO1_POLICY" \
  --kp 30 --kd 0.75 --velocity 0.7 0 0 --duration 20 --real-time

# 单一指令headless；每次实验使用独立output目录。
python sim2sim/go1/run.py --policy "$GO1_POLICY" --headless \
  --kp 40 --kd 1 --velocity 0.7 0 0 --duration 20 \
  --output artifacts/sim2sim/my_pd40

# 五组指令：静止、.5前进、.7前进、原地转向、.7前进并转向。
python sim2sim/go1/evaluate.py --policy "$GO1_POLICY" \
  --kp 40 --kd 1 --duration 20 --output artifacts/sim2sim/my_pd40_suite

# 固定同一策略/模型/初态/指令，仅改变40/1、20/.5、30/.75。
python sim2sim/go1/evaluate.py --policy "$GO1_POLICY" \
  --pd-sweep --duration 20 --output artifacts/sim2sim/my_sweep
python sim2sim/go1/plot_results.py --summary artifacts/sim2sim/my_sweep/summary.json \
  --output artifacts/sim2sim/my_sweep/gain_comparison.png
```

`run.py`默认GUI，`--headless`关闭GUI，`--real-time`启用墙钟同步；正常结束、几何跌倒或关闭窗口时退出。
`--model`可指定兼容的Go1 torque MJCF；含position/velocity servo、额外actuator、非单位gear或错误力矩范围的模型会报错。
`--zero-action`为nominal姿态PD诊断，结果标明`mode=zero_action`，不能当作策略表现。

### YAML与逐关节增益

默认设置见[defaults.yaml](defaults.yaml)。CLI显式指定的值优先于YAML；没有CLI覆盖时使用YAML，
再使用dataclass默认值。支持全局浮点数、12维列表、按关节组或完整关节名映射：

```yaml
controller:
  type: pd
  kp: {hip: 40.0, thigh: 40.0, calf: 40.0}
  kd: {default: 1.0, FL_hip_joint: 0.9}
  action_scale: 0.25
  hip_reduction: 1.0
```

```bash
python sim2sim/go1/run.py --policy "$GO1_POLICY" --config my_config.yaml --kd 0.75
```

运行时打印有效配置及策略来源，JSON记录完整gain向量、JIT和checkpoint SHA256。
`--pd-sweep`固定声明的三组参数，因此不能同时传入`--kp/--kd`。

## 3. 模型来源及纯力矩执行

模型：[Google DeepMind Menagerie / unitree_go1](https://github.com/google-deepmind/mujoco_menagerie/tree/f3b6013629c7b94f8ee93fd977b161fa8350f9bc/unitree_go1)。
固定Git SHA：`f3b6013629c7b94f8ee93fd977b161fa8350f9bc`。
原始XML、scene、mesh、BSD-3-Clause LICENSE与SHA256清单保存在`models/unitree_go1/`。

原模型含`<position kp="100">`。本实现保留原文件供复核，另生成`go1_torque.xml`：

1. 删除所有position actuator及其default配置。
2. 添加12个`motor`，gear=1、无动态/偏置反馈，ctrlrange和forcerange均为关节侧Nm限制。
3. 移除原home keyframe中的位置控制目标；由代码以Lab默认站姿和根高度0.4m reset。
4. `scene_torque.xml`仅改变include目标，保留原始地面和视觉配置。

未修改模型惯量、质量、碰撞、被动阻尼、armature、frictionloss或foot contact参数以改善结果。
每次加载校验模型清单，并检查编译后的actuator gain/bias/dynamics和传动比，不会叠加内置position PD。
重建方法：

```bash
python sim2sim/go1/models/prepare_model.py \
  --source /path/to/pinned_menagerie/unitree_go1 \
  --revision f3b6013629c7b94f8ee93fd977b161fa8350f9bc
```

### 控制时序

每20ms：构建45D当前帧→加入270D历史→JIT输出12D action→更新q_target。
其间每5ms重新读取q/dq并计算：

```text
q_target = default_joint_pos + action_scale * action
tau_computed = kp * (q_target - q) - kd * dq
tau_applied = clip(tau_computed, -effort_limit, effort_limit)
data.ctrl[named_motor_ids] = tau_applied
mj_step(); mj_forward()
```

`mj_forward`同步积分后的transform/contacts，避免使用前一物理步的派生状态。
默认physics=200Hz、decimation=4、policy=50Hz，与实际PD训练配置一致。
hip/thigh限制23.7Nm，calf35.55Nm；hip_reduction仅按名字作用于`*_hip_joint`，默认1。

## 4. 观测与映射

默认`--profile lab`：

```text
commands(原始物理单位，scale=1)       3
body angular velocity ×0.25          3
projected gravity                    3
q - default_joint_pos               12
dq ×0.05                            12
previous raw action                 12
                                    45
```

遵守Lab逐term的raw clip[-100,100]再scale规则；action裁剪发生在绝对位置target[-100,100]。
history为`[current,past1,...,past5]`，reset全部清零，首次只有当前帧有效；上次action只在策略周期更新。
Actor没有输入真实线速度、地形高度或privileged observation；真实COM线速度仅用于评价指标。
JIT内部包含Estimator的3D速度/16D latent及Actor，直接270→12，不依赖Critic。

policy顺序：`FL/FR/RL/RR hip`，然后同顺序thigh、calf。
默认角：`[.1,-.1,.1,-.1, .8,.8,1,1, -1.5,-1.5,-1.5,-1.5]`。
实际MJCF按FR/FL/RR/RL逐腿排列，因此使用`mj_name2id`、`jnt_qposadr`、`jnt_dofadr`和actuator transmission建立映射。
本模型policy→qpos索引为`[10,7,16,13,11,8,17,14,12,9,18,15]`；代码不把该数组硬编码为索引假设。

MuJoCo free joint quaternion为WXYZ，但策略不直接使用四元数：
从`xmat[trunk]`取得机身到世界的R，`gravity_b=R.T@[0,0,-1]`，`omega_b=R.T@omega_w`。
COM Jacobian计算真实COM速度，避免link、COM主惯性轴或世界轴混用。

开发审计把**真实Isaac Lab状态**移植到MuJoCo，4环境×32步：
45D/270D最大误差均`4.77e-7`，JIT动作误差`1.31e-6`。
这验证的是相同状态下的语义一致性，不宣称两物理引擎的轨迹一致。

旧Gym可显式选择`--profile legacy`：逐腿policy/action顺序、command×[2,2,.25]、Gym scale后clip和raw action clipping。
JIT manifest必须标为legacy，防止把Lab策略误当旧策略。相关映射/clip有单测；
**本轮完整性能表使用新Lab PD策略，没有将旧Gym策略的表现混入。**

## 5. 真实参数扫描结果

五组指令均从相同nominal姿态开始，无额外预热、外力、噪声或随机化。
三组gain的初始状态SHA256、模型SHA256、policy/checkpoint SHA256均相同；每条件1次确定性episode。

### 默认PD40/1（20秒）

| 指令[vx,vy,wz] | 实际vx | 实际wz | Linear RMSE | Yaw RMSE | 平均base高度m | 存活秒 |
| --- | --- | --- | --- | --- | --- | --- |
| [0,0,0] | .001 | .005 | .011 | .010 | .255 | 20 |
| [.5,0,0] | .391 | -.051 | .191 | .130 | .262 | 20 |
| [.7,0,0] | .533 | -.066 | .247 | .177 | .251 | 20 |
| [0,0,.3] | -.007 | .106 | .032 | .201 | .267 | 20 |
| [.7,0,.3] | .524 | .143 | .252 | .280 | .254 | 20 |

默认五组均无几何跌倒，GUI前进转向另实际运行8秒。
**存在欠跟踪、偏航误差和机身偏低，不能宣称与Isaac Lab运动质量一致或sim2sim gap已解决。**
zero-action nominal PD单独运行4秒，平均高度0.294m、无跌倒、末2秒稳定站立比例1。

### Gain敏感性

| Kp/Kd | 完成20秒的条件 | 前进.5 / 前进.7 / 前进转向存活秒 | .7前进实际vx / RMSE |
| --- | --- | --- | --- |
| 40 / 1 | 5/5 | 20 / 20 / 20 | .533 / .247 |
| 20 / .5 | 2/5 | 1.515 / 1.285 / 1.275 | .391 / .474（跌倒前） |
| 30 / .75 | 5/5 | 20 / 20 / 20 | .427 / .348 |

20/.5的三个运动条件均触发base-height几何跌倒。静止虽未跌倒，平均base高度仅0.191m，
末2秒稳定站立比例0，因此不判为良好站立。低增益结果保留，没有调整action scale或模型补偿。
提前跌倒的RMSE只使用跌倒前样本，不能脱离存活时长直接比较优劣。

### 输出和指标定义

- `summary.json`：完整case/group结果、有效参数、policy来源和模型物理参数。
- `summary.csv`：便于对比的速度、RMSE、高度、存活、饱和等标量。
- 每case的`results.json`及本地`trace.npz`：200Hz位置、速度、roll/pitch、计算/施加力矩、
  q_target/q/dq、力矩饱和、四足接触/法向力、完整qpos/qvel；另有50Hz actor帧及action。
- Linear RMSE：`sqrt(mean((vx-cmdx)^2+(vy-cmdy)^2))`，Yaw RMSE同理，按每物理步统计。
- Joint tracking RMSE：每关节target与实际角度误差；torque RMS/peak/saturation按真实计算与裁剪统计。
- Foot contact：地面与指定足geom接触且法向力>1N，顺序FL/FR/RL/RR。
- Geometric fall：base高度<0.16m、|roll/pitch|>1rad、或trunk-ground法向力>1N，连续0.1s。
- Stable standing：零指令下高度>0.20m、|roll/pitch|<0.35rad、水平速度<0.15m/s、至少2足接触，
  报告末2秒满足比例；未终止不自动等价于稳定站立。
- 记录world displacement、水平路径长度、机身前向速度积分；坐标定义分别标明。
- 每case的fall rate为单次episode的0/1；跨五组指令的fall case比例不作为总体跌倒概率。

结果：[summary.json](../../artifacts/sim2sim/pd_sweep/summary.json) ·
[CSV](../../artifacts/sim2sim/pd_sweep/summary.csv) ·
[PNG图](../../artifacts/sim2sim/pd_sweep/gain_comparison.png) ·
[PDF图](../../artifacts/sim2sim/pd_sweep/gain_comparison.pdf) ·
[真实rollout帧](../../artifacts/sim2sim/pd_sweep/rollout_frame.png)。

## 6. 与Isaac Lab的物理差异

同机实际读取，而非根据旧版本假设：

| 参数 | Lab官方Go1 USD + PD任务 | 本次Menagerie |
| --- | --- | --- |
| 总质量 | 13.10045kg | 12.74345kg |
| hip/thigh/calf joint ranges | 同名关节范围一致（float精度内） | 相同 |
| 显式控制Kp/Kd | 40/1 | 40/1 |
| 内部position驱动 | 0，PD为外部力矩计算 | 完全移除，只有motor |
| 被动关节damping | 无MuJoCo式的额外1/2阻尼；Lab显式路径内部drive gains为0 | hip1、thigh/calf2 Nm·s/rad |
| armature | 0 | .01 |
| joint friction | 0 | frictionloss .2Nm |
| 额定/solver速度限制 | hip/thigh30.1、calf20.06rad/s | 不另加kinematic velocity clamp；记录超额比例 |
| Foot friction/contact | PhysX材质及其接触求解 | foot friction [.8,.02,.01]、condim6、软接触参数 |
| 自碰撞/碰撞几何 | 官方USD，配置禁用self collision | 原Menagerie碰撞和排除规则 |
| 接触求解 | PhysX GPU | MuJoCo原始模型求解设置 |

MuJoCo的被动damping是物理模型参数，**没有当作控制Kd，也没有为凑等效增益将它私自清零**。
质量/惯量、碰撞、摩擦和求解器等差异仍同时存在；当前对照不能把欠跟踪独立归因于其中某一项。
本次全部case的关节额定速度超限比例为0。

## 7. 测试、复现与开发工具

```bash
HIM_GO1_TEST_POLICY="$GO1_POLICY" python -m unittest discover -s sim2sim/go1/tests -v

# 重新校验/绑定现有JIT的checkpoint来源；不恢复optimizer、不修改权重。
python sim2sim/go1/prepare_policy.py \
  --checkpoint logs/himloco_rsl_rl/go1_rough_pd/2026-10-22_23-51-36_go1_pd_him_seed1/model_10000.pt \
  --policy "$GO1_POLICY"
# 新checkpoint需要导出时用 --output /path/to/export 替代 --policy。

# 可选跨引擎开发审计，需要现有Isaac环境，不训练。
python sim2sim/go1/audit_lab_contract.py --headless --policy "$GO1_POLICY"
```

实际通过：

1. 10项MuJoCo CPU测试：真实名称映射、拒绝position servo、XYZW/WXYZ边界下的姿态/重力、
   angular velocity坐标、blind actor无真实线速度、clip顺序、history/reset、YAML/CLI及group gains、
   PD40/1和20/.5及30/.75公式、正负裁剪、物理actuator_force/qfrc_actuator、200/50Hz和JIT有限输出。
2. 既有7项HIM数据/算法回归在Lab开发审计中通过，未对已训练策略运行PPO更新。
3. 4环境×32步真实Lab frame/history/action对照通过。
4. 五指令×三gain的完整headless实验及GUI8秒闭环通过；跌倒按实际结果保留。
5. 从`/tmp`使用`python -I`独立进程复跑.7前进20秒，20个trace数组全部逐值相同。
6. 真实MuJoCo270D历史的PyTorch/重载JIT/ONNX回归：batch1/7/64及256真实观测通过。
   重载PyTorch/JIT误差0；ONNX全部最大9.54e-6，真实观测最大2.86e-6、平均4.13e-7。
7. 原checkpoint与原有用户未提交文件SHA256未变；原训练、Play、checkpoint、W&B和导出源文件无diff。

审计报告：[lab_contract/validation.json](../../artifacts/sim2sim/lab_contract/validation.json)、
[export_regression/validation.json](../../artifacts/sim2sim/export_regression/validation.json)。
本机运行日志为`/tmp/himloco_mujoco_{tests,sweep,gui,zero,repeat,lab_contract,export_regression}.log`。

## 8. 文件职责、Git与边界

`config.py`协议/配置；`mujoco_adapter.py`模型、坐标、观测/动作映射；
`policy_runner.py`独立JIT/history；`controllers.py`逐物理步PD；`simulation.py`共享闭环/指标；
`run.py`GUI入口；`evaluate.py`批量gain对比；`plot_results.py`离线图；
`prepare_policy.py`校验/导出工具；`audit_lab_contract.py`可选开发审计；`models/`固定模型；`tests/`单测。

开发起点`73fc02a`已按用户要求推送至`origin/feat/go1-pd-baseline`，旧Actuator Net分支
`feat/go1-isaacsim61`保持`7c48d8a`。MuJoCo代码在当前PD分支本地提交，不自动推送；提交可用
`git log --oneline -3`查看。

本轮交付PD后端；可选官方Actuator Net MuJoCo后端尚未实现，未将PD20或PD40标作网络力矩推理。
没有MuJoCo训练、实机控制或Unitree通信。GUI能执行、控制闭环成立与高质量速度跟踪分别评价；
目前默认增益可稳定完成本次平地套件，但欠跟踪尚待后续模型差异实验处理。

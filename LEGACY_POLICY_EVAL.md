# 旧 Isaac Gym Go1 策略在 Isaac Lab 的迁移评估

## 证据与范围

本轮在既有 `himloco_lab` Conda 环境、Isaac Sim 6.1.0 / editable Isaac Lab 3.0.0、PhysX GPU 上运行。
本功能独立于正常训练/Play。旧 checkpoint 和正常 Go1 默认观测/动作配置均未修改。

证据分为：

1. **静态源码**：旧 `legged_robot.py`、`legged_robot_config.py`、`go1_config.py`、HIMActorCritic 与当前 manager 配置。
2. **确定性数值**：`audit_legacy_network.py` 严格加载及原源码网络对照；`tests/test_legacy_adapter.py` 的三项适配测试。
3. **真实仿真**：40环境×20秒×direct/adapted，平地与混合rough各一组；GUI为4环境×8秒、前进转向。

## 权重与网络

旧文件：`/home/sjc/HIMLoco-Go1/legged_gym/logs/rough_go1/Sep29_01-01-22_baseline_seed1/model_7500.pt`。
使用 `torch.load(map_location='cpu', weights_only=True)` 成功，无 pickle 兼容性降级。
`model_state_dict` 共30个float32参数张量；strict=True成功，missing/unexpected/shape mismatch均为空。
Actor/Critic隐藏层[512,256,128]；Estimator encoder为270→128→64→19（3速度+16 latent），target为45→128→64→16，prototype为32×16。
源码中的 RunningMeanStd/Normalization 类没有被 act_inference 调用，checkpoint也无运行标准化状态。
文件还包含两个optimizer的state_dict、infos=None，但推理只加载model_state_dict。
**内部iter=0，与文件名7500不同；本实验不据此恢复训练迭代。**

64×270相同随机输入，两套原/新源码网络严格加载同一权重，Actor最大/平均误差均为0；重复推理完全一致，3D velocity、16D latent、12D action均有限。
完整参数名、shape和dtype记录在 [network_audit.json](artifacts/round2/network_audit.json)。
SHA256：`2de4c315944b5db672c3af79c09a0d25867e37ba4c4e3c03ba503b322a3143a9`。

## 观测、动作与坐标语义

| 内容 | 原Isaac Gym | 正常Lab Go1 | adapted处理 |
| --- | --- | --- | --- |
| 单帧 | command3,ang_vel3,gravity3,q_rel12,dq12,last_action12 | 相同组顺序 | 45D组顺序不变 |
| command观测 | 实际[vx,vy,wz]×[2,2,.25] | 实际物理command | 乘[2,2,.25]，不改变物理command |
| ang_vel | body frame×.25 | COM角速度转入root link轴×.25 | 沿用；角速度不依赖参考点 |
| projected gravity | inverse(root quaternion)×[0,0,-1] | root link轴、XYZW | 沿用处理好的重力向量，不交换四元数 |
| q_rel | q-default，逐腿顺序 | q-default，逐关节类型顺序 | 先还原绝对角，再按名字排列并减旧default |
| dq | rad/s×.05 | rad/s×.05 | 按名字排列 |
| last_action | 旧raw action，逐腿顺序 | Lab raw action顺序 | 由上一绝对target逆变换；reset时明确为0 |
| 目标角 | old_default+.25×action，hip_reduction=1 | Lab_default+.25×action | 保持同名关节的绝对目标角 |
| 执行器 | PD，Kp40/Kd1，可能带训练随机化/延迟 | 官方ActuatorNetMLP | 保留官方网络，不修改控制器 |
| 初始root高度 | .42m | .40m | 保留Lab配置，属于动力学/初态差异 |

旧真实DOF顺序通过已安装的Isaac Gym实际加载训练URDF、使用训练AssetOptions后调用 `gym.get_asset_dof_names` 获得，**没有把URDF文本顺序当作运行顺序**。
证据：[legacy_asset_order.json](artifacts/round2/legacy_asset_order.json)。

```text
旧：FL_hip, FL_thigh, FL_calf, FR_hip, FR_thigh, FR_calf,
    RL_hip, RL_thigh, RL_calf, RR_hip, RR_thigh, RR_calf（均带_joint）
新：FL_hip, FR_hip, RL_hip, RR_hip, FL_thigh, FR_thigh, RL_thigh, RR_thigh,
    FL_calf, FR_calf, RL_calf, RR_calf（均带_joint）
旧→Lab action排列：[0,3,6,9,1,4,7,10,2,5,8,11]
Lab→旧 joint排列：[0,4,8,1,5,9,2,6,10,3,7,11]
```

默认角按**同名关节**一致：左hip+.1、右hip-.1，前thigh .8、后thigh1.0、calf -1.5rad；无符号或单位差异。
测试还人为改变default和action scale，验证适配器能够处理绝对目标偏移及其prev_action逆映射。
旧base与Lab trunk采用机器人前方+x、左方+y、上方+z；两套数学接口均把速度/重力转换到机体轴。
Lab根位置为link位置、root_lin_vel_b为COM速度；actor不直接接收线速度，角速度和重力转换不受COM参考点平移影响。
评价线速度使用Lab实际COM body velocity，距离另由世界root link位置计算，未混用缩放观测。

history都是[current,past1,...,past5]，共270D。适配器逐帧变换且保留全零padding。
本实验使用当前wrapper的干净reset：只填当前帧，清空过去帧，last_action=0。
旧源码reset_idx没有显式清空obs_buf/history，并且compute_observations使用self.actions；不复制该跨episode历史问题。旧BaseTask.reset还会执行一次zero-action物理step，而本实验使用Lab原生reset，没有额外启动step；属于已知启动瞬态差异。
评估仅统计每环境**首个episode**，因此该修复不被用来解释direct/adapted差异。

## 实际评价方法

固定seed=1，默认站姿、零初始速度/偏航，无观察噪声、外部推扰、质量/COM/摩擦随机化。
命令每步保持，physical command和policy command的缩放分开。direct/adapted之间reset，逐值断言初始root位置、速度和四元数一致。
平地每个command8环境；rough为seed固定的3×10混合地形、初始level0/1、difficulty0~.5，包含坡面、噪声坡面、楼梯及障碍，并非只选平地。
这些是一个seed的有限样本，没有多seed置信区间。

survival：首episode存活时长，20秒右删失；fall：原生base_contact terminated；termination包含truncated。
距离包括world x位移、body vx积分和水平路径；vx/wz为实际状态、按各环境有效episode取均值。
linear RMSE为sqrt(mean((vx-cmdx)^2+(vy-cmdy)^2))，yaw RMSE为sqrt(mean((wz-cmdwz)^2))，仅统计终止前有效步。
roll/pitch范围也只统计有效步。自定义EvaluationEnv在reset前捕获真实状态，避免把新episode姿态、速度、位置计入跌倒步。
所有结果含完整距离、姿态范围、动作峰值和clipping次数的JSON；表格为摘要。各组均未触发±100 action clipping。

### plane，20秒

| 模式 | command[vx,vy,wz] | 存活均值s | 跌倒率 | 实际vx | 实际wz | 线速度RMSE | yaw RMSE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| direct | [0, 0, 0] | 8.62 | 62.5% | -0.129 | -0.167 | 0.151 | 0.270 |
| direct | [0.5, 0, 0] | 2.32 | 100.0% | -0.122 | -0.006 | 0.831 | 1.624 |
| direct | [0.7, 0, 0] | 1.50 | 100.0% | -0.027 | -0.956 | 0.885 | 1.493 |
| direct | [0, 0, 0.3] | 1.55 | 100.0% | -0.230 | -0.178 | 0.372 | 0.830 |
| direct | [0.7, 0, 0.3] | 0.36 | 100.0% | 0.029 | -0.498 | 0.756 | 1.224 |
| adapted | [0, 0, 0] | 20.00 | 0.0% | -0.003 | 0.003 | 0.023 | 0.018 |
| adapted | [0.5, 0, 0] | 20.00 | 0.0% | 0.272 | -0.027 | 0.238 | 0.095 |
| adapted | [0.7, 0, 0] | 20.00 | 0.0% | 0.479 | -0.002 | 0.236 | 0.122 |
| adapted | [0, 0, 0.3] | 20.00 | 0.0% | -0.004 | 0.003 | 0.022 | 0.301 |
| adapted | [0.7, 0, 0.3] | 20.00 | 0.0% | 0.482 | 0.261 | 0.236 | 0.139 |

完整数据：[legacy_plane/results.json](artifacts/round2/legacy_plane/results.json)。

### rough，20秒

| 模式 | command[vx,vy,wz] | 存活均值s | 跌倒率 | 实际vx | 实际wz | 线速度RMSE | yaw RMSE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| direct | [0, 0, 0] | 2.67 | 100.0% | -0.038 | -0.083 | 0.397 | 1.064 |
| direct | [0.5, 0, 0] | 2.02 | 100.0% | -0.180 | 0.138 | 0.851 | 1.501 |
| direct | [0.7, 0, 0] | 1.55 | 100.0% | -0.041 | -0.771 | 0.892 | 1.505 |
| direct | [0, 0, 0.3] | 1.60 | 100.0% | -0.208 | -0.132 | 0.391 | 0.893 |
| direct | [0.7, 0, 0.3] | 0.36 | 100.0% | 0.053 | -0.539 | 0.735 | 1.233 |
| adapted | [0, 0, 0] | 20.00 | 0.0% | -0.003 | 0.003 | 0.023 | 0.021 |
| adapted | [0.5, 0, 0] | 14.73 | 50.0% | 0.282 | -0.019 | 0.243 | 0.123 |
| adapted | [0.7, 0, 0] | 12.94 | 62.5% | 0.500 | 0.027 | 0.251 | 0.189 |
| adapted | [0, 0, 0.3] | 20.00 | 0.0% | -0.004 | 0.003 | 0.021 | 0.301 |
| adapted | [0.7, 0, 0.3] | 15.23 | 62.5% | 0.469 | 0.249 | 0.268 | 0.175 |

完整数据：[legacy_rough/results.json](artifacts/round2/legacy_rough/results.json)。

## 结论及不能归因的部分

- 权重结构兼容，原/新网络数值相同；不是网络加载失败。
- direct的运动失稳包含明确的关节排列和command缩放错误，不能据此宣称物理sim2sim gap很大。
- adapted在本次平地20秒内可以站立、前进、边走边转向且无跌倒；vx=.5/.7指令实际约.272/.479m/s，存在欠跟踪。
- 纯yaw=.3时实际约.003rad/s，未实现可靠原地转向。粗糙地形前进相关指令跌倒50%~62.5%，未验证可靠通过地形。
- GUI真实运行：direct前进转向约.36s终止；adapted完成8s，平均vx=.487、wz=.268，无跌倒。见 [GUI结果](artifacts/round2/legacy_gui/results.json)。
- 语义适配后的剩余差异可能包含Actuator Net与旧PD、URDF/USD质量惯量/碰撞、摩擦、初态、接触及地形实现。上述第二轮尚未进行同USD切换PD的控制实验；下面补充2026-10-22的独立PD对照。
- 旧Gym相同command重测仍可进一步分离策略本身与物理模型影响；本轮未改reward或权重追求表面行走效果。

## 复现与文件

```bash
conda activate himloco_lab
python scripts/himloco_rsl_rl/audit_legacy_network.py \
  --checkpoint /home/sjc/HIMLoco-Go1/legged_gym/logs/rough_go1/Sep29_01-01-22_baseline_seed1/model_7500.pt
python -m unittest discover -s tests -p test_legacy_adapter.py -v
python scripts/himloco_rsl_rl/eval_legacy_go1.py \
  --checkpoint /home/sjc/HIMLoco-Go1/legged_gym/logs/rough_go1/Sep29_01-01-22_baseline_seed1/model_7500.pt \
  --headless --terrain plane --mode both --num_envs 40 --duration 20 --output artifacts/round2/legacy_plane
# rough：将 --terrain plane 改为 rough，并修改 --output。
# GUI：去掉 --headless，加 --velocity 0.7 0 0.3 --real_time，--num_envs 4 --duration 8。
```

旧资产顺序复核使用已有旧环境，不更换Lab仿真后端：

```bash
LD_LIBRARY_PATH=/home/sjc/anaconda3/envs/himloco/lib \
  /home/sjc/anaconda3/envs/himloco/bin/python scripts/himloco_rsl_rl/audit_legacy_asset.py \
  --output artifacts/round2/legacy_asset_order.json
```

新增 `inference/legacy_go1.py`、`utils/checkpoint.py`，以及上述audit/eval脚本和adapter测试。
项目入口的可选UI和环境wrapper改为按需导入，使纯张量网络检查无需启动Kit；任务仍照常注册。
仿真循环用no_grad而非inference_mode：本地Actuator Net每步roll重新分配缓存，inference tensor不能在下一次普通reset中原地修改；这是独立评估循环的执行上下文修复，不修改官方源码。
原始日志：`/tmp/himloco_network_audit.log`、`/tmp/himloco_legacy_{asset,plane,rough,gui}.log`。

## 2026-10-22：adapted旧权重切换PD40/1的控制实验

在`feat/go1-pd-baseline`上新增`--actuator actuator_net|pd`；默认仍为Actuator Net。
PD直接复用`Go1PDPlayEnvCfg`的`IdealPDActuator`，Kp40、Kd1、力矩上限23.7/35.55Nm。
旧`model_7500.pt`严格加载，原有command缩放、按名字的关节/动作映射、默认角及history适配器均未修改；
adapter的3项单元测试再次通过。没有重新训练或修改checkpoint。

关闭噪声、外力和随机化，其余沿用旧评估入口；两种执行器分别启动仿真。
结果新增根位置/速度/四元数以及关节位置/速度的初始状态SHA256，跨执行器逐项完全相同。
Actuator Net不被标注为某组等效PD增益；PD运行时检查12关节Kp=40、Kd=1。

### 平地，指令[1,0,0]

seed1、16环境、20秒，每环境首episode：

| 执行器 | 20秒存活率 | 实际vx m/s | linear RMSE m/s | yaw RMSE rad/s |
| --- | --- | --- | --- | --- |
| Actuator Net | 100% | 0.77293 | 0.26767 | 0.15808 |
| PD40/1 | 100% | 1.02979 | 0.09925 | 0.05914 |

PD的linear/yaw RMSE分别降低约62.9%/62.6%。两者均无action clipping。
另用PD在Kit GUI运行4环境、8秒：0跌倒，平均vx=1.01340m/s。
完整报告：[comparison.json](artifacts/legacy_pd_comparison/comparison.json)。

### Rough，与上次报告相同地形和命令

seed1、40环境、每指令8环境、20秒；3×10混合地形、difficulty0~0.5、初始level0/1。
与上次相同，包含坡面、带噪坡面、楼梯和障碍，不代表最高难度地形。
本次PD和Actuator Net复跑的六项初始状态SHA256全部相同。

| 指令[vx,vy,wz] | 上次Actuator Net存活率 | 本次Actuator Net复跑 | PD40/1存活率 | 上次/PD平均存活秒 |
| --- | --- | --- | --- | --- |
| [0,0,0] | 100% | 100% | 100% | 20.00 / 20.00 |
| [0.5,0,0] | 50% | 50% | 100% | 14.73 / 20.00 |
| [0.7,0,0] | 37.5% | 37.5% | 100% | 12.94 / 20.00 |
| [0,0,0.3] | 100% | 100% | 100% | 20.00 / 20.00 |
| [0.7,0,0.3] | 37.5% | 62.5% | 100% | 15.23 / 20.00 |

PD共40/40完成时限，无terminated/truncated；上次26/40，本次Actuator Net复跑28/40。
前进转向组Actuator Net复跑存在样本差异，保留两组结果，不把本次有限样本当作稳定概率估计。

| 指令 | 上次 / PD实际vx | 上次 / PD linear RMSE |
| --- | --- | --- |
| [0.5,0,0] | 0.282 / 0.452 | 0.243 / 0.144 |
| [0.7,0,0] | 0.500 / 0.680 | 0.251 / 0.118 |
| [0.7,0,0.3] | 0.469 / 0.678 | 0.268 / 0.130 |

前进转向时PD实际wz=0.269rad/s，指令0.3。原地转向仍较弱：
PD实际wz约0.107rad/s、yaw RMSE约0.213；上次约0.003rad/s、RMSE约0.301。
存活100%不等于所有指令跟踪都已解决。

这些结果支持：在已对齐语义后，执行器模型差异是旧策略迁移表现的重要影响因素。
这不是新PD策略训练结果，也不能单凭它确定Actuator Net拟合的是Kp20/Kd0.5。
仅测试一个seed、每指令8环境、20秒；尚未扩展到多seed、最高地形等级或随机化鲁棒性测试。
完整逐项结果：[rough_comparison.json](artifacts/legacy_pd_comparison/rough_comparison.json)。

### GUI与定量评估命令

旧Gym权重 + 已对齐adapted + PD40/1，平地GUI：

```bash
python scripts/himloco_rsl_rl/eval_legacy_go1.py \
  --checkpoint /home/sjc/HIMLoco-Go1/legged_gym/logs/rough_go1/Sep29_01-01-22_baseline_seed1/model_7500.pt \
  --mode adapted --actuator pd --terrain plane \
  --num_envs 4 --seed 1 --duration 20 --velocity 1.0 0 0 --real_time \
  --output artifacts/legacy_pd_comparison/pd_visual
```

粗糙地形GUI可将`--terrain plane`改为`rough`，例如指令`--velocity 0.7 0 0.3`。
完整五组rough headless评估：

```bash
python scripts/himloco_rsl_rl/eval_legacy_go1.py \
  --checkpoint /home/sjc/HIMLoco-Go1/legged_gym/logs/rough_go1/Sep29_01-01-22_baseline_seed1/model_7500.pt \
  --mode adapted --actuator pd --headless --terrain rough \
  --num_envs 40 --seed 1 --duration 20 --output artifacts/legacy_pd_comparison/pd_rough
```

对照时改成`--actuator actuator_net`并使用独立output目录。
实际日志：`/tmp/himloco_legacy_pd_vx1.log`、`/tmp/himloco_legacy_actnet_vx1.log`、
`/tmp/himloco_legacy_pd_vx1_gui.log`、`/tmp/himloco_legacy_pd_rough.log`、
`/tmp/himloco_legacy_actnet_rough.log`。

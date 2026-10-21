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
- 语义适配后的剩余差异可能包含Actuator Net与旧PD、URDF/USD质量惯量/碰撞、摩擦、初态、接触及地形实现。本轮没有同USD切换PD的控制实验，也未在旧Gym对同一组物理命令重测，不能把欠跟踪/原地旋转问题单独归因于Actuator Net。
- 下一步优先在旧Gym重复相同command，再在同Lab USD上做独立PD对照，以分离策略本身、执行器与物理模型影响；本轮未改reward或权重追求表面行走效果。

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

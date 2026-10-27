# Go1 PD 部署导向的温和训练调整建议

2026-10-26。**已按用户确认落实为独立DeployV1训练/Play任务，完成32环境×2iteration的工程验证；尚未启动正式训练。**
当前已成功部署的model_10000、原Go1 Actuator Net/PD任务、观测/动作接口和HIM算法保持原样。

## 已确认事实与尚待验证的归因

用户反馈：本Lab策略前进、横移较好，前进时后膝不再触地；后退仍会碰膝，纯转向比前进不稳定。
这是有效的真机观察，但当前返回日志没有机身高度、膝/小腿接触或外部速度真值，不能仅凭它唯一归因。

实际训练快照和源码确认：

- `base_height_l2.target_height=.30`、weight=-1。它惩罚trunk参考高度相对扫描地形均值的平方误差，
  不直接规定膝的位置，也不等于机身上表面距地高度。偏离5cm时该项在共同dt因子前只有-.0025，
  因此“有高度reward”不意味着高度被强制保持，更不能单独证明该项是碰膝主因。
  既往本项目原生Lab平地前进.7m/s的4环境×20秒记录，取t>2秒，trunk原点高度均值约.2705m、P5约.2664m；
  这进一步说明当前策略未严格维持.30m。它是仿真测量，不能替代真机测高。
- thigh/calf非足部接触reward没有启用；终止主要检测trunk接触，并非膝部接触即终止。
- `feet_height_body`目标为机身坐标z=-.20、weight=-.01，并乘足端水平相对速度。
  它不显式识别swing阶段，也不是离地20cm，更不直接约束膝净空。
- 默认前thigh=.8、后thigh=1.0rad；前后腿几何/负载与姿态偏置不对称，后退不等于前进轨迹简单反放。
  第一版仍保留这些默认角，避免同时改变动作offset和部署接口。
- `min_command_norm=.2`作用在`norm([vx,vy])`；<=.2时vx/vy一起置零，yaw不受此项影响。
  某一轴小于.2而另一轴较大仍会出现；这不是PPO/HIM的理论最低速度。
- `heading_command=True`且`rel_heading_envs=1.0`：原训练所有环境的yaw指令都由heading误差闭环生成，
  不能将配置里yaw范围[-2,2]理解为充分覆盖持续原地恒定转速。

更合理的工作假设是：当前奖励/采样对膝净空和低速/倒退/纯转向覆盖不足，加上真实关节跟踪滞后、
模型/地面/负载差异，使已有较小的仿真净空进一步缩小。训练改动可改善策略裕度，但不替代动力学核对。

## DeployV1初值（已实现）

使用独立任务`Unitree-Go1-Velocity-PD-DeployV1`，**不覆盖当前成功baseline**。HIM-PPO/Estimator、45/270/238维观测、
Kp40/Kd1、action scale .25、q0、.005s×4、地形、随机化和PPO超参数均保持。

| 项目 | 当前 | 建议V1 | 目的 |
| --- | --- | --- | --- |
| base_height目标 | .30m | **.32m** | 先增加2cm机身/膝部裕度，不直接跳到.35或.36 |
| base_height权重 | -1.0 | **-2.0** | 适度增加高度约束，仍需检查实际高度分布 |
| thigh/calf接触惩罚 | 未启用 | **weight=-.5，threshold=1N** | 复用官方undesired_contacts，先排斥非足部承重/擦地 |
| min_command_norm | .20m/s | **.10m/s** | 增加.1～.2附近非零平移命令；.15可作为更保守对照 |
| rel_heading_envs | 1.0 | **.5**，仍保留heading_command=True | 一半保持heading训练，一半接收持续采样yaw |

新增接触项时使用`SceneEntityCfg('contact_forces', body_names=['.*_thigh', '.*_calf'])`，
实际解析必须恰好匹配8个非足部刚体，确认没有选入foot。1N是接触事件阈值，不是力矩限制。
本地Lab3官方函数从`net_normal_forces_w_history.torch`取各刚体历史力模长最大值，超过阈值的刚体数作为惩罚；
它不是力越大无限加罚的函数，也不区分接触地面/地形/其他刚体。上坡、楼梯偶发擦碰应由适度reward处理，
第一版不将其升级为立即termination。若粗糙地形明显退化，可先将权重减至-.25；不要首先大幅加至-5/-10。

**必须验证该项真的会触发**：正常平地站立为0；人为构造小腿触地时计数上升；检查各腿分项。
若仿真中小腿从不接触，该项始终为0，就不能靠它教会“距地还差2cm”的裕度，仍需评估几何净空和实际高度。
此时再考虑小的净空奖励或进一步模型对齐，而非不断增大一个永远为0的惩罚权重。
若新训练的高度分布几乎不变，可将高度权重单独从-2比较到-3；先检查实际分布，不连续抬高目标值。

第一版不改`feet_height_body`（目标-.20、权重-.01），不同时加大抬脚、动作平滑或能耗惩罚。
需要注意：机身高度增加2cm且机身坐标足目标不变时，平地世界坐标中的目标足高也约增加2cm；
该reward并非严格世界坐标轨迹。若后续发现抬腿折叠加剧，可单独比较足目标-.22、权重不变，
使近水平姿态下绝对足高接近原设定。不要先把抬脚reward直接删除，rough terrain仍需净空。

## 低速与转向：避免只改门限就宣称解决

降低门限只增加可采样区域。当前vx/vy主要在[-1,1]二维均匀采样，.1～.2m/s合速度样本仍只占少量；
且线速度tracking的std=.5m/s，对小误差比较宽：.1m/s指令而完全不动时，单个tracking项约为
`exp(-.1**2/.5**2)=.961`。因此低速“不动也得较高tracking reward”是另一项需要观察的因素。
第一轮先不同时缩紧tracking std；若低速仍弱，第二轮优先改善命令覆盖。

最小的第二步是从原采样中划出少量明确工况，如各10%的低速直行/倒退、纯转向：

- 直行/倒退：vy=wz=0，vx覆盖正负.1～.6，确保负速度和低速非零确实进入训练。
- 纯转向：vx=vy=0，wz覆盖正负.2～.8；这部分应关闭heading覆盖，避免命令被再次改写。
- 其余样本保留原混合速度与rough terrain。确有转速欠跟踪时，再比较yaw tracking weight .5→.65。

这是命令分布调整，不需要换算法、增加估计器或加入MPC，也不应在部署端强制将小摇杆命令跳成最低速度。
部署摇杆deadzone=.05和训练的平移norm门限是不同机制；本轮同步的真机包仍保持原线性摇杆处理。

## 对照和验收

先分别验证：现有baseline、仅新增contact项、contact+高度、再加入低速/heading调整；资源有限时也可先训练完整V1，
但不把运动改善唯一归因于某一个改动。保持seed、训练时长和地形一致；正式对照采用随机初始化。
可先从当前成功checkpoint短程微调验证方向，但应标为warm-start实验，与从零训练结果分开。

在Lab、MuJoCo和后续真机分别检查：

- 前后/横向低速：vx或vy=±.1、±.15、±.3、±.6；纯yaw=±.2、±.4、±.6，以及组合命令。
- 按指令分组的速度RMSE、存活、trunk高度均值/P5、roll/pitch和小腿/大腿接触率。
- 膝部几何净空P5/最小值、实际/目标关节差、力矩饱和、机械范围拒绝次数；reward总分不能替代这些指标。
- rough terrain能力和静止抖动不能明显退化。高度不能只看目标参数，应看实际分布。

固定低速评估需确认实际送给策略的命令值。本项目当前Play的`--velocity`路径仍保留训练采样器门限，
旧任务的小命令可能被清零。新`Unitree-Go1-Velocity-PD-DeployV1-Play`已将min_command_norm设为0，
用于固定低速命令评估，并保持heading_command=False；训练任务门限仍为.10。旧任务Play与训练默认值保持原样。

依据：本项目`source/himloco_lab/himloco_lab/tasks/locomotion/velocity_env_cfg.py`、Go1配置、
`mdp/rewards.py`、`mdp/commands/commands.py`及实际`go1_lab_deploy/provenance/training_env.yaml`。
Isaac Lab接触数据说明：[官方ContactSensor文档](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.sensors.html)。
以上数值是温和的起始配置，不代表已收敛或已在真机验证的新策略。


## 已执行验证与正式启动

实现文件为`robots/go1/deploy_v1_env_cfg.py`，共享原PD物理/观测/算法和rough terrain；
训练相对原PD任务严格只有五处变化：height目标、height权重、非足部contact项、平移norm门限、heading环境比例。
Runner只改变experiment_name为`go1_rough_pd_deploy_v1`，PPO超参数与网络完全继承。

实际PhysX验证32环境：

- 原PD/Actuator Net配置实例化前后不变，单帧/Actor历史/Critic/Action为45/270/238/12。
- 新contact项恰好匹配8个thigh/calf，不包含foot；正常站立计数0。
- 真实降低机器人产生碰撞后计数最大6，直接与PhysX normal force history对照一致，未伪造sensor buffer。
- heading与直接yaw两组都存在；Kp40/Kd1、200Hz物理/50Hz策略频率保持。

实际rough terrain训练32env×2iteration：6400步，随机初始化、TensorBoard日志，checkpoint`model_2.pt`保存并以weights_only=True读回；
PPO与Estimator独立optimizer均40步，权重、optimizer状态和loss有限。新接触reward已出现在日志中。
这是工程smoke，不是运动质量或收敛证明；本轮没有执行4096env正式训练。

报告：`artifacts/go1_deploy_v1/validation.json`及`training_smoke.json`。

正式训练（从随机初始化，不加resume）：

```bash
conda activate himloco_lab
WANDB_PROJECT=himloco-go1-baseline_lab WANDB_MODE=online \
python scripts/himloco_rsl_rl/train.py \
  --task Unitree-Go1-Velocity-PD-DeployV1 --headless \
  --num_envs 4096 --seed 1 --max_iterations 10000 \
  --logger wandb --run_name go1_pd_deploy_v1_seed1 \
  --wandb_group go1-deploy-v1 --wandb_tags go1 isaaclab3 physx pd deploy_v1
```

原命令的脚本和参数接口保持；**task要换成新名称**，继续用原PD/Actuator Net任务会训练原配置。
日志根目录为`logs/himloco_rsl_rl/go1_rough_pd_deploy_v1/`，默认不恢复旧权重。

新任务Play：

```bash
python scripts/himloco_rsl_rl/play.py \
  --task Unitree-Go1-Velocity-PD-DeployV1-Play --num_envs 16 \
  --checkpoint /absolute/path/to/new_run/model_10000.pt --velocity 0.1 0 0
```

heading混合只影响训练指令生成：50%环境根据heading误差计算wz、另外50%使用采样wz，
Actor仍只接收同样的物理速度命令。Play和真机仍直接给vx/vy/wz，没有新增heading输入或部署闭环。
未来使用新权重时，按原导出/打包流程更新JIT及其来源摘要、config/manifest即可；本轮成功部署的旧包未修改。

## 完整训练后的实际结果（2026-10-27）

用户已完成10000iteration。当前配对平地测试确认机身和后膝原点高度提高，但中低速后退与纯转向存在跟踪退化；
不能将本版称为所有工况都改善，实机碰膝仍未验证。实际数据、模型路径和可视化命令见
[DeployV1完整训练评估](GO1_DEPLOY_V1_EVALUATION_20261027.md)。

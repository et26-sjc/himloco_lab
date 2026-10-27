# V1偏航漂移与checkpoint复核（2026-10-27）

用户确认观察来自Isaac Lab Play：机头逐渐转偏、路线走弧线。未改变任何训练参数或部署控制。

## 可确认的机制与归因边界

Play给定wz=0表示目标角速度为0，并不自动保持初始绝对航向。旧PD和V1的Play均关闭heading生成，
因此不能把新旧差异归因为“这次Play才关了heading”。训练heading比例1→.5只改变采样分布，
不改变部署输入/流程；是否导致这次偏航偏置，需要单变量消融，现有五项同时变化不足以确认。

本地yaw tracking为`.5*exp(-(wz_cmd-wz_actual)^2/.5^2)`。wz误差.02rad/s时，
只比最大值少约.000799，但持续18s可累积约20.6度；平均reward接近上限并不能说明直线保持良好。
之前将height惩罚与tracking奖励上限比较，只能说明总量级，不能据此排除小偏差修正被牺牲的可能。
`flat_orientation_l2`只惩罚projected gravity的xy分量，即roll/pitch，不能用于修正yaw漂移。

更准确的结论是：V1提高了几何裕度，同时带来部分运动跟踪取舍；不能把原因单独归于-2、heading比例或sim2real。
本次现象在原生Lab已出现，应先在训练/模型选择层核对。

## 不重训的模型选择检查

原PD和V1的8000/9000/9500/10000在同一平地场景下各测试9组命令×4个名义复制环境×20s，
初态SHA相同，统计剔除前2s。新增直接记录root世界位置和连续yaw，避免仅用平均速度判断走弧线。
全部测试完成horizon，但这是名义工况，不是多随机seed鲁棒性评估。

| 模型 | +.6前进18s累计转偏绝对值 | 前进横向偏离绝对值m | -.3实际vx | -.3后膝原点P5 cm | .3转向实际wz |
| --- | --- | --- | --- | --- | --- |
| PD baseline | 19.3° | 1.57 | -0.262 | 3.6 | 0.298 |
| DeployV1 8000 | 7.6° | 0.34 | -0.172 | 7.2 | 0.262 |
| DeployV1 9000 | 3.5° | 0.45 | -0.227 | 6.5 | 0.254 |
| DeployV1 9500 | 6.9° | 0.19 | -0.245 | 6.5 | 0.256 |
| DeployV1 10000 | 25.2° | 2.19 | -0.186 | 8.2 | 0.223 |

横向偏离是直接root位置差投影到t>2s首帧航向的法向，不是速度积分估计。
后膝原点高度仍不是碰撞表面净空。

9500在本组条件下是更均衡的预览候选：前进转偏比10000小，倒退速度更接近指令，
保留部分后膝高度收益；并非全部指标都优于10000，低速倒退依然很弱，-.3倒退的横向偏移仍存在。
9000虽+.6偏航更小，body横移偏置和其他速度工况更差，不能仅按一个转偏角挑模型。
8000的-.1倒退后膝高度P5仅约1cm，也不宜据较直的快速前进认定全面改善。

## 先可视化9500

```bash
conda activate himloco_lab
GO1_V1_RUN=logs/himloco_rsl_rl/go1_rough_pd_deploy_v1/2026-10-26_23-18-08_go1_pd_deploy_v1_seed1
python scripts/himloco_rsl_rl/play.py \
  --task Unitree-Go1-Velocity-PD-DeployV1-Play \
  --checkpoint "$GO1_V1_RUN/model_9500.pt" --num_envs 8 --seed 1 \
  --velocity 0.6 0 0 --real-time
```

将velocity改为`-.3 0 0`核对倒退自然程度；再比较`.3 0 0`、`-.6 0 0`、`-.1 0 0`。
正常Play默认rough terrain，与本次平地定量条件不同。
各候选JIT导出在新run的`exported_checkpoints/model_<iter>/`，未改checkpoint或已验证的真机部署包。

如仍需继续训练，先做单变量诊断：只恢复rel_heading_envs=1.0，保持V1其余值，与同初始checkpoint/相同时长对照。
这是用于检验命令分布的影响，不是宣称恢复比例一定解决漂移。之后再独立测试明确的纯前/后直行样本
（vy=wz=0，且不被heading生成覆盖），优先补倒退覆盖；不同时继续提高高度/碰撞权重。
需要更严格yaw tracking时另做std或权重对照，不把多个未经验证的改动混成一个新结论。

[全部数值](../artifacts/go1_deploy_v1/checkpoint_scan_20261027/comparison.json)、
[轨迹图](../artifacts/go1_deploy_v1/checkpoint_scan_20261027/trajectories.png)。
heading定义与本地源码核对一致：[Isaac Lab官方command说明](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.envs.mdp.html)。

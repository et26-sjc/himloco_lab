# DeployV1 完整训练后的配对评估（2026-10-27）

结论：**机身和后膝几何裕度改善，部分速度跟踪退化，未证明真机碰膝已经解决。**

## 模型与检查

新run：`logs/himloco_rsl_rl/go1_rough_pd_deploy_v1/2026-10-26_23-18-08_go1_pd_deploy_v1_seed1/`。
已完成10000iteration、4096000000步，最终checkpoint权重有限。使用`model_10000.pt`，
这是最终模型，不是通过独立评估筛选的最佳模型。旧PD对照为`2026-10-22_23-51-36_go1_pd_him_seed1/model_10000.pt`。

已通过严格checkpoint加载，导出新JIT到新run的`exported/policy.pt`，batch1/7/64与原PyTorch输出最大误差0。
新JIT SHA256：`077aa65a43d11365673e161d83b5cf8a57210bac9d2335627fcc0998cfd3df2a`。
旧部署包JIT未替换，也未连接机器人。

## 原生Lab同条件测试

使用同一冻结的平地场景、同seed与名义初始q/dq/root状态（SHA匹配）、无观测噪声、无随机化/推力、
Kp40/Kd1、.005s×4，每命令4个平移复制环境、20秒，丢弃前2秒计算稳定段统计。
Actor都是确定性JIT推理，直接显式注入物理命令，绕过训练采样的低速清零。
两版每个命令均4/4完成horizon；未检出>1N的选定thigh/calf接触。
这不是随机鲁棒性置信区间，也不覆盖rough terrain或真实机器人。

| 命令[vx,vy,wz] | 旧/新实际vx | 旧/新实际wz | 旧/新机身均高cm | 旧/新后膝原点高度P5 cm | 旧/新线速度RMSE |
| --- | --- | --- | --- | --- | --- |
| [0.0, 0.0, 0.0] | 0.004 / -0.000 | 0.018 / 0.003 | 26.3 / 28.0 | 7.9 / 9.6 | 0.005 / 0.000 |
| [0.3, 0.0, 0.0] | 0.316 / 0.293 | 0.023 / -0.020 | 26.5 / 28.1 | 3.8 / 6.5 | 0.025 / 0.039 |
| [-0.3, 0.0, 0.0] | -0.262 / -0.186 | 0.031 / -0.006 | 25.1 / 28.0 | 3.6 / 8.2 | 0.048 / 0.121 |
| [0.6, 0.0, 0.0] | 0.609 / 0.597 | 0.019 / -0.025 | 27.0 / 28.1 | 3.5 / 5.7 | 0.033 / 0.033 |
| [-0.6, 0.0, 0.0] | -0.561 / -0.526 | 0.028 / 0.006 | 25.3 / 27.7 | 5.6 / 8.2 | 0.054 / 0.077 |
| [0.0, 0.3, 0.0] | 0.034 / 0.020 | 0.015 / -0.021 | 26.3 / 27.9 | 5.0 / 8.7 | 0.070 / 0.048 |
| [0.0, 0.0, 0.3] | -0.007 / 0.011 | 0.298 / 0.223 | 26.4 / 27.8 | 5.4 / 7.8 | 0.035 / 0.028 |
| [0.1, 0.0, 0.0] | 0.090 / 0.087 | 0.019 / -0.010 | 26.3 / 28.2 | 5.8 / 7.7 | 0.025 / 0.039 |
| [-0.1, 0.0, 0.0] | -0.036 / -0.009 | 0.023 / -0.028 | 25.4 / 28.0 | 3.1 / 8.0 | 0.080 / 0.093 |

后膝是RL/RR calf刚体原点（膝关节）的世界高度，P5为较低5%分位，**不是碰撞外表面离地净空**。
原模型在这些平地条件下也没有腿部>1N接触，不能据此声称已消除实机碰膝。

新策略站立漂移减少，横移RMSE从.070改善至.048；.6m/s前进跟踪基本保持。
-.3m/s后退虽然膝原点高度P5从约3.6cm升至8.2cm，速度却从-.262降至-.186m/s。
-.1m/s后退仅约-.009m/s，低速倒退仍很弱。纯yaw .3rad/s的实际转速从.298降至.223rad/s。
因此门限下调和heading混合没有自动带来这些工况的改善，不能用训练平均reward掩盖局部退化。
五项参数和训练分布同时变化，尚未做消融，不能将速度退化单独归因于height权重-2。

## 独立MuJoCo

同一固定Menagerie模型、PD40/1、额外passive damping0、同初态，4种命令各20秒，丢弃前2秒。
两版全部完成，无几何跌倒。-.3m/s后退时机身均高25.8→27.9cm，后膝原点P5 4.7→7.8cm，
实际vx却从-.181→-.116m/s；纯yaw .3的实际转速从.089→.059rad/s。
也是增加几何裕度但运动响应更保守；MuJoCo结果不能替代真机结果。

额外用胶囊体支撑函数计算后小腿非足部collision geometry对平面的高度，并排除四个foot sphere；
该指标对应固定Menagerie碰撞近似，不是Go1真实外壳。几何接触统计不等于Lab的>1N力阈值统计。

## 可视化与复现

```bash
conda activate himloco_lab
GO1_V1_RUN=logs/himloco_rsl_rl/go1_rough_pd_deploy_v1/2026-10-26_23-18-08_go1_pd_deploy_v1_seed1
python scripts/himloco_rsl_rl/play.py \
  --task Unitree-Go1-Velocity-PD-DeployV1-Play --num_envs 8 --seed 1 \
  --checkpoint "$GO1_V1_RUN/model_10000.pt" --velocity -0.3 0 0 --real-time

python sim2sim/go1/run.py --policy "$GO1_V1_RUN/exported/policy.pt" \
  --kp 40 --kd 1 --passive-damping 0 --velocity -0.3 0 0 \
  --duration 30 --real-time --output artifacts/sim2sim/deploy_v1_backward_view
```

依次将velocity改为`.6 0 0`、`-.6 0 0`、`0 .3 0`、`0 0 .3`、`-.1 0 0`。
Lab正常Play默认仍是rough terrain，和上面的平地定量场景不同。旧策略低速对照不可直接沿用旧Play采样门限，
本轮配对工具已显式给定原始物理命令，确保低速命令确实进入Actor。

```bash
BASE_POLICY=logs/himloco_rsl_rl/go1_rough_pd/2026-10-22_23-51-36_go1_pd_him_seed1/exported/policy.pt
NEW_POLICY="$GO1_V1_RUN/exported/policy.pt"
python scripts/himloco_rsl_rl/compare_go1_policies.py --headless \
  --baseline "$BASE_POLICY" --candidate "$NEW_POLICY" --output artifacts/go1_deploy_v1/native_new
python sim2sim/go1/compare_motion.py \
  --baseline "$BASE_POLICY" --candidate "$NEW_POLICY" --output artifacts/go1_deploy_v1/mujoco_new
```

完整JSON：[原生Lab](../artifacts/go1_deploy_v1/evaluation_20261027/native_lab_final/comparison.json)、
[MuJoCo](../artifacts/go1_deploy_v1/evaluation_20261027/mujoco/comparison.json)、
[对照图](../artifacts/go1_deploy_v1/evaluation_20261027/motion_comparison.png)。
本轮没有重训、改动新旧checkpoint或替换已验证的真机部署包。

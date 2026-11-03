# Go1 PD DeployV1 参数存档

DeployV1 是从标准 `Unitree-Go1-Velocity-PD` 派生的独立实验，不是当前标准 PD 训练配置。本文件仅保存配置差异，**不会注册任务或改变训练行为**。完整代码、训练评估和 Newton 审计保存在本地 `archive/go1-deploy-v1` 分支（原提交 `470468b`）；Newton 后续研究保存在 `feat/go1-newton`。

| 项目 | 标准 PD baseline | DeployV1 训练 |
|---|---|---|
| 机身高度目标 | 0.30 m | 0.32 m |
| `base_height_l2` 权重 | -1.0 | -2.0 |
| 大腿和小腿接触惩罚 | 未启用 | `undesired_contacts`，权重 -0.5，阈值 1 N；body 匹配 `.*_thigh`、`.*_calf` |
| 平移命令范数死区 `min_command_norm` | 0.20 m/s | 0.10 m/s |
| `rel_heading_envs` | 1.0 | 0.5，保留 heading 功能 |
| 实验名称 | `go1_rough_pd` | `go1_rough_pd_deploy_v1` |

接触惩罚针对接触刚体数量，并非连续接触力惩罚；足端未被选入。命令范数死区作用于二维平移命令，不是独立作用于每个轴，也不改变 yaw。DeployV1 Play 额外设置 `heading_command=False`、`min_command_norm=0.0`，以支持直接指定包括低速在内的速度命令。

其余关键条件继承标准 PD：Go1 USD、PD Kp=40/Kd=1、动作尺度 0.25、physics dt=0.005 s、decimation=4、地形与随机化、观测、HIM-PPO、Estimator 和 PPO 超参数。高度和接触权重是奖励项，并非强制保证机身高度或膝部净空。

DeployV1 已完成一轮 10000 iterations 训练。用户观察到整体尚可，但后退不自然，前进偏航漂移较原 baseline 明显；这些观察不应当被解释为某一个参数的单独因果效应。相关模型及日志位于本地被 Git 忽略的 `logs/himloco_rsl_rl/go1_rough_pd_deploy_v1/2026-10-26_23-18-08_go1_pd_deploy_v1_seed1/`，需自行备份。

如需查看或恢复完整实验实现与评估，可在不切换当前分支的情况下运行：

```bash
git show archive/go1-deploy-v1:source/himloco_lab/himloco_lab/tasks/locomotion/robots/go1/deploy_v1_env_cfg.py
git show archive/go1-deploy-v1:docs/GO1_DEPLOY_V1_EVALUATION_20261027.md
git show archive/go1-deploy-v1:docs/GO1_DEPLOY_V1_DRIFT_REVIEW_20261027.md
```

仅有参数记录不足以直接运行 DeployV1 任务；需要时应从归档分支恢复其配置与任务注册，再做验证。

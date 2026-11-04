# 数据保留状态（2026-11-04）

用户要求优先保留代码、有效权重和报告，并释放轨迹占用。此前报告中的数据路径与 hash 是实验完成时的记录；现在相应二进制轨迹/cache 已清理，不表示实验结果改变。

- 删除 1394 个 `.npy/.npz`，释放 219.70 GiB；其他项目和 Step1 数据未动。
- 正式 35 组任务、780 个模型实例的权重均校验有效，本地保留于 `logs/impact_force_study_v1/`。locomotion baseline 和 JIT 保留。
- [data_provenance](data_provenance) 保存数据报告、cache metadata 与仿真配置。
- [training_provenance](training_provenance) 保存 HPO/训练曲线、训练报告和验证工作点，提交到分支；二进制权重不上传 Git。
- [删除记录与完整历史清单](../archives/force_study_v1_20261104/README.md) 保留删除范围、数据 hash 和本地方法/权重快照信息。

当前报告、图和统计结果可直接查阅。逐帧重算与新的持续学习实验需要重新采集数据；不能只靠这些统计或权重还原旧轨迹。

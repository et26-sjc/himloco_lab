# Continual Replay v1 保存索引

主实验已完成：4096 总环境、5 组独立采集×2 次初始化、9 种方法，主任务为 completed-interval D/BCE。严重度 Replay 未通过预设增益标准。

- [完整报告](../../../../docs/GO1_IMPACT_CONTINUAL_REPLAY_RESULTS.md)
- [机器可读结果](../../continual_replay_v1/confirmation/summary.json)
- [最终审计](../../continual_replay_v1/final_verification.json)
- [保留片段复算](../../continual_replay_v1/native_audit.json)
- [逐文件清单与哈希](manifest.json)
- [清理记录](DATA_REMOVAL.json)

本地 `methods_results_weights.tar.gz` 为 774.3 MiB，共 544 个文件，已逐项回读校验。保存方法/配置/报告、选定节点权重、最终 Adam/RNG/受限 Replay 状态、小规模原生 force/target/proprio 审计片段。压缩包与模型二进制留本地，代码、报告和清单提交研究分支 `feat/go1-impact-replay`。这是本地快照，不是异地备份。

按磁盘约束，删除 138 个完整 compact 数组和 5 个重复恢复文件，删除文件字节数合计 48.25 GiB。创建的较小 `final_training_state.pt` 保留原模型、Adam、RNG 和受限 buffer；清理的恢复副本含多余的全历史磁盘索引。完整数据的历史哈希均保留；没有备份完整轨迹，也没有恢复上一阶段已清理的约 220 GiB 数据。

有效权重位于 `logs/impact_continual_replay_v1/confirmation/rep1` 至 `rep5`：`seed*_freeze_A0.pt`、`seed*_*_B5.pt`、`seed*_*_C5.pt`；`stage0.pt` 与 `final_training_state.pt` 另保留优化器等状态。其它固定块端点权重也仍留在原 logs。初始化及方法对应关系见每组 `training.json` 的 `names`。

核验快照：

```bash
python scripts/impact/archive_continual_results.py
python scripts/impact/audit_continual_native.py
```

完整逐帧统计不能在已清理数据上重新计算。重跑必须在隔离 checkout/空输出目录重新采集，并移开已发布的 `artifacts/impact/continual_replay_v1` 参考结果目录，避免完成标记触发跳过；还需提供报告哈希对应的 baseline checkpoint/JIT。PhysX 不保证重新采集逐比特一致。新方法另用新最终测试 seed，不将已查看测试集继续用于未知测试。

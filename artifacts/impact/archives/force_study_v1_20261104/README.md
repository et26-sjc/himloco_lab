# Force study v1：方法、权重与报告保存

日期：2026-11-04（Asia/Shanghai）。**按用户磁盘预算要求，约 220 GiB 轨迹和缓存已清理；代码、有效权重、报告和关键元信息保留。**

- [DATA_REMOVAL.json](DATA_REMOVAL.json)：删除 1394 个 `.npy/.npz`，释放 235,905,525,066 bytes（219.70 GiB）。范围仅为 `datasets/go1_impact/force_study_v1/`；Step1 数据未动。
- 原位保留 35 组正式任务的 780 个模型实例，文件哈希与锁定清单一致，位于 `logs/impact_force_study_v1/`。baseline/JIT 权重也保留。
- [数据元信息](../../force_study_v1/data_provenance)：97 份 JSON/YAML/诊断标记已另存，包括原数据报告、cache metadata 和配置。有效运行与失败诊断的状态保持原记录。
- [manifest.json](manifest.json)：清理前的完整文件大小与 SHA-256 历史记录；不表示二进制轨迹仍存在。
- 本地 `methods_results_weights.tar.gz`：约 180 MiB，464 个代码/协议/结果/HPO 与正式权重等文件，压缩包内容已逐个回读校验。二进制压缩包不上传 Git，代码和报告直接提交研究分支。

这是代码、权重与结果保存，不是原始数据备份。现有结果可查看，无法在缺少轨迹的情况下重新计算全部逐帧指标；需要重新采集，且 GPU 仿真不保证逐比特一致。运行依赖以保存的 environment.json 与 baseline HEAD 对应仓库为准，不声称压缩包包含完整 conda/Isaac Sim 安装。

在仓库根目录校验本地方法/结果/权重压缩包：

```bash
conda run -n himloco_lab python scripts/impact/archive_force_stage.py --verify
```

`--include-data` 将明确报告数据已按要求清理，不再尝试验证不存在的轨迹。创建模式拒绝覆盖已有快照；后续代码/README 的迭代不会覆盖压缩包内的保存版本。

下一阶段：[持续学习与 Replay 实验方案](../../../../docs/GO1_IMPACT_CONTINUAL_REPLAY_PLAN.md)。

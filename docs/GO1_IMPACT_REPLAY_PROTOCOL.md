# Go1 本体感知冲击表征与事件 Replay：实验准备协议

版本：2026-11-03。分支：`feat/go1-impact-replay`；起点：`b482a92cb95d9bf95c352e8fcd5735ae34659b93`。

最新实际证据见 [接触力/loading-rate 阶段报告](GO1_FORCE_LOADING_RATE_RESULTS.md)：五组独立数据、七种方法与物理时间尺度诊断。下文保留最初准备协议和历史验收状态；Replay 及在线适应仍未完成实验验证。

本文件保留最初研究协议。后续已经实现并执行 A0/A1：七类受控地形、4096 env 独立采集、平滑加载标签、分类/回归/有序表征、五个初始化的静态监督训练、验证集校准及独立测试。第一步的实际结论与边界见 [GO1_IMPACT_STEP1.md](GO1_IMPACT_STEP1.md)，配置见 `configs/impact/step1.json`。事件 Replay、在线适应和控制收益仍待实现与检验。最初工程准备记录另见 `artifacts/impact/preparation_20261103.json`，不能把 smoke 当作科学结果。

## 1. 想提出什么模块，讲什么故事

研究问题：在冻结、已经能走的 Go1 上，有限带宽的本体观测能否提供**有时间边界、可校准、可跨场景保留的冲击事件/严重度信息**？

候选小模块是“多阈值有序冲击表征 + 事件片段记忆”。部署前向只读本体历史，输出每足不同严重度的超阈概率；训练端利用仿真接触监督，并以事件级 Replay 保留旧场景能力。它可以未来作为状态估计、风险监测或策略输入的可选组件。本轮不把收益绑在一个下游控制器上。

希望建立的证据链：

1. 明确 200 Hz 求解器信号和 50 Hz 本体历史能支持哪些任务，先验证监督而非假设 GT 完美。
2. 在同样时间窗/输入/预算下，判断分类、稳健回归、概率回归、有序表示分别适合何种信息。
3. 验证有序表示是否比单阈值二分类保留更多严重度信息，又比点回归更稳定。
4. 在真实数据分布转移且策略冻结时，验证事件记忆是否比同字节的随机记忆更有效地保留旧域，同时保有新域适应。
5. 检验未见场景/后端；准确报告能预测的部分与不可辨识的部分。

这些是可被否定的假设。若有序头不优于强回归，保留回归；若无持续学习漂移，不能声称 Replay 必要；若只在事件发生后有效，将模块定位为识别而非预警。

本体接触估计和经验回放都不是新概念，不宣称首次。创新候选在于有限带宽下冲击的严重度表示、时间可观测性和事件记忆的实证改进；最终需要消融、相关工作和跨域结果支持。

## 2. 冻结的实验基座与 4096 env 语义

| 条件 | 固定值 |
|---|---|
| locomotion | `Unitree-Go1-Velocity-PD`，冻结 Actor 与原 HIMEstimator |
| 原始模型 | `go1_rough_pd/2026-10-22_23-51-36_go1_pd_him_seed1/model_10000.pt` |
| checkpoint SHA256 | `3e2ec748f7657ac502334ce69e1275780ecf4cf63251c8afad7b25f5615b29f3` |
| 合并 JIT SHA256 | `82d69e89c1a5759d003da86d056e1152e4c5317a3dc19b49cbb11c45b661c336` |
| 执行器 | 显式 PD Kp40/Kd1，动作尺度 .25，12 个原生顺序关节 |
| 频率 | 默认 PhysX 200 Hz，策略/冲击模型输入 50 Hz |
| 第一组输入 | 当前 45D + 过去五帧 = 270D，首尾时间相隔 100 ms |
| 正式并行数 | **4096**；16/64 仅用于先验工程验证，不能代替正式实验 |
| 独立监督训练 | 不更新 locomotion PPO，也不更新 HIMEstimator；优化 batch 独立于 num_envs |

4096 env 是仿真采样规模，监督模型的 batch、更新次数、训练样本重复次数另行控制。不能用 4096 env 自动证明充分训练。先用学习曲线与延长训练确认收敛；比较方法共享同一缓存数据和每 seed 的 Stage-0 初始模型。

机器可读总计划：`configs/impact/go1_impact_replay.json`；已执行的第一步协议：`configs/impact/step1.json`。后续 Replay 阶段的 `planned_not_implemented` 是有意的。当前不增加旧网络的 16D 控制状态，不复用 MC 输出空间，不引入导纳、静音奖励或策略残差。

## 3. 三种时序任务，不能混报

记区间开始为 t，本体历史为 h_t，即将执行动作为 a_t，物理区间 W=(t,t+20ms]，连续监督候选为 S(W)。

| 任务 | 输入 | 监督 | 主要结论 |
|---|---|---|---|
| D：已发生事件识别 | h_(t+20ms) | S(W) 或其超阈标签 | 检测延迟、事件召回 |
| P20：未来风险 | h_t，另做 a_t 消融 | S(W) 或其超阈标签 | 严格接触前预警能力 |
| P40：未来 40ms 风险 | 同上，不读未来动作 | S((t,t+40ms]) | 更长时域能力/不确定性 |
| N：接触后严重度 | 已有微接触响应的因果历史 | 后续短窗严重度 | 仅当 A0 显示还有可观测剩余峰值才启动 |

所有模型使用相同因果边界。D、P20、P40 是不同任务，不横比 F1 后宣称方法更好。冻结确定性策略时 a_t 由 h_t 决定，拼接动作不增加信息论意义的信息；可能改善模型表达，需实测。

未来预测不违反因果律；无法保证唯一恢复完全隐藏的未来接触条件。配对隐藏几何实验应验证可观测前缀相同，并与 privileged oracle 比较条件风险。不能将未来状态、接触标记、地形 seed、域 ID 或已实现的未来动作加入部署输入。

terminal 区间若完整，保留其冲击；跨 episode 的未来 horizon 必须 masked，不把它当负样本，也不将跌倒/强冲击样本统一删除。post-frame 必须使用 reset 前 final_obs，不能用新 episode 第一帧。告警提前量扣除测量陈旧、推理和传输延迟；首次微接触后的告警不算接触前预测。

## 4. A0：监督信号和采集验收

本机 PhysX `net_forces_w` 是兼容别名，实际返回法向合力，不能声称法向+摩擦总力。使用 `net_normal_forces_w_history` 的三维世界系向量；模长能看到水平法向碰撞，但同时相反法向会抵消。暂不引入复杂 per-contact/friction 过滤。

独立 sensor 的 history length = decimation + rise_steps。默认 4+2=6：时间为 [-5,0,5,10,15,20] ms，后四个才属于当前区间。原 sensor 的 3 帧不能提供完整区间。原 sensor 保留，新增 sensor 不用于任何 reward 或输入。

候选连续量均保存，不在 A0 内选阈值：

- `normal_peak_n`：区间内法向合力模长最大值。
- `positive_increment_n`：区间内 max [F(k)-F(k-10ms)]_+。
- `positive_rate_fixed_window_n_per_s`：上量除固定 10ms。
- `positive_rate_one_step_n_per_s`：相邻物理步差分正峰值，作为旧方法对照。
- `normal_integral_ns`：20ms 内 Σ ||F_normal|| dt，包含普通支撑，不是剥离支撑后的碰撞冲量。

首个区间缺少足够 episode 内历史，增量 valid=false，不能将 reset 零填充值造成的跳变当正例。受力窗口本身仍完整保存。若采用体重归一化，使用固定名义质量；不按每个样本的隐藏随机质量重新定义标签。

物理测试：悬空/自由下落前段、名义站立稳定段、受控落下、正常步行、不同入射速度、上/下台阶、立面撞击、短重接触。检查支撑总力与重力/加速度关系，不强求每脚 mg/4。比较 dt=5ms 与 2.5ms，均保持策略 50Hz；检查事件匹配、严重度排序和积分稳定性，而非假设数值逐样本一致。

频率诊断要区分两件事：对同一高频轨迹先积分/滤波再降采样，用于研究测量带宽；重新用不同 physics dt 仿真则同时改变求解器与运动轨迹。二者不能混成一个“采样率实验”，也不能只用整段载荷积分相同证明短窗冲击标签收敛。

当前 `airborne` 是 2m 起点自由落体，**不是悬挂夹具**；只有碰地前窗口可作为零接触样本。`stand/drop` 使用零动作 PD，不执行步态 JIT，属于传感诊断。`plane_walk/rough_walk` 使用冻结 JIT。

工程验收包括：足序、六帧覆盖、原/新增 sensor 重叠值、跨策略步 trace 连续性、terminal 保全、原始帧重建策略历史、A/A 重复性和 A/B 插桩扰动。原生 recorder 会额外算观测并可能消耗噪声 RNG，因此本实现不使用 recorder，不复制物理 stepping loop。

只有 A0 的原始轨迹分析完成，才锁定主连续量、事件阈值、迟滞、事件合并间隔、ordinal 阈值及标签版本摘要。第一步采用独立定义的 10ms 平滑加载增量及 20/40/80N 操作性阈值，保留其物理幅度尚未收敛的限制，详见第一步报告。**不存在已经验证的 Go1 5000N/s 阈值。** 若标签主要由求解器抖动决定，停止当前标签路线。

## 5. A1/A2：充分训练下的公平表示比较

对 D 与 P20 分别执行相同数据的表示比较，P40 是后续难度扩展。

| 方法 | 作用 |
|---|---|
| 常数概率、相位/运动学规则、小线性模型 | 排除类别比例和周期步态捷径 |
| binary BCE | 直接学习 P(S>τ|h) |
| raw MSE | 历史点回归基线，不是唯一回归代表 |
| 归一化 Huber / 合理非负变换回归 | 排除量纲、长尾、离群值的可修复问题 |
| 概率回归或分位数回归 | 建模条件不确定性与尾部，不仅阈值化均值 |
| 单调 ordinal 超阈概率 | 候选模块：保留严重度次序，避免单阈值锁死 |

E[S|h] 与 P(S>τ|h) 不同。点回归阈值化不理想不能推出幅度不可学。概率回归通过其分布得到超阈概率，与分类共享事件指标，并另报幅度/覆盖率。

有序头候选输出 p_j=P(S>τ_j|h)，满足 p_1≥p_2≥...≥p_K。初期 K=3 或 4，仅作为设计候选；阈值不能看 test 后选。必须比较独立阈值头与单调约束的增量价值，避免把更多监督当成单调结构的贡献。初期共享小型 encoder 控制容量，不先追求大型网络；具体结构在训练阶段实现并登记。

二分类先比较自然采样 BCE 与必要的正类加权；不预置旧 pos_weight=3，不沿用旧控制 correction。训练重采样与概率校准分开。force 辅助头是后续消融，不是默认必要部分。

反捷径测试：单帧 vs 六帧、历史打乱、正常落足 vs 同相位硬碰撞、事件前后不同截断输入、错位标签诊断、已知信息更完整的 oracle。不能以“能预测周期 touchdown”替代“能区分异常冲击”。

表示创新通过条件：在配对 seeds 下，相对最强回归/二分类参考有明确的事件、严重度或跨域校准收益，且不是仅训练量/容量增加。如果没有，可保留更简单模型，转而只验证事件 memory。

## 6. B：有真实 shift、同一数据流的 Replay

先用同一缓存流排除闭环轨迹变化。采用 `A→A` 与 `A→B→C→A`。Stage-0 必须充分训练；所有方法从同 seed 同权重、同 optimizer 状态开始。流中先评后学，在线指标不能用已训练过的 batch 冒充泛化。

域划分初案（A0 后锁定具体范围）：A=已知平地/低等级 rough 与基础指令；B=速度及横移/转向变化；C=地形几何变化。传感噪声/延迟另作独立 shift，再测组合。第一轮不同时变化所有因素。冻结策略仍可能因域变化而跌倒，须报告失败率及失败片段，不能只比较幸存轨迹。

核心算法：freeze、finetune、近期样本重训（计算对照）、固定 Stage-0 anchor、全历史随机 reservoir、事件 Replay、全部已见数据训练参考。anchor 不能代表保存全部后续域；参考训练不能读取未发生的未来数据或 test。

Replay 比例统一为混合后旧样本占比 ρ=0/.2/.5；不复用旧“追加 ratio=.25”的歧义。存储预算 1/8/32 MiB 分别比较，不以相同条目数掩盖事件片段更大的开销。

两种预算协议必须分开报告：

1. 相同新样本暴露、相同更新次数：允许追加旧样本，另加同计算量的近期新域样本重训对照。
2. 相同总 batch 与更新次数：旧样本替换部分新样本，明确新样本曝光减少的代价。

记录 unique 新/旧样本、重复曝光、正负事件数、总 forward/backward 样本、optimizer steps 和墙钟时间。不存在追加旧样本后所有预算仍完全相同的免费对照。

事件 memory 候选：保留事件附近的因果上下文，并混合硬冲击、普通落足、容易误报但无冲击的片段。模型输入仍只读当时可用历史；片段内后续数据只用于后续样本监督。采样用独立 Generator，保存 buffer、cursor/RNG、optimizer、标签版本和源 dataset hash。事件平衡/困难样本采样必须与事件片段组织方式分开消融，避免只提高正类占比。

最小 2×2 消融（同输入/损失预算/字节数）：

| 表示 × memory | 随机窗口 Replay | 事件片段 Replay |
|---|---|---|
| binary BCE | 参考 | memory 单独贡献 |
| ordinal | 表示单独贡献 | 完整候选及交互 |

若新旧域在隐藏变量下真实冲突，Replay 只能折中；用域 oracle 诊断，不承诺无限适应且永不遗忘。

## 7. 数据独立性、指标与成功判据

train/replay、选型 validation、校准 validation、固定监测集合、最终旧域/新域/未见域 test 分开。按完整 episode、地形实例和采集 seeds 隔离，禁止相邻帧随机切分。持续监测集若用于选型，就不是最终盲测。重叠历史不能跨 split，事件片段也不能拆进不同 split。

Primary operating point 预设为 **1 次误报/robot-minute 下的事件召回**，并报 .1/5 的敏感性；这是研究比较点，不声称已满足某应用需求。阈值在 validation 锁定；一足事件一对一匹配、重复报警不重复计召回。匹配容差/事件合并在 A0 后锁定，不能为每种模型放宽。

同时报告 old/new event AP、自然分布 per-foot AP、原始混淆矩阵、Brier/可靠性曲线、严格接触前 lead time、检测延迟分布和严重度分层回归误差。不同 prevalence 的 AP 不直接互比；同一固定 test 上比较方法差值。持续学习报告完整轨迹、最终保留、最大退化和新域适应速度，不能只挑最佳 checkpoint。

正式核心比较建议 5 个配对 training/data seeds；每个 seed 内算法共享数据流和 Stage-0 初值。先给 per-seed 差值，再按 episode/地形做分层不确定性估计，不把四腿和相邻帧视为独立重复。未收敛结论与多 seed 结论分开。

Replay 通过标准：相对 finetune 和计算匹配参考改善旧域保留，且在预先约定的新域容忍区间内；新域容忍区间和最小实用效果需在 pilot 后、final test 前登记。事件 Replay 的创新标准是相对相同字节随机 Replay 的额外收益，而不是仅胜过无 Replay。

如果置信区间包含零或新域代价明显，结论应为不确定/存在权衡，不选择性宣布成功。

## 8. 规模、预算与资源

正式默认 4096 env。4096×2500 steps=10,240,000 transitions，每 env 50s；这是数据预算起点，不是“训练完成”的证明。跨 seed 的数据集和网络初值分别标识，避免只更换网络初始化却声称地形独立复验。

学习曲线建议 1,048,576 / 10,240,000 / 51,200,000 transitions。最后一级仅在仍有样本收益且存储允许时执行；不是所有模型、域、seed 自动保存 5120 万样本。监督 batch 初始候选 1024，检查 1k/5k/20k/50k optimizer steps；依据 validation plateau 和延长训练判定充分性。

NPZ 按最多 16 个策略步一 shard，保存原始 45D 前/后帧、12D 动作、六次三维接触、连续候选量、episode/terminal 元数据。历史可重建，不重复落盘每条 270D。初始估计约 0.8–1.0kB/transition（最终以实际 shard 大小测算），约 8–10GB/千万条；先规划数据规模再启动长采集。原始数据在被忽略的 `logs/` 或 `datasets/`，Git 仅收协议和小 JSON 报告。

显存报告区分 Torch allocator 与 PhysX/Warp/Kit/driver，不用 torch peak 伪装总显存。短时 4096 吞吐不能代表正式训练或长期传感器可靠性。GPU smoke 若 OOM，不把降到小 env 的结果冒充 4096；先定位采集 buffer/物理容量。

## 9. 当前可运行命令

在仓库根目录，使用 `conda activate himloco_lab`。每次选新 output，工具拒绝覆盖；失败目录保留 `INCOMPLETE`，不能作为完成数据。

CPU 契约：

```bash
python -m unittest discover -s tests -p 'test_impact*.py' -v
```

小规模时序/terminal smoke（不是科学训练）：

```bash
python scripts/impact/collect.py --headless --device cuda:0 \
  --num_envs 16 --steps 64 --scenario plane_walk --force-timeout-step 31 \
  --audit-baseline-contracts \
  --output logs/impact_preparation/smoke16
```

A/A、A/B 插桩比较：

```bash
python scripts/impact/collect.py --headless --device cuda:0 --num_envs 8 --steps 128 \
  --scenario plane_walk --baseline-observer --output logs/impact_preparation/base_a
python scripts/impact/collect.py --headless --device cuda:0 --num_envs 8 --steps 128 \
  --scenario plane_walk --baseline-observer --output logs/impact_preparation/base_b
python scripts/impact/collect.py --headless --device cuda:0 --num_envs 8 --steps 128 \
  --scenario plane_walk --output logs/impact_preparation/observed
python scripts/impact/compare_collections.py logs/impact_preparation/base_a logs/impact_preparation/base_b
python scripts/impact/compare_collections.py logs/impact_preparation/base_a logs/impact_preparation/observed
```

有噪声的插桩检查另加 `--observation-noise`，先看同条件 A/A。比较工具最多存八个 env 的状态前缀，不宣称覆盖所有未来轨迹。

独立离线检查 shard 摘要、episode/terminal 连续性、post→next pre 对齐、reset mask 及 NumPy 重算连续监督量：

```bash
python scripts/impact/audit_collection.py logs/impact_preparation/smoke16
```

该审计针对当前字段 schema；早期工具开发中更名前的临时 smoke 不作为正式数据复用。

4096 可行性 smoke，以及千万条原始数据的采集示例。后者仍是 A0 候选量采集：阈值保持未锁定，不等于正式模型训练或泛化验收；先完成小规模审计和存储规划再运行。

```bash
python scripts/impact/collect.py --headless --device cuda:0 \
  --num_envs 4096 --steps 128 --scenario rough_walk \
  --output logs/impact_preparation/scale4096

# 默认 4096 env、2500 steps；这是 A0 原始候选监督数据，不是 impact-model 训练。
python scripts/impact/collect.py --headless --device cuda:0 \
  --output datasets/go1_impact/pilot_seed1
```

诊断 `stand/drop/airborne` 更换 scenario；改变 dt 用 `--physics-dt .0025`，自动 decimation=8、history=12，保持 policy_dt=.02。传感噪声和 baseline 随机化由独立 flag 显式启用，实际配置落盘，不能把关闭随机化的 A0 结果当正式域泛化结果。

第一步完整入口现为 `python scripts/impact/run_step1.py --data-root datasets/go1_impact/replication2 --run-root logs/impact_replication2`；应使用新目录。该命令依次采集、审计、准备数据、训练、锁定报警点、评估、导出。它运行 A0/A1，不运行 Replay；详细参数、已执行命令和结果见第一步报告。后续顺序为改进物理监督/反捷径对照 → 独立数据复验 → Replay → simulator-online。

## 10. 不能提前宣称的结论

旧 MC 从回归改分类同时改了监督时间；旧 Replay 还增加 batch 与训练量；旧 1024×1000 运行并非从未做过充分规模探索。这些历史结果只能支持待检验假设。

本阶段成功只说明独立模块具有可信准备基础。仿真持续监督不等于真机无标签在线适应；MuJoCo 接触信号要重新定义并验证；没有独立真机监督时不承诺幅度精度。未来使用这个模块改善控制，还需另做下游任务实验，不能从分类精度直接推出控制收益。

## 11. 相关工作定位

- [Lin 等，CoRL/PMLR 2022](https://proceedings.mlr.press/v164/lin22b.html)：已有本体感知 contact-event 学习；本项目不把普通 contact classifier 当首次贡献。
- [Kang 与 Xiong，ICRA 2025](https://www.leggedai.com/publication/2025icra/)：已有含 Go1 的 GRF/state 估计，输入和约束不同；不能据旧 MSE 失败否定连续力估计。
- [Yang 等，ICML 2021](https://proceedings.mlr.press/v139/yang21m.html)：长尾连续目标需要认真处理，保留强回归对照。
- [Chaudhry 等，2019](https://arxiv.org/abs/1902.10486)：简单经验回放已是强基线；需检验事件记忆的独立价值。
- [Guo 等，ICML 2017](https://proceedings.mlr.press/v70/guo17a.html)：区分排序与校准，不假定后处理在新域自动有效。

历史材料：`~/Documents/impact_estimator.md` 与 `~/Documents/Go1_IsaacLab_Impact_Replay_Research_Review.md`。后者建议的“首轮不改文件”不适用于本轮用户已明确授权建立新分支和开展准备。

## 12. 本轮已经完成的准备验证

完整小报告：[preparation_20261103.json](../artifacts/impact/preparation_20261103.json)。原始 NPZ、完整配置、控制台和逐次命令在本地 `logs/impact_preparation/`，不加入 Git。

| 检查 | 结果与边界 |
|---|---|
| 新增 CPU 契约 | 13/13，包括首子步、固定物理窗口、侧向法向接触、terminal 对齐、损坏 shard、reset mask 和 episode 隔离 |
| 现有 HIM/PD 契约 | 11/11，包含实际独立 Estimator/PPO 更新测试和配置公平性 |
| 16 env terminal smoke | 64 steps，强制 16 行 timeout，全部 pre-reset contact trace 保全 |
| 有噪声 A/A 与 A/B | 各 8 env×128 steps；记录动作、q/dq、root、done 最大差均 0；只证明该有限前缀 |
| 4096 env rough smoke | 128 steps，524,288 transitions，采集+写盘 9.743s，无 OOM；不含约 29.9s 的环境启动 |
| 接触时序 | 新旧 sensor 重叠值、跨周期 contact history、部署历史重建误差均 0 |
| 原始数据独立审计 | 9 个选定集合全部通过摘要、时序、终止、mask 和独立 NumPy 重算检查 |
| 传感诊断 | 2m 自由落体的前 0.2s 所有足端接触为 0；名义站立与落地均有非零载荷 |

当前 shard 约 796 bytes/transition，千万条约 8.15GB（十进制，不含全部侧文件）。4096 短测吞吐约 53.8k transitions/s，仅是当前采集器包含写盘的观测，不能外推长期训练速度。

零动作落地的 8 个同名义初态环境、单个 seed 中：physics dt 5ms/2.5ms 的足端最大法向合力分别约 176.86/268.09N；整个 1.28s 的每 env 载荷积分约 164.458/164.456Ns。后者包含支撑，且不是事件对齐短窗统计。结果提醒我们不能直接用离散峰值或 dF/dt 的单一阈值跨 dt 宣称物理等价；**主标签、ordinal 阈值和事件定义仍未锁定**。

一次批量启动在进入环境构造前出现 Isaac Sim 原生退出码 139，栈涉及 XOpenDisplay/platforminfo/telemetry；根因未确定。失败目录保留 INCOMPLETE 和日志。同参数新目录重试通过上述 4096 检查，未修改 PhysX 设置或添加启动绕过。

本轮结论为“准备链路通过，A0 物理标签验证仍待完成”。分类、回归、有序表示、Replay、五 seed 和跨域效果均尚无实验证据。

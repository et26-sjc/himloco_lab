# 前腿异常摆动：被动关节阻尼对照

2026-10-25，已训练PD `model_10000.pt` / 同一JIT，Kp40/Kd1、action scale .25、200/50Hz均不变。
用户观察到的前腿thigh后偏、calf幅度大、膝盖高于机身，已在轨迹中确认。

## 排查顺序和证据

1. 使用原生Lab PD Play在平地运行4环境×20秒，固定.7m/s、nominal reset、关闭随机化。
2. 将1000步Lab关节/机身姿态逐一移植到MuJoCo，检查同名膝盖/足端位置。
3. 相同姿态下最大膝位置差`1.96e-6m`、足位置差`2.12e-6m`。
   同时45D/270D观测误差`6.56e-7`、JIT动作误差`1.43e-6`。
   此次没有发现前后腿排列、关节方向、零位或FK坐标错位。
4. 在独立诊断XML中对damping、armature、frictionloss做8组组合实验。
   原始模型不改，所有组合保持同一policy、初态、指令、Kp/Kd、频率、力矩限制与其他参数。

此前验收覆盖了数值/观测语义、力矩计算与存活；没有将前后腿幅度、膝高作为单独的运动质量判据。
本轮补充了实际Lab/MuJoCo gait与FK比较。

## 结果

以下关节和膝高统计取t>2秒，腿顺序FL/FR/RL/RR：
“膝盖高于机身”在表中指机身坐标z>0，即高于trunk原点，不是与mesh最高点比较。

| 条件 | 前腿thigh均值rad | 前腿膝盖高于机身的采样比例 |
| --- | --- | --- |
| 原生Isaac Lab | .849 / .895 | 0 / 0 |
| 原Menagerie MuJoCo | 1.704 / 1.591 | 45.8% / 38.2% |
| 只将额外passive damping设0 | .878 / .901 | 0 / 0 |
| 只将armature设0 | 1.721 / 1.567 | 47.4% / 37.2% |
| 只将frictionloss设0 | 1.685 / 1.589 | 43.9% / 38.2% |

单独取消额外damping就消除了这组条件下的反常前腿折叠；单独修改另外两项没有消除。
原始前腿calf峰峰值约1.20/1.33rad，取消额外damping后约.41/.48rad，前后腿幅度接近。
同一20秒.7m/s指令，实际vx从.533变为.633，线速度RMSE从.247变为.095m/s。

机制：MuJoCo原模型的物理joint damping为hip1、thigh/calf2Nm·s/rad。
本项目外部控制器还施加`-Kd*dq`且Kd=1；连续时间形式下，速度阻尼项之和是`-(Kd+b)*dq`。
它不等于训练时仅有显式Kd1且无额外被动关节阻尼的条件；位置伺服已删除，但这项物理阻尼仍然存在。
策略在不同的闭环动态下输出明显不同的前腿目标，并非只出现被动角度跟踪误差。

结论仅针对本次固定策略/速度/模型组合。零额外阻尼是训练仿真的参数对齐选项，
不代表已经辨识出Go1硬件的真实机械阻尼或摩擦。

## 显式修正选项

新增`--passive-damping 0`，仅在内存中覆盖12个关节的物理damping；
**Kp40/Kd1、armature .01、frictionloss .2、质量、碰撞、模型XML和策略均保持不变。**
不传该参数仍保留Menagerie原值，原始实验结果保留供复现。

```bash
GO1_POLICY=logs/himloco_rsl_rl/go1_rough_pd/2026-10-22_23-51-36_go1_pd_him_seed1/exported/policy.pt
python sim2sim/go1/run.py --policy "$GO1_POLICY" \
  --kp 40 --kd 1 --passive-damping 0 \
  --velocity 0.7 0 0 --duration 30 --real-time

python sim2sim/go1/evaluate.py --policy "$GO1_POLICY" \
  --kp 40 --kd 1 --passive-damping 0 --duration 20 \
  --output artifacts/sim2sim/gait_diagnosis/damping_zero_suite
```

YAML顶层也可设置`passive_damping: 0`，支持标量、12维列表或hip/thigh/calf分组；
此字段与`controller.kd`独立。运行时打印它，结果同时记录authored/effective damping、
runtime_overrides及effective_dynamics_sha256；CSV也带覆盖值和实际动力学指纹。

五组20秒回归均无跌倒：

| 指令[vx,vy,wz] | 原模型实际vx / wz | damping0实际vx / wz | 原 / 新linear RMSE |
| --- | --- | --- | --- |
| [0,0,0] | .001 / .005 | .002 / .004 | .011 / .010 |
| [.5,0,0] | .391 / -.051 | .454 / .004 | .191 / .068 |
| [.7,0,0] | .533 / -.066 | .633 / -.000 | .247 / .095 |
| [0,0,.3] | -.007 / .106 | .003 / .089 | .032 / .018 |
| [.7,0,.3] | .524 / .143 | .634 / .239 | .252 / .093 |

纯原地转向仍欠跟踪，不能宣称全部sim2sim差异已解决。
GUI使用显式damping0运行8秒，无跌倒；11项CPU单测通过，包含“物理damping覆盖不改变控制Kd”的回归。
运行时覆盖与独立XML零damping实验的20组trace数组逐值相同。

## 数据与复现

- [真实Lab/FK审计](../../artifacts/sim2sim/gait_diagnosis/isaaclab/validation.json)
- [8组物理参数对照](../../artifacts/sim2sim/gait_diagnosis/passives/comparison.json)
- [步态对照图](../../artifacts/sim2sim/gait_diagnosis/passives/gait_comparison.png)
- [修正后rollout画面](../../artifacts/sim2sim/gait_diagnosis/passives/corrected_frame.png)
- [damping0五指令结果](../../artifacts/sim2sim/gait_diagnosis/damping_zero_suite/summary.json)

```bash
python sim2sim/go1/audit_lab_contract.py --headless --gait-diagnostics --steps 1000 \
  --policy "$GO1_POLICY" --output artifacts/sim2sim/gait_diagnosis/isaaclab
python sim2sim/go1/diagnose_dynamics.py --policy "$GO1_POLICY"
HIM_GO1_TEST_POLICY="$GO1_POLICY" python -m unittest discover -s sim2sim/go1/tests -v
```

没有修改原生训练任务、reward、JIT权重或原始Menagerie资产；没有重训、改action scale或翻转关节符号。

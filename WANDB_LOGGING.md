# Go1 W&B / TensorBoard 日志

## 运行方式

默认logger仍为TensorBoard。W&B是可选依赖，本轮在既有环境额外安装W&B 0.31.0，未重装Isaac Sim、Lab、PyTorch或CUDA。

```bash
conda activate himloco_lab
python -m pip install -e 'source/himloco_lab[wandb,validation]' --no-build-isolation

# 本轮实际执行：32环境、3次更新、offline、开启有限采样histogram。
WANDB_MODE=offline python scripts/himloco_rsl_rl/train.py \
  --task Unitree-Go1-Velocity --headless --num_envs 32 --seed 1 \
  --max_iterations 3 --logger wandb --run_name wandb_offline_smoke \
  --wandb_group round2-validation --wandb_tags go1 smoke \
  --wandb_histograms --wandb_histogram_interval 1 --wandb_histogram_samples 512
```

4096环境正式W&B命令（**本轮未启动完整训练**；用户自行管理在线认证）：

```bash
WANDB_PROJECT=himloco-go1-baseline WANDB_MODE=online \
python scripts/himloco_rsl_rl/train.py \
  --task Unitree-Go1-Velocity --headless --num_envs 4096 --seed 1 \
  --max_iterations 10000 --logger wandb --run_name go1_him_seed1 \
  --wandb_group go1-baseline --wandb_tags go1 isaaclab3 physx
```

默认project=`himloco-go1-baseline`。`--log_project_name`优先于WANDB_PROJECT，随后为配置默认值。
`--wandb_mode offline|online|disabled`优先于WANDB_MODE；WANDB_ENTITY直接传给SDK。
支持`--wandb_group`、`--wandb_tags TAG...`、WANDB_RUN_GROUP、WANDB_TAGS（逗号分隔）。
默认关闭histogram和checkpoint上传；可用`--wandb_histograms`、`--wandb_histogram_interval`、`--wandb_histogram_samples`控制采样。
默认histogram间隔10 iterations、每类最多2048个标量样本。只有需要上传checkpoint时才显式加`--wandb_upload_checkpoints`，也支持`--no-wandb_upload_checkpoints`。
没有索取、读取打印API key或执行在线登录；online连接、云端resume和artifact上传没有在本轮实际验证。

## 单一日志路径

项目内`TrainingLogger`始终提供本地TensorBoard与`metrics.jsonl`；logger=wandb时再初始化一次W&B。
`sync_tensorboard=False`，不使用自动TensorBoard同步；每次iteration把完整指标字典只调用一次run.log，随后finish session。
W&B的step显式为已完成更新数，所有指标默认以trainer/iteration为横轴；`Train/.../time`显式使用Timing/total_time横轴。
参考[官方Run API](https://docs.wandb.ai/ref/python/experiments/run/)的log/define_metric接口。
已验证同一次SDK历史记录的`_step`等于trainer/iteration，3次更新对应3条history，没有重复初始化或重复W&B history。
TensorBoard模式不import/初始化wandb，也不生成wandb目录。

## 指标清单

| 类别 | 实际记录 |
| --- | --- |
| 进度 | `trainer/iteration`, `trainer/timesteps`, `Metrics/total_timesteps` |
| HIM | `Loss/Estimation Loss`, `Loss/Swap Loss`，沿用PPO update所有minibatch的平均值 |
| PPO | `Loss/value_function`, `Loss/surrogate`, `Loss/learning_rate`, `Policy/mean_noise_std` |
| 训练统计 | `Train/mean_reward`, `Train/mean_episode_length`, 两者的`/time`曲线 |
| 性能 | `Perf/total_fps`, `Perf/collection_time`, `Perf/learning_time` |
| 时间 | `Timing/total_time`, `Timing/iteration_time` |
| 课程 | `Episode/terrain_level`, `Episode/max_command_x`，及原生`Curriculum/terrain_levels`, `Curriculum/lin_vel_cmd_levels` |
| 分布（可选） | `Distributions/actions`, `Distributions/values` |
| 原生诊断 | `Metrics/base_velocity/error_vel_xy`, `Metrics/base_velocity/error_vel_yaw`, `Metrics/success_rate`；以及启用的`Episode_Termination/*` |

计数从1开始表示已完成的更新数，与checkpoint文件名一致；旧baseline的console/log迭代起点可能为0，比较时需对齐。
Episode length单位为policy steps（50Hz），reward为episode累计加权reward。没有episode结束时不伪造Train/mean_reward或episode项。
Perf FPS和Timing/total_time基于rollout+更新耗时，不含场景初始化及日志提交/上传耗时，与旧runner计时口径相同。
本轮没有为了approx KL或诊断去额外执行策略forward，也没有修改Estimator/PPO数学更新。

### Episode reward名称与归一化

| 当前Lab term | 兼容旧baseline的key |
| --- | --- |
| track_lin_vel_xy | Episode/rew_tracking_lin_vel |
| track_ang_vel_z | Episode/rew_tracking_ang_vel |
| base_linear_velocity | Episode/rew_lin_vel_z |
| base_angular_velocity | Episode/rew_ang_vel_xy |
| flat_orientation_l2 | Episode/rew_orientation |
| joint_acc | Episode/rew_dof_acc |
| energy | Episode/rew_joint_power |
| base_height_l2 | Episode/rew_base_height |
| feet_height_body | Episode/rew_foot_clearance |
| action_rate | Episode/rew_action_rate |
| smoothness | Episode/rew_smoothness |

只读取已启用且weight非零的terms，不给缺失reward填0。
Lab累加的是`term_value × weight × sim_policy_dt`；新增`Episode/rew_*`在reset前按**该episode实际执行的秒数**除后取平均，再在iteration内汇总reset批次。
这对应旧Gym的按episode秒数归一化思想。单独维护实际步数，避免init_at_random_ep_len的虚拟初始步数稀释首个episode的reward rate。
集成测试把episode_length_buf人为置为超时前一步，确认全部分项rate之和仍等于本次实际reward/0.02s。

原生`Episode_Reward/<Lab term>`同时保留；本地Isaac Lab RewardManager按**最大episode_length_s=20秒**归一化，短episode时与新增key数值不同。
不能通过重命名认为Gym与Lab数值完全等价：机器人执行器、质量惯量/接触、观测缩放、地形、传感器及部分reward计算细节不同。
name alias只说明贡献对应关系；含权重的贡献通常是reward每秒的量，不是原始力矩、米或加速度本身。

### Histogram开销

仅指定iteration采样；每10个rollout step，从GPU action/value张量头部截取小片段，按iteration预算分配采样量。
收集时不做CPU传输；iteration结束只将截断后的最多指定数量样本传至CPU，构建W&B/TensorBoard histogram。
32环境测试预算512，实际action样本510、value样本320；从W&B二进制中解码bin counts确认，不是只检查本地镜像文件。
未把4096×100的全部action/value传输或上传。默认不启用采样。

## Checkpoint与恢复

新格式增加`checkpoint_format_version=2`、`tot_timesteps`、`tot_time`、`wandb_run_id`。
模型与两个optimizer照常保存/恢复；日志改动不改变Estimator独立优化路径。
累计步数在runner实际完成update时增加，即使没有logger也维护；恢复不根据当前环境数量重算历史步数。

```bash
WANDB_MODE=offline python scripts/himloco_rsl_rl/train.py \
  --task Unitree-Go1-Velocity --headless --num_envs 32 --seed 1 \
  --max_iterations 2 --resume \
  --checkpoint logs/himloco_rsl_rl/go1_rough/2026-10-22_00-23-55_wandb_offline_smoke/model_3.pt \
  --logger wandb --run_name wandb_offline_resume --wandb_group round2-validation \
  --wandb_histograms --wandb_histogram_interval 1 --wandb_histogram_samples 512
```

在线resume默认使用checkpoint的run ID（resume=allow），也可显式`--wandb_run_id`。
**offline不是服务端append/resume**：默认新建本地session并保留parent_run_id与累积计数，方便分别sync后按group查看；不伪称W&B原生offline resume。
云端同一run的resume未实际验证。

第一轮model_4没有累计计数，读取checkpoint旁的`params/env.yaml`和`agent.yaml`重建：
4×4096×100=1,638,400。该fallback假设此前所有iteration维持源配置的环境/rollout规模；本机model_4符合条件。
若历史规模变化且checkpoint未记计数，应先补齐元数据，不能静默用当前环境数量猜测。缺失源参数时，训练resume明确报错。
旧格式没有累计计时，无法精确恢复；其Timing/total_time从恢复后的新计时基点开始。新格式恢复可连续累计计时。

## 已执行验收

- 32envs、3 updates：iteration `[1,2,3]`，timesteps `[3200,6400,9600]`。
- 从model_3恢复32envs、2 updates：iteration `[4,5]`，timesteps `[12800,16000]`，time继续累计。
- 实际SDK数据中所有要求的标量/11项reward/课程/histogram均存在，tag为go1、smoke，group为round2-validation。
- W&B关闭：默认TensorBoard从第一轮model_4恢复32envs、1 update，iteration5、timesteps1,641,600，无wandb目录。
- 原有7项HIM合约+真实PPO/保存恢复/导出回归通过，另有3项legacy adapter测试通过。

验收读取**实际`.wandb` protobuf history**并校验framing checksum，不连接网络：

```bash
python scripts/himloco_rsl_rl/inspect_wandb_offline.py \
  logs/himloco_rsl_rl/go1_rough/2026-10-22_00-23-55_wandb_offline_smoke \
  --max_histogram_samples 512 --output artifacts/round2/wandb_smoke.json
```

W&B 0.31已移除旧Python DataStore；检查脚本按[官方历史格式源码](https://github.com/wandb/wandb/blob/v0.19.11/wandb/sdk/internal/datastore.py)读取framing，再使用当前安装的protobuf schema解码。
报告：[初训](artifacts/round2/wandb_smoke.json)、[恢复](artifacts/round2/wandb_resume.json)、[旧计数重建](artifacts/round2/counter_resume.json)。
本机日志：`/tmp/himloco_wandb_{smoke,resume}.log`、`/tmp/himloco_tb_counter_resume.log`、`/tmp/himloco_round2_regression.log`。
SDK session及checkpoint在上述logs目录，二进制训练产物不提交Git。

修改文件：`rsl_rl/utils/training_logger.py`、`rsl_rl/runners/him_on_policy_runner.py`、`rsl_rl/config/rl_cfg.py`、`envs/him_manager_based_rl_env.py`、`cli_args.py`、`setup.py`、`validate_go1.py`，新增offline检查脚本。

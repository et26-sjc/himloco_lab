# 2026-10-26 跟踪误差停止判据的小幅调整

后续实机记录：修改后用户反馈走得明显更好，日志`him_lab_20240615_182005_14220.jsonl`中策略运行约63.82秒，没有跟踪停止；连续超0.50rad的最长记录约0.08秒。末尾停止原因为FR calf候选目标低于现有机械下界约0.00184rad，候选未下发。本次固定发布流程另将横移符号改为-1，见[固定流程](DEPLOYMENT_GUIDE.md)。

## 调整依据

用户在后台模式、速度上限 `[0.6, 0.3, 0.6]` 下确认前进、横移、转向均可执行。两次运行均由 Python 的单次跟踪误差停止判据结束：

| Go1 日志 | 策略接管后时间 | 关节 | 上一帧已发布目标 | 停止时实测 | 绝对误差 |
| --- | --- | --- | --- | --- | --- |
| `him_lab_20240615_175703_11925.jsonl` | 21.22秒 | RR calf | -1.4393rad | -1.9694rad | 0.5301rad |
| `him_lab_20240615_175934_12563.jsonl` | 7.30秒 | RL calf | -1.3733rad | -1.9034rad | 0.5300rad |

文件名时间来自机器人的旧系统时钟。本次分析日期为2026-10-26。两次最后前进命令均约0.6m/s，Python周期约20ms；桥接在Python请求停止后锁停。最新一次在目标快速变化后超限，较早一次还有后小腿估算力矩升高，不能仅凭这些日志把全部停止归为误报或断言电机已限流。

用户授权小幅调整，要求避免过多限制。本次只修改跟踪停止判据和诊断日志；没有增加命令滤波、强制最小速度或动作裁剪。

## 新判据

`himloco.json` 中：

```json
"max_tracking_error_rad": 0.5,
"tracking_error_timeout_s": 0.15,
"max_instant_tracking_error_rad": 1.0
```

- 比较量仍是**当前实测关节位置与上一帧已发布的目标**，不是默认姿态，也不是尚未发布的新策略目标。
- 仅在`RAMP`和`RUNNING`阶段，同一个关节的绝对误差持续大于0.50rad、从首次观测超限起计时达到0.15秒才停止。50Hz离散采样下通常约0.16秒；不是保证在两次采样之间检测连续运动。
- 每个关节独立计时。某关节任一采样回到0.50rad以内（含边界）即清零该关节计时；不同关节不能串接累计时间。
- 任一关节单次绝对误差大于1.00rad仍立即停止。
- `CALIBRATE`、`SETTLE`、`READY`仍沿用原0.50rad单次停止，没有放宽标定就绪容差。
- 旧配置不包含两个新字段时，保留原单次0.50rad停止行为。

这是允许短时滞后的改动，会延后持续跟踪失败时的停止。它不证明所有大误差正常，不保证再无其他原因触发停止。

## 保持的接口与参数

原JIT权重、Lab45维观测和6帧历史、名称关节映射、四元数约定、原始动作历史回填、`q0+0.25*action`、Kp40/Kd1、零目标速度/前馈力矩均保持原样。遥控速度上限仍为`[0.6,0.3,0.6]`，未新增命令渐变或动作步长限制。

C++桥接和LCM协议未修改；机械范围、SDK保护、通信新鲜度、周期门限、机身倾角、L2停止、运行中R2停止和故障锁存仍保留。本改动不增加自动重启，不启动电机。

## 新日志

每周期增加：

- `tracking_error_max_rad`、`tracking_error_max_joint`：相对上一帧目标的最大误差和关节。
- `tracking_error_over_limit_s`、`tracking_error_over_limit_joint`：各关节连续超限时间中最长的一项。它可以与最大误差对应的关节不同。
- `tracking_error_limit_rad`、`tracking_error_timeout_s`、`tracking_error_instant_limit_rad`：本周期实际使用的门限。

若再次因跟踪误差停止，终端打印关节、误差和持续时间；JSONL的`fault_details`记录`instant`或`sustained`原因、12关节误差和计时、实测位置速度及上一帧目标。停止请求仍在日志格式化和磁盘写入之前发送。

## 启动和回退

上传不会让已运行的Python热更新，也不会重新解锁桥接。先结束旧策略和已锁停的旧桥接，再按原方式显式启动：

```bash
sudo bash /home/unitree/go1_lab_deploy/stop_background.sh
sudo bash /home/unitree/go1_lab_deploy/start_background.sh
tail -f /home/unitree/go1_lab_deploy/logs/latest_policy.console.log
```

仍为第一次R2标定、READY后释放并再按R2接管，L2各阶段停止，运行中R2停止。后台的`tail -f`中Ctrl+C只停止查看日志。

最初上传这项跟踪调整时仅核对文件完整性和配置，未运行单元测试、JIT回放或启动电机。随后用户进行了本文开头记录的实机运行。初始离线报告属于旧版本，不能算作本次跟踪改动的独立回归验证。

本机备份：`/home/sjc/himloco_lab_review_20261026/before_tracking_20261026_203917.tar.gz`。
Go1备份：`/home/unitree/himloco_releases/go1_lab_deploy_before_tracking_20261026_203917.tar.gz`。
需恢复旧行为时，将两个新字段改为`tracking_error_timeout_s=0.0`、`max_instant_tracking_error_rad=0.5`并重新生成manifest，或恢复整个备份；只对下次显式启动生效。

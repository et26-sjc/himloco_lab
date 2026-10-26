# 真机返回版本合并与仓库回归（2026-10-26）

输入为用户已经在Go1运行的`/home/sjc/go1_lab_deploy_20261026.tar.gz`，SHA256：
`c49fb6ded57685333f1cb94d57a7836796003a2fb0eed5d654a2245075494c18`。
原归档完整性清单74个文件全部匹配，原归档保持只读原样；本次在仓库内合并并生成新的manifest/归档。

## 合并内容

- 新增后台启动、状态、停止三个入口；nohup/setsid、独立日志、Docker init/SIGINT和归属标签。
- 新增后台网络准备：新建套接字之前将本部署组播主机路由固定到SDK内部接口；不改SSH/default route。
- 用户当前命令上限`.6/.3/.6`和摇杆符号`[1,-1,-1]`原样同步。
- RAMP/RUNNING的跟踪保护按同一关节独立计时：>0.5rad持续0.15s，或>1rad立即停止；
  返回阈值以内清除该关节计时，不能跨关节累计。50Hz通常在首次超限后约0.16s检测持续超限。
- 校准、SETTLE、READY仍是原0.5rad单次停止，旧配置缺少新字段时维持原行为。
- 合并跟踪诊断日志、实机运行摘要和固定操作流程。机械限位、按钮、watchdog和故障锁存保留。
- 新JIT与原JIT逐字节相同；Lab观测/动作接口、Kp40/Kd1、C++ bridge和四份LCM schema均不改。

真机运行的用户记录见`runtime/lab_tracking_run_20261026.json`，最新摘要包含约63.82秒策略活动，
并记录一次FR calf目标越继承边界.00184rad被拒绝。该原始运行日志不随包提供，摘要保留其SHA。
返回包把横移符号修正后的专门真机核对标记保留为未完成；本次离线回归不将它改成已验证。
原始deployment_contract内的硬件状态是最初导出时的记录，后续实机结果单独保存在runtime文档中。

## 本轮真正执行的回归

| 项目 | 结果 |
| --- | --- |
| 部署Python unittest | 87项通过，新增持续/瞬时误差、恢复、跨关节不累计、RAMP及旧配置测试 |
| MuJoCo独立CPU测试 | 11项通过，包含实际PD JIT |
| 原HIM张量/算法回归 | 7项通过 |
| C++ bridge guard构建及ctest | 1/1通过 |
| Bash语法 | 11个脚本通过，未执行网络/启停操作 |
| 当前代码回放原生Lab .3m/s参考 | 2048样本通过，frame/action最大误差3.82e-6，SDK绝对目标误差9.54e-7rad |
| 当前Controller模拟接管 | 15秒通过，含标定期理想化支撑，RUNNING10.8秒，无fall/guard/NaN |

命令（在主仓库已安装环境中）：

```bash
PYTHONPATH=go1_lab_deploy python -m unittest discover -s go1_lab_deploy/go1_lab_deploy/tests/himloco -v
HIM_GO1_TEST_POLICY=go1_lab_deploy/models/policy.pt python -m unittest discover -s sim2sim/go1/tests -v
python -m unittest discover -s tests -p test_him_contracts.py -v
python go1_lab_deploy/go1_lab_deploy/scripts/validate_replay.py \
  --config go1_lab_deploy/himloco.json \
  --reference artifacts/go1_lab_deploy/lab_reference_cmd03/reference.npz \
  --output artifacts/go1_lab_deploy/replay_new.json
python scripts/himloco_rsl_rl/validate_deploy_handover.py --support-until-ready \
  --output artifacts/go1_lab_deploy/handover_new.json
```

数值回放与夹具接管仍不是新的真机试验。本轮没有连接机器人、启动Docker控制容器、发送LCM/UDP或改变本机路由。
旧离线报告保留在原路径，新报告在`validation/returned_release/`，主仓库对应原始结果在
`artifacts/go1_lab_deploy/returned_20261026/`。模拟接管的支撑约束仅作用于校准期floating root，
12个关节始终使用模拟动力学；不能将它描述成全程无约束真机运动。

原生Lab冷启动短暂机械目标越界的既有发现仍保留，原限位不放宽；这与逐字段数值回放通过是两个结论。
训练reward、命令采样和PPO超参数未修改。下一版温和训练建议另见主仓库`docs/GO1_DEPLOYMENT_TUNING_V1.md`。

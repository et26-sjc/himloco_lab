# Go1 Lab HIM 固定部署流程

更新日期：2026-10-26。默认部署目录为`/home/unitree/go1_lab_deploy`，开发源目录为`/home/sjc/himloco_lab/go1_lab_deploy`。

## 1. 当前发布包含什么

| 项目 | 当前值或行为 |
| --- | --- |
| 策略 | Lab PD训练`model_10000.pt`对应原始完整JIT，Estimator+Actor |
| JIT SHA256 | `82d69e89c1a5759d003da86d056e1152e4c5317a3dc19b49cbb11c45b661c336` |
| 输入 | 45维单帧、6帧历史、当前帧在前，共270维 |
| 目标 | `q0+0.25*raw_action`，按真实名称映射到SDK，只缩放一次 |
| 控制 | Python50Hz，原C++桥接500Hz，MCU执行Kp40/Kd1，目标速度和前馈力矩为0 |
| 遥控最大命令 | 前后±0.60m/s，横移±0.30m/s，转向±0.60rad/s |
| 遥控符号 | `[vx,vy,yaw]`对应`command_axis_signs=[1,-1,-1]` |
| 摇杆死区 | 0.05；回中发送零速度，不做最低速度跳变 |
| 标定/接管 | 3秒标定、6个实测历史帧、1秒策略渐入 |
| 跟踪停止 | RAMP/RUNNING：同一关节误差连续超0.50rad约0.15秒，或单次超1.00rad |
| 其他停止 | 机械范围、机身倾角、状态/遥控过期、控制周期、SDK功率保护和按钮停止保留 |
| 目标步长停止 | `null`，不额外限制每帧策略目标变化 |
| 故障退出 | 零增益停止请求并锁存，手动重启，不自动重新使能 |

## 2. 从初始包到当前包的修改

1. **后台启动**：新增`start_background.sh`、`status_background.sh`、`stop_background.sh`；桥接和Docker客户端使用`nohup setsid`脱离SSH。Docker后台不带`-it`，使用`--init`和SIGINT停止，以便Python先发送停止请求并保存日志。
2. **网络**：新增`scripts/prepare_background_network.sh`。创建新LCM套接字之前，把该部署组播地址的路由指定到SDK所在机器人内部网口；本机为`239.255.76.67/32 dev eth0`。不修改SSH/默认路由，不加入开机自启。
3. **速度**：`max_abs_command`由`[0.3,0.1,0.15]`改为`[0.6,0.3,0.6]`，仍线性响应摇杆，不加入最低速度或动作裁剪。
4. **跟踪判据**：原0.50rad单次停止改为策略阶段的短时容忍和较大误差立即停止；每关节独立计时。标定、SETTLE、READY阶段仍为原0.50rad单次停止。新增具体关节、误差、持续时间和故障数组记录。
5. **横移方向**：用户报告左摇杆左右与横移相反，因此仅把`command_axis_signs[1]`从`+1`改为`-1`。前后轴和转向轴不变；不改变IMU坐标、关节映射、策略输入的坐标定义或权重。该修正需要下次实机确认左右对应。
6. **发布同步**：上述代码、默认配置和说明一起写入初始源码目录、原解压目录及归档包，更新manifest；初始版本和每次修改前版本均有备份。这样再次复制整个包时不会恢复旧横移符号或旧单帧跟踪停止。

原JIT、Lab原生观测/历史/推理链路、位置目标转换、Kp40/Kd1、C++桥接及四份LCM协议未改。包装脚本和部署停止判据属于部署流程修改，不属于策略权重修改。

## 3. 当前Go1再次启动

使用原`jetson-model-deployment`镜像；新入口会创建临时容器`him_go1_lab_policy`，不要求先启动`foxy_controller`。先按之前已成功的方式进入Go1低层开发模式，结束原厂运动程序，并让支撑避开四腿活动范围、能够在停止时承接机身。

电脑通过外部网线SSH到Go1：

```bash
ssh unitree@192.168.123.15
```

上一轮已经停止后，在Go1的同一个SSH终端执行：

```bash
sudo bash /home/unitree/go1_lab_deploy/stop_background.sh
sudo bash /home/unitree/go1_lab_deploy/start_background.sh
tail -f /home/unitree/go1_lab_deploy/logs/latest_policy.console.log
```

启动时按一次Enter确认后台桥接。按钮按下后松开，不需要一直按住：

1. 等`HIM phase: WAIT_R2`，摇杆回中，第一次按R2，开始标定。
2. 等`HIM phase: READY`，观察姿态；释放R2，再按一次R2接管。
3. 经过`RAMP`进入`RUNNING`，摇杆开始发送运动命令。

整个过程只用一个桥接和一个策略容器。SSH重新连上后先查看状态，不要重复启动已有控制器。

## 4. 遥控器操作

方向以**机器人自身前后左右**为准，不按操作者面对机器人时的左右判断。策略坐标为x向前、y向左、z向上，正yaw向左转。

| 操作 | 命令/行为 |
| --- | --- |
| 左摇杆向前/后 | 前进/后退，`vx`正/负 |
| 左摇杆向左/右 | 左移/右移，`vy`正/负；本次修正横移符号 |
| 右摇杆左右 | 转向；沿用此前可用的yaw符号-1 |
| 摇杆回中 | 零速度命令；策略继续运行，不锁定空间位置，也不保证绝对不漂移 |
| L2 | 所有阶段请求停止，桥接也直接处理此按钮 |
| R2在WAIT_R2 | 标定 |
| R2在READY | 策略接管 |
| R2在RAMP/RUNNING | 请求停止 |

支持逐渐释放上提承重的流程沿用此前实机操作；不要让支架或绳索碰腿。结束时零增益不保持站立，承托需准备好。

## 5. 后台、网线、查看和结束

策略和遥控接收均运行在Jetson/机器人上。进入RUNNING后可断开**电脑连接的外部网线**，无需持续保持SSH；内部MCU网线和Jetson供电要保留。现有日志未单独标记拔线时刻，因此不把本次报告当作独立的拔线连续性测量。

后台看日志的`Ctrl+C`只结束`tail`，不会停止机器人。用遥控器L2或运行中的R2停止。电脑仍连接时，也可在机身有承托的条件下执行：

```bash
sudo bash /home/unitree/go1_lab_deploy/stop_background.sh
```

查看进程、最近输出、CPU和内存：

```bash
sudo bash /home/unitree/go1_lab_deploy/status_background.sh
```

状态脚本会给出可复制的`top -p PID列表`；`docker stats`中100%CPU约表示一个逻辑核。日志保存在`logs/`，容器退出后仍保留。故障后需要结束旧桥接再重启，不会自动恢复动作；不要使用Ctrl+Z暂停控制进程。

## 6. 复制归档到另一目录或重装

本机可移植归档为`artifacts/go1_lab_deploy/go1_lab_deploy_20261026.tar.gz`，在`himloco_lab`仓库内；同时更新原`/home/sjc/go1_lab_deploy_20261026.tar.gz`。解压后顶层目录是`go1_lab_deploy`。

源代码、配置与JIT一起复制。归档不带运行日志或Jetson编译产物。若是全新目录，需要在Go1上准备桥接；本机已验证桥接可从以下位置复用：

```bash
bash /home/unitree/go1_lab_deploy/scripts/prepare_bridge.sh --reuse \
  /home/unitree/himloco_releases/20261025_forward_trial_3/go1_gym_deploy/unitree_legged_sdk_bin/build/lcm_position_him
```

原路径中的桥接及其`.sha256`需要一起存在。原WTW二进制`lcm_position`不能替代`lcm_position_him`。已有部署目录的桥接本次未替换，无需重新编译。

若重装Go1或更换Jetson，首次环境预检可用：

```bash
sudo bash /home/unitree/go1_lab_deploy/scripts/check_environment.sh
```

这只离线加载配置和策略，不启动电机。更改源码或JSON后，开发机需显式刷新manifest再打包；不要只在机器人上改JSON而留下旧摘要。

## 7. 已确认的运行结果和后续事项

本次整理前的`him_lab_20240615_182005_14220.jsonl`策略接管后运行约63.82秒，共3191帧，控制周期最大约20.70ms；最大跟踪误差0.7211rad，连续超0.50rad的最长时间约0.08秒，未触发新的持续跟踪停止。用户确认行走改善，可前进、横移和转向。

该次最后因FR calf候选目标-2.698369rad低于部署下界-2.696534rad约0.001836rad而停止，候选未下发。本次仅同步流程和翻转横移符号，保留这项机械范围；它仍可能再次触发。摘要和原日志SHA见`runtime/lab_tracking_run_20261026.json`。机器人时钟显示2024年，报告日期以2026-10-26为准。

原归档整理时未新增运行单元测试或仿真回放；本次仓库合并已补齐回归，见[合并验收](REPOSITORY_SYNC_20261026.md)。本次合并没有启动新电机实验。对未达预期的步态仍需进行策略与动力学对照，当前记录不把全部现象归为已证实的单一原因。

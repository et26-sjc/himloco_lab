# 独立 JIT / ONNX 导出验收

## 接口与独立性

`validate_export.py` 从已有checkpoint严格加载两套网络、重新导出、启动一个 `python -I` 子进程从磁盘加载模型。
该子进程仅import torch、ONNX Runtime、NumPy和标准库，并断言sys.modules没有Isaac Lab/Sim或himloco_lab；不使用Runner、Critic或privileged observation。
可直接复制 `standalone_inference.py` 和导出模型到无Isaac Lab的Python环境。

- JIT `policy.pt`：270D history → encoder → 3D velocity+L2归一化16D latent → concat当前45D → Actor →12D action。
- ONNX `encoder.onnx`：`obs_history[batch,270] → encoder_output[batch,19]`。
- ONNX `policy.onnx`：`obs[batch,64] → actions[batch,12]`，其输入是 `[history[:,:45], encoder_output]`，不是270D。
- 本轮将ONNX默认改为动态batch。接口检查仍严格固定270/19/64/12维；保持兼容batch1调用。
- JIT通过script导出，未trace冻结batch/设备。CPU上验证batch1/7/64和256条真实rollout输入；未执行CUDA专项导出验收或JIT控制轨迹对照。
- 输出是raw action，不包含默认站姿、.25动作缩放或legacy语义适配。旧模型必须在外部使用legacy adapter处理观测和动作。

## 实测误差

比较“初次加载PyTorch”“重建并再次严格加载PyTorch”“独立磁盘JIT”“独立磁盘split ONNX”。
两套PyTorch及JIT在全部CPU用例最大/平均误差均为0；ONNX如下。误差是相对初次加载PyTorch的action绝对误差，检查容差atol/rtol均为3e-5。

| checkpoint | 输入 | ONNX最大误差 | ONNX平均误差 |
| --- | --- | --- | --- |
| Lab model_4 | random_batch_1 | 4.47034836e-08 | 2.15756391e-08 |
| Lab model_4 | random_batch_7 | 8.94069672e-08 | 2.00788719e-08 |
| Lab model_4 | random_batch_64 | 1.49011612e-07 | 2.36909354e-08 |
| Lab model_4 | real_rollout | 2.38418579e-07 | 2.72955276e-08 |
| Gym model_7500 | random_batch_1 | 2.38418579e-06 | 8.18322121e-07 |
| Gym model_7500 | random_batch_7 | 4.29153442e-06 | 7.53573033e-07 |
| Gym model_7500 | random_batch_64 | 5.24520874e-06 | 8.37734376e-07 |
| Gym model_7500 | real_rollout | 3.33786011e-06 | 4.13844333e-07 |

真实输入来自本轮Lab仿真采集：新模型用Lab原始history，旧模型用adapted legacy history；这些rollout由legacy评估产生，不宣称是新短训策略自身的运动质量评测。
完整报告：[新模型](artifacts/round2/exported_lab/validation.json)、[旧模型](artifacts/round2/exported_legacy/validation.json)。

## 复现

先按LEGACY_POLICY_EVAL.md采集观测，随后无需启动SimulationApp或训练：

```bash
python scripts/himloco_rsl_rl/validate_export.py \
  --checkpoint logs/himloco_rsl_rl/go1_rough/2026-10-21_23-38-08_resume4096/model_4.pt \
  --observations artifacts/round2/legacy_plane/direct_observations.pt \
  --output artifacts/round2/exported_lab
python scripts/himloco_rsl_rl/validate_export.py \
  --checkpoint /home/sjc/HIMLoco-Go1/legged_gym/logs/rough_go1/Sep29_01-01-22_baseline_seed1/model_7500.pt \
  --observations artifacts/round2/legacy_plane/adapted_observations.pt --observation_key policy_observations \
  --output artifacts/round2/exported_legacy
```

独立部署最小JIT推理：

```python
import torch
policy = torch.jit.load("artifacts/round2/exported_lab/policy.pt", map_location="cpu").eval()
with torch.inference_mode():
    action = policy(torch.zeros(7, 270))  # [7,12]；实际控制需传入真实history
```

Play现在默认不导出；需要时传 `--export jit|onnx|all`，或单独调用验证脚本。
导出文件保存于独立目录，不覆盖checkpoint；二进制观测和导出物被gitignore排除，JSON报告纳入提交。
修改 `utils/export_policy.py`、`utils/__init__.py` 和 `play.py`，新增 `validate_export.py`、`standalone_inference.py`。
原始日志：`/tmp/himloco_export_lab.log`、`/tmp/himloco_export_legacy.log`。

"""Render the locked results and a standalone scientific figure."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ROOT=Path('artifacts/impact/observability_v1')
SPEC=json.loads(Path('configs/impact/observability_v1.json').read_text())


def write_report():
    payload=json.loads((ROOT/'summary.json').read_text());summary=payload['summary']
    labels=[('phase','bce','Phase proxy'),('kinematics','bce','Joint kinematics'),
            ('full','bce','Full BCE'),('full','mse','Full continuous MSE')]
    low=min(row['matched_auc']['CI95'][0] for row in summary.values())
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True,sharey=True)
    for axis,(role,domain) in zip(axes.flat,[('test','seen'),('test','heldout'),('high_speed','seen'),('high_speed','heldout')]):
        for kind,method,label in labels:
            rows=[summary[f'{role}/{domain}/{kind}/{method}/{h}']['matched_auc'] for h in range(6)]
            axis.plot(range(6),[r['mean'] for r in rows],marker='o',label=label)
            axis.fill_between(range(6),[r['CI95'][0] for r in rows],[r['CI95'][1] for r in rows],alpha=.12)
        axis.axhline(.5,color='grey',linestyle=':');axis.axvline(2.5,color='grey',linestyle='--')
        axis.set_title(f'{role}: {domain} terrain');axis.set_ylim(max(0,min(.45,low-.03)),1.01);axis.grid(alpha=.2)
        axis.set_xticks(range(6),['pre40','pre20','pre-last','post-first','post20','post40'],rotation=20)
    axes[0,0].legend(fontsize=8);axes[0,0].set_ylabel('Within-stratum AUROC');axes[1,0].set_ylabel('Within-stratum AUROC')
    fig.suptitle('Go1 contact severity: phase-controlled discrimination vs causal cutoff\n5 independent data replications; shading = paired-data mean 95% t interval')
    fig.tight_layout();fig.savefig(ROOT/'observability_results.png',dpi=180);plt.close(fig)
    def value(key,metric='matched_auc'):
        row=summary[key][metric];return f"{row['mean']:.4f} [{row['CI95'][0]:.4f}, {row['CI95'][1]:.4f}]"
    primary=payload['primary']
    allmeta={}
    for rep in range(1,6):
        for role in ('train','validation','test','high_speed'):
            allmeta[f'{rep}/{role}']=json.loads((ROOT/f'rep{rep}/{role}/data/metadata.json').read_text())
    counts={role:sum(m['counts']['kept'] for key,m in allmeta.items() if key.endswith('/'+role)) for role in ('train','validation','test','high_speed')}
    censored=sum(m['counts']['outcome_censored'] for m in allmeta.values())
    term=sum(m['terminations'] for m in allmeta.values())
    text=[
        '# Go1 冲击严重度：接触时序与额外本体信息验证结果','',
        '状态：正式实验完成。4096 总环境，五组独立采集 × 两次初始化，三个输入组、四种方法、六个截止时刻，共 240 个独立 MLP。所有权重锁定后才评分最终测试。','',
        '## 预设主比较','',
        f"正常速度、训练排除的随机块、首次接触后可用策略边界处：完整 BCE − 相位代理 BCE 的分层内 AUC 差为 **{primary['mean']:.4f} [{primary['CI95'][0]:.4f}, {primary['CI95'][1]:.4f}]**。预设实用门槛为差值的 95% CI 下界 >0.03；本轮**{'通过' if payload['primary_passed'] else '未通过'}**。五组配对差为 `{primary['replications']}`。",'',
        '这一比较只检验超出所选运动学相位代理的额外信息；代理不是所有可能时序信息的充分统计量，不能声称彻底排除了时序解释。完整本体的关节速度、姿态、动作等也可能编码地形或接触动态，本轮不将其归因于单一通道。','',
        '## 实验方式与时间语义','',
        '冻结原 HIMLoco-Go1 PD actor、HIMEstimator、Kp40/Kd1、动作尺度0.25，不更新 PPO，不加入控制或 quiet reward。原 baseline 200 Hz 配置保持原样，本研究采集 400 Hz 物理、50 Hz 本体输入。训练速度0.4–0.9 m/s，高速留出1.0–1.1 m/s；每次 reset 采样指令。七类地形共享84个几何块，六类训练，随机块排除；四秒 episode 约束移动距离。4096不是每类地形各4096，也不是4096份独立几何。','',
        '主指标包含机器人经过起始平台、进入地形的自然接触轨迹，没有单独按障碍接触子集验收；随机块类别的测试事件不保证每一次都实际踩到凸块。因此本轮不能声称在纯障碍碰撞子集上验证了同等收益。','',
        '接触起点是同足连续40 ms不超过5 N之后，连续两个400 Hz样本超过5 N，取第一个时刻。结果为起点后40 ms内的20 ms平均载荷正向20 ms差分峰值，事件阈值2000 N/s。它是仿真有限时间尺度法向合力代理，不是原生瞬时导数、完整摩擦合力或硬件力真值。','',
        '这些例子均由 privileged 接触起点离线对齐；模型输入不含接触真值、地形ID或未来量。这是**事件对齐的可观测性诊断**，尚不提供部署时事件触发器、连续报警 FAR 或控制收益。不能将本轮事件窗口AP与过去全区间AP直接比较，因为标签窗口与事件比例不同。','',
        '特别地，接触前AUC是在离线已经知道这些窗口对应即将接触的条件下比较强弱接触，不能当作连续行走中提前发现未知接触的能力。六种截止分别训练独立头，部署时尚缺少对应的因果接触阶段定位。','',
        '| 输入截止 | 实际相对接触时刻范围 | 含义 |','|---|---|---|',
        '| pre40 | −60 至 −42.5 ms | 最后接触前边界的前两步 |',
        '| pre20 | −40 至 −22.5 ms | 最后接触前边界的前一步 |',
        '| pre-last | −20 至 −2.5 ms | 严格早于本轮5 N定义的接触起点 |',
        '| post-first | 0 至 17.5 ms | 第一帧不早于接触起点 |',
        '| post20 | 20 至 37.5 ms | 已观察到部分甚至接近完整结果窗 |',
        '| post40 | 40 至 57.5 ms | 结果窗已结束，仅为事后可观测性参考 |','',
        '接触前只指本轮力阈值和采样分辨率定义之前，不等于几何上严格无接触；本轮也未计入硬件测量陈旧、推理和通信时延。接触后早期分数可能同时利用已出现的冲击信息，不能将整窗识别称为未来剩余峰值预测。','',
        '## 输入与公平对照','',
        '相位代理使用六帧四足髋俯仰位置/速度的单位圆表示，丢弃幅度，并加入当前指令；训练集拟合中心和尺度。关节组使用六帧完整关节位置/速度及当前指令；完整组使用270D本体历史，包括角速度、重力方向和动作。各组MLP隐藏层不同以近似匹配参数，实际参数数列在训练报告；不是每个架构都独立HPO后的最优上限。','',
        '每组、每个截止和初始化独立参数/优化状态；完整组同时对照相同连续严重度的MSE。统一自然事件采样、batch1024、30,000更新和学习率日程；采用固定最后一步，不按验证或测试挑最好权重。训练曲线保留，有限预算不代表信息论极限。','',
        '## 主时刻的四类泛化条件','',
        '| 测试条件 | 相位 BCE | 关节 BCE | 完整 BCE | 完整 MSE |','|---|---|---|---|---|']
    for role,domain in [('test','seen'),('test','heldout'),('high_speed','seen'),('high_speed','heldout')]:
        text.append('| '+role+'/'+domain+' | '+' | '.join(value(f'{role}/{domain}/{k}/{m}/3') for k,m,_ in labels)+' |')
    text+=['','表为分层内AUROC均值及95%区间。分层使用足、接触前相位八分箱、指令速度四分箱及地形类别；类别仅用于离线评价。每层至少五个正事件和五个负事件，只统计层内正负对。高速指令落入最外侧速度箱，没有制造训练中不存在的相位观测。','',
           '## 未见随机块：随截止时刻变化','',
           '| 截止 | 相位 BCE AUC | 关节 BCE AUC | 完整 BCE AUC | 完整 MSE AUC | 完整 BCE AP |','|---|---|---|---|---|---|']
    for h,name in enumerate(SPEC['cutoff_names']):
        text.append('| '+name+' | '+' | '.join(value(f'test/heldout/{k}/{m}/{h}') for k,m,_ in labels)+
                    ' | '+value(f'test/heldout/full/bce/{h}','AP')+' |')
    text+=['','![分层内可辨识性曲线](../artifacts/impact/observability_v1/observability_results.png)','',
           '全条件、各初始化、分层覆盖率、校准Brier、弱接触FPR及召回、MSE整体/尾部RMSE见 `artifacts/impact/observability_v1/summary.json` 及各rep结果。AP依赖事件比例，不同域不能只凭AP高低判断迁移是否更好。','',
           '## 数据规模、审计与统计边界','',
           f'五组保留事件数：{counts}。完整结果窗被终止或记录尾部截断的事件共{censored}，另排除每次reset后500 ms；正式采集合计非超时终止{term}。候选起点、初始排除、删失和各截止有效数量都保留在数据metadata。未满足40 ms低载荷间隔的擦碰/短重接触不在主事件范围，不能外推为所有impact类型。','',
           '21/4096个环境的完整原生接触和本体审计片段按固定env ID保留。原采集器检查历史、原/新增接触传感器一致性和终止捕获；独立审计从原生向量重算事件起点、严重度和reset边界。float32模长缓存产生的微小差异以N/s报告，允许误差0.01 N/s。','',
           '先平均两个网络初始化，再对五组独立采集计算t区间。大量事件、四足和共享地形机器人不作为独立科学重复。只有预设单一主比较作确认性判定，其余域、时刻与方法差值为诊断，没有对大量次级比较作多重校正。五组仍是小样本。','',
           '验证轨迹按tile_id%3分为概率校准、阈值选择、独立验证，仅使用训练地形类别。阈值取验证普通接触分数的97.5%分位；最终测试不再重选。这里FPR的负例为普通接触事件，绝不是每分钟连续流误报率，也没有未知域概率或误报保证。','',
           '## 复现与保存','',
           '协议：`configs/impact/observability_v1.json`；启动：`python scripts/impact/run_observability.py`，conda环境 `himloco_lab`。训练与测试脚本分离，`model_lock.json`保存240模型、方法源码和协议的哈希。权重位于 `logs/impact_observability_v1/rep*/{phase,kinematics,full}/models.pt`。','',
           '上一阶段方法、结果与有效权重保持独立。本轮大体积原始分片逐角色转换后删除；评分结束后按保存索引清理全量本体帧，保留事件表、预测、审计片段、权重、报告与哈希。事件表和小审计不替代完整本体测试集，清理后逐帧重训/预测需要重新采集，PhysX不保证逐比特一致。','']
    Path('docs/GO1_IMPACT_OBSERVABILITY_RESULTS.md').write_text('\n'.join(text))
    print('OBS_REPORT_WRITTEN',flush=True)


if __name__=='__main__':write_report()

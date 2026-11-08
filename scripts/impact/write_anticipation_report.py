"""Render final future-risk/shared-HIM evidence without choosing new points."""
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from collect_anticipation import ART,LOG,SPEC
from evaluate_anticipation import ci,dump


def label(group,loss,selected):
    return f'joint_{loss}_lambda{selected[loss]:g}' if group=='joint' else f'{group}_{loss}'


def main():
    s=json.loads((ART/'summary.json').read_text());spec=json.loads(SPEC.read_text())
    selected=s['selected_joint_lambda'];groups=s['groups']
    lines=['# Go1未来落足风险与HIM共享主干：实验结果','',
        f"完成日期：{datetime.now(ZoneInfo('Asia/Shanghai')).date()}。分支`feat/go1-impact-anticipation`。",'',
        '**任务：当前六帧本体历史，预测未来80ms内下一次合格落足是否强加载；不是接触后识别。**',
        '主统计要求扣除5ms主机推理预算后，距5N定义的落足起点至少20ms。尚未完成几何接触/机载执行器验收，无下游控制收益主张。','',
        '## 实验条件与实际增加的模块','',
        '完整原HIMLoco JIT冻结并采集轨迹。影子网络复制原HIM的270→128→64主干，增加64→64→4冲击head；所有head不参与采集动作。A专用、B冻结读出、C联合HIM原损失、D原HIM继续更新但截断impact梯度。',
        '分别训练BCE、同二值目标的概率MSE/Brier、未来落足严重度MSE。C权重0.1/1/10全部保留；主机制比较只用独立开发集预选权重。补充原始历史从头BCE、相位BCE和19维冻结读出诊断。',
        'A/从头模型有效训练参数47364，B的64维head为4420，C/D含原HIM路径为64295；相位模型41156，比完整历史从头模型少约13.1%。核心四组impact结构相同，相位对照并非严格等参数。所有目标采用同一LR/更新预算；本轮不证明每种目标已经达到各自最优可达性能。',
        '4096总环境，400Hz物理、50Hz输入/策略；5组独立数据×2初始化。六地形家族训练、箱块留出，正常0.4–0.9m/s，提速1.0–1.1m/s。每个模型60000更新、batch2048；共230个正式未来模型。',
        '平地、1–4cm粗糙、5–25%上下坡、2–10cm上下台阶构成已见家族；2–10cm箱块仅对本轮预测器留出，不保证原locomotion策略从未见过类似地形。固定质量/摩擦、不加推搡；保留观测噪声和小幅重置姿态/关节扰动。',
        '每复制train2500、validation/test/high_speed各600策略步，总88,064,000个机器人区间。共享84地形块，四足与4096环境均不作为额外独立数据复制。',
        '评分仅含每episode至少500ms历史、且后续120ms标签完整的有效区间；启动与结束附近被删失。误报预算对应这些有效区间，不能直接当作覆盖启动/跌倒等完整部署过程的误报保证。推理本身不读取未来，删失只用于离线评分。',
        '接触法向合力模长先20ms均值，再取20ms滞后正差分；落足后40ms峰值>2000N/s定义强加载。这不是完整摩擦合力、原生数值导数或损伤阈值。','',
        '## 预设主判据','',
        f"原始历史从头BCE相对相位BCE的提前检出率差为{s['primary_difference']['mean']*100:.2f}个百分点，95%CI[{s['primary_difference']['lower']*100:.2f}, {s['primary_difference']['upper']*100:.2f}]。主判据{'通过' if s['primary_pass'] else '未通过'}：要求CI下界>3个百分点且双方测试FAR上界≤1次/机器人分钟。",'',
        '| 方法 | 正常已见家族检出率 | 正常箱块检出率 | 正常箱块FAR/min | 提速箱块检出率 | 提速箱块FAR/min |',
        '|---|---:|---:|---:|---:|---:|']
    methods=['phase_bce','scratch_bce']+[label(g,l,selected) for g in ('dedicated','frozen','joint','moving') for l in ('bce','brier','mse')]
    for method in methods:
        a=groups[f'test_seen/20/0/{method}'];b=groups[f'test_heldout/20/0/{method}'];c=groups[f'high_speed_heldout/20/0/{method}']
        lines.append(f"| {method} | {a['recall']['mean']*100:.2f}% | {b['recall']['mean']*100:.2f}% | {b['far']['mean']:.3f} | {c['recall']['mean']*100:.2f}% | {c['far']['mean']:.3f} |")
    lines+=['','表中均使用验证锁定阈值，FAR不是各方法相等的实际误报率；预算是否达标以summary中的保守上界为准，不只看均值。四足合计每机器人分钟。额外10/20ms延迟与40ms提前档位完整保留，不能在测试后替换主档位。','',
        '## 共享机制对比：次级描述性结果','',
        '| 目标/损失 | 对比 | 检出率差/百分点，95%CI | 双方FAR预算通过 |','|---|---|---:|---|']
    for name,result in s['shared_comparisons'].items():
        d=result['difference'];loss,comparison=name.split('/')
        lines.append(f"| {loss} | {comparison} | {d['mean']*100:.2f} [{d['lower']*100:.2f}, {d['upper']*100:.2f}] | {result['both_far_pass']} |")
    d=s['sharing_by_target_interaction']
    lines+=['',f"共享条件是否扩大BCE相对连续MSE的报警优势：交互差{d['mean']*100:.2f}个百分点，95%CI[{d['lower']*100:.2f}, {d['upper']*100:.2f}]。这是目标与共享条件的组合，不是纯BCE/MSE损失归因；同二值目标的Brier对照需同时查看。",'',
        '## 分类目标与损失函数分别比较','',
        '以下都是固定主档位的描述性成对比较。BCE与Brier使用完全相同的二值监督，区别在损失；BCE与连续MSE还改变了监督目标，不能只归因于MSE优化。只有双方误报预算达标，检出率差才可用来比较报警效果。','',
        '| 组 | 对比 | 检出率差/百分点，95%CI | 双方FAR预算通过 |','|---|---|---:|---|']
    loss_comparisons={}
    for group in ('dedicated','frozen','joint','moving'):
        a=groups[f'test_seen/20/0/{label(group,"bce",selected)}']
        for loss in ('brier','mse'):
            b=groups[f'test_seen/20/0/{label(group,loss,selected)}']
            delta=ci(np.array(a['recall']['replication_values'])-b['recall']['replication_values'])
            passed=max(a['far_upper_conservative'],b['far_upper_conservative'])<=1
            loss_comparisons[f'{group}/bce-{loss}']=dict(difference=delta,both_far_pass=passed)
            lines.append(f"| {group} | BCE−{loss.upper()} | {delta['mean']*100:.2f} [{delta['lower']*100:.2f}, {delta['upper']*100:.2f}] | {passed} |")
    lines+=['',
        '## 连续未来严重度的数值精度','',
        '这里回归的是下一次落足后40ms最大平滑loading-rate，无80ms内落足时为0；不是瞬时力或过去20ms加载率。排序和数值指标使用固定seed9861抽取的每域至多100000机器人区间，所有模型共享样本；主连续报警统计仍使用全量有效时刻。','',
        '| MSE组 | 正常已见家族RMSE N/s | 正常箱块RMSE N/s | 箱块>4000N/s子集MAE N/s |','|---|---:|---:|---:|']
    for group in ('dedicated','frozen','joint','moving'):
        method=label(group,'mse',selected)
        a=s['ranking'][f'test_seen/{method}'];b=s['ranking'][f'test_heldout/{method}']
        lines.append(f"| {group} | {a['RMSE_n_per_s']['mean']:.2f} | {b['RMSE_n_per_s']['mean']:.2f} | {b['tail_gt4000_MAE_n_per_s']['mean']:.2f} |")
    lines+=['','## 原HIM能力与策略输入变化','',
        '下表为最终验证role0固定样本、两初始化先平均后跨五组汇总。动作RMS变化是在同一历史上将影子速度/latent交给原Actor所得，不是闭环行走性能。swap loss和梯度方向仅作诊断，不据此单独宣称因果干扰。','',
        '| 组/BCE | 速度RMSE | Actor动作RMS变化 |','|---|---:|---:|']
    preservation={}
    for group in ('dedicated','frozen','joint','moving'):
        values={k:[] for k in ('velocity_rmse','actor_action_RMS_shift')}
        for rep in range(1,6):
            final=json.loads((LOG/f'rep{rep}/curve.json').read_text())[-1]['models']
            rows=[m for m in final if m['entry']['group']==group and m['entry']['loss']=='bce' and
                  (group!='joint' or m['entry']['lambda']==selected['bce'])]
            for metric in values:values[metric].append(float(np.mean([m[metric] for m in rows])))
        preservation[group]={k:ci(v) for k,v in values.items()}
        lines.append(f"| {group} | {np.mean(values['velocity_rmse']):.4f} | {np.mean(values['actor_action_RMS_shift']):.4f} |")
    lines+=['','A为独立预测副本；其原HIM输出变化不判定冲击任务成败，也意味着不能直接用A替换原Estimator。','',
        '全部联合权重的敏感性结果如下。开发选择只最小化冲击目标损失，没有同时约束原HIM能力，因此被选中不等于可安全接回Actor。','',
        '| 联合目标 | λ | 正常已见检出率 | FAR保守上界/min | 验证速度RMSE | Actor动作RMS变化 |',
        '|---|---:|---:|---:|---:|---:|']
    tradeoff={}
    for loss in ('bce','brier','mse'):
        for weight in spec['lambda_candidates']:
            method=f'joint_{loss}_lambda{weight:g}';alarm=groups[f'test_seen/20/0/{method}']
            values={k:[] for k in ('velocity_rmse','actor_action_RMS_shift')}
            for rep in range(1,6):
                final=json.loads((LOG/f'rep{rep}/curve.json').read_text())[-1]['models']
                rows=[m for m in final if m['entry']['group']=='joint' and m['entry']['loss']==loss and m['entry']['lambda']==weight]
                for metric in values:values[metric].append(float(np.mean([m[metric] for m in rows])))
            tradeoff[method]={k:ci(v) for k,v in values.items()}
            lines.append(f"| {loss} | {weight:g} | {alarm['recall']['mean']*100:.2f}% | {alarm['far_upper_conservative']:.3f} | {np.mean(values['velocity_rmse']):.4f} | {np.mean(values['actor_action_RMS_shift']):.4f} |")
    lines+=['','## 已完成20ms的阳性诊断','',
        '第一复制另外以同样原始历史估计刚完成20ms内最大平滑loading-rate，使用相同四组及三类目标。只有一组数据×两初始化，不把它写成五复制的泛化结论。','',
        '| 组 | MSE RMSE N/s | MSE排序AP | BCE排序AP |','|---|---:|---:|---:|']
    final=json.loads((LOG/'rep1/current_diagnostic/curve.json').read_text())[-1]['models']
    for group in ('dedicated','frozen','joint','moving'):
        chosen={}
        for loss in ('mse','bce'):
            r=[m for m in final if m['entry']['group']==group and m['entry']['loss']==loss and
               (group!='joint' or m['entry']['lambda']==selected[loss])]
            chosen[loss]={k:float(np.mean([m[k] for m in r])) for k in ('AP',*(('severity_RMSE_n_per_s',) if loss=='mse' else ()))}
        lines.append(f"| {group} | {chosen['mse']['severity_RMSE_n_per_s']:.2f} | {chosen['mse']['AP']:.4f} | {chosen['bce']['AP']:.4f} |")
    numerical=s['ranking']['test_seen/dedicated_mse']
    alarm_delta=loss_comparisons['dedicated/bce-mse']
    same_target_delta=loss_comparisons['dedicated/bce-brier']
    shared_delta=s['shared_comparisons']['bce/C-A']['difference']
    lines+=['','## 本轮可以回答到哪一步','',
        f"1. **MSE是否完全学不会：**A连续未来严重度RMSE为{numerical['RMSE_n_per_s']['mean']:.1f}N/s，训练均值基线为{numerical['training_mean_baseline_RMSE_n_per_s']['mean']:.1f}N/s，零基线为{numerical['zero_baseline_RMSE_n_per_s']['mean']:.1f}N/s。结合上面的接触后诊断，数值学习能力应按目标和时窗判断；平均RMSE改善仍不等于尾部足够精确。",'',
        f"2. **数值估计与风险报警是否等价：**A的BCE减连续MSE及时检出率差为{alarm_delta['difference']['mean']*100:.2f}个百分点，95%CI[{alarm_delta['difference']['lower']*100:.2f}, {alarm_delta['difference']['upper']*100:.2f}]，双方误报预算通过={alarm_delta['both_far_pass']}。这比较的是二值风险目标与连续严重度目标两种完整方案。",'',
        f"3. **是否只是MSE损失的问题：**同二值标签下，A的BCE减Brier差为{same_target_delta['difference']['mean']*100:.2f}个百分点，95%CI[{same_target_delta['difference']['lower']*100:.2f}, {same_target_delta['difference']['upper']*100:.2f}]，双方误报预算通过={same_target_delta['both_far_pass']}。Brier本身就是概率MSE，需与连续目标比较分开解读。",'',
        f"4. **共享是否复现旧问题：**BCE的C−A差为{shared_delta['mean']*100:.2f}个百分点，95%CI[{shared_delta['lower']*100:.2f}, {shared_delta['upper']*100:.2f}]。只支持当前目标、权重网格、预算和固定策略数据中的共享判断；不能当作旧实验的唯一归因。",'',
        f"5. **是否已经证明可用于提前控制：**预设未来主判据{'通过' if s['primary_pass'] else '未通过'}。无论分类方案相对回归的差异如何，本轮均没有控制干预、执行器响应或真实损伤结果，因此尚未证明动作收益。"]
    lines+=['','## 证据边界与复现','',
        '共享在当前Go1任务中的影响不能自动还原旧实验的唯一原因；旧head接入层、监督时窗、尺度、更新安排和闭环均可能不同。19维补充读出有效head参数较少，也不能独自证明不可逆信息瓶颈。',
        'D−B还混合了最终表征改变和训练中head追踪移动特征的影响，不能独自确诊追踪滞后。若出现下降，需冻结D的最终trunk重新训练head作进一步干预；梯度负余弦同样不是冲突致损的充分证据。',
        '未进行PPO重训、Replay、新动作控制或真机部署。未来标签只能成熟后用于监督，部署推理不读取它。高AP不等于低误报、有效提前量或动作收益。',
        '验证选点采用FAR上界≤0.5/min；正式所有模型/工作点锁定后才采集测试。主差先平均两个初始化，再对五数据组计算t区间；多项共享机制结果为描述性次级比较，没有在测试中挑显著时刻。',
        '启动：`conda run -n himloco_lab python scripts/impact/run_anticipation.py`。有效权重：`logs/impact_shared_anticipation_v1/rep*/{models,phase}.pt`。原始大数组已按逐角色审计清单清理，完整重训需重新采集。',
        '[协议](GO1_IMPACT_SHARED_REPRESENTATION_PLAN.md)；[主汇总](../artifacts/impact/shared_anticipation_v1/summary.json)；[模型/工作点锁](../artifacts/impact/shared_anticipation_v1/model_lock.json)；[开发选择](../artifacts/impact/shared_anticipation_v1/development/decision.json)。']
    dump(ART/'him_preservation.json',preservation)
    dump(ART/'loss_comparisons.json',loss_comparisons)
    dump(ART/'joint_tradeoff.json',tradeoff)
    Path('docs/GO1_IMPACT_SHARED_ANTICIPATION_RESULTS.md').write_text('\n'.join(lines)+'\n')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained');x=np.arange(4);gs=['dedicated','frozen','joint','moving']
        for j,loss in enumerate(('bce','brier','mse')):
            vals=[groups[f'test_seen/20/0/{label(g,loss,selected)}'] for g in gs]
            means=np.array([v['recall']['mean'] for v in vals])*100
            lo=np.array([v['recall']['lower'] for v in vals])*100;hi=np.array([v['recall']['upper'] for v in vals])*100
            bars=axes[0].bar(x+(j-1)*.23,means,.23,label=loss.upper(),yerr=np.stack((means-lo,hi-means)),capsize=2)
            for bar,value in zip(bars,vals):
                if value['far_upper_conservative']>1:bar.set_hatch('//');bar.set_alpha(.45)
        axes[0].set(xticks=x,xticklabels=gs,ylabel='Timely event recall (%)',title='Seen families, lead >=20ms; hatched = FAR fail');axes[0].legend()
        vals=[s['ranking'][f"test_seen/{label(g,'mse',selected)}"]['RMSE_n_per_s'] for g in gs]
        means=np.array([v['mean'] for v in vals]);lo=np.array([v['lower'] for v in vals]);hi=np.array([v['upper'] for v in vals])
        axes[1].bar(x,means,yerr=np.stack((means-lo,hi-means)),capsize=3)
        axes[1].set(xticks=x,xticklabels=gs,ylabel='Future severity RMSE (N/s)',title='Same future target, continuous MSE')
        fig.suptitle('Next touchdown <=80ms; 5ms inference allocation; no intervention tested')
        fig.savefig(ART/'sharing_results.png',dpi=180);plt.close(fig)
    except ImportError:
        pass
    print('ANTICIPATION_REPORT_WRITTEN',flush=True)


if __name__=='__main__':main()

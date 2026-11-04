"""Natural-prevalence ranking, calibration, and causal event alarms."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def average_precision(y, score):
    y, score = np.asarray(y).ravel().astype(bool), np.asarray(score).ravel()
    if not y.any():
        return None
    order = np.argsort(-score, kind='stable')
    ys, ss = y[order], score[order]
    ends = np.r_[np.flatnonzero(ss[:-1] != ss[1:]), len(ss) - 1]
    tp = np.cumsum(ys, dtype=np.float64)[ends]
    precision = tp / (ends + 1)
    return float(np.sum(np.diff(np.r_[0, tp]) * precision) / y.sum())


def fit_calibration(score, y):
    s, target = np.asarray(score, np.float64).ravel(), np.asarray(y, np.float64).ravel()
    def objective(params):
        slope = np.exp(params[0])
        logits = slope * s + params[1]
        residual = expit(logits) - target
        return (np.logaddexp(0, logits) - target * logits).mean(), np.array([
            np.mean(residual * slope * s), np.mean(residual)])
    result = minimize(objective, [0., 0.], method='L-BFGS-B', jac=True,
                      bounds=[(-6, 6), (-20, 20)], options={'maxiter': 150})
    return {'slope': float(np.exp(result.x[0])), 'intercept': float(result.x[1]),
            'success': bool(result.success), 'loss': float(result.fun)}


def calibrate(score, params):
    return expit(params['slope'] * score + params['intercept'])


def frame_metrics(severity, score, probability, point=None, threshold=40.):
    y = severity > threshold
    p = np.asarray(probability)
    bins = np.minimum((p * 10).astype(int), 9)
    ece = sum(float(np.mean(bins == b)) * abs(float(p[bins == b].mean()) - float(y[bins == b].mean()))
              for b in range(10) if np.any(bins == b))
    result = {'per_foot_AP': average_precision(y, score), 'prevalence': float(y.mean()),
              'Brier': float(np.mean((p - y) ** 2)), 'ECE_10': ece,
              'constant_Brier_reference': float(y.mean() * (1 - y.mean())),
              'rows': int(len(y)), 'positive_foot_intervals': int(y.sum())}
    if point is not None and np.isfinite(point).all():
        error = point - severity
        result.update(MAE_n=float(np.abs(error).mean()), RMSE_n=float(np.sqrt((error ** 2).mean())),
                      negative_prediction_fraction=float(np.mean(point < 0)))
        result['severity_bins_MAE_n'] = [float(np.abs(error[(severity >= lo) & (severity < hi)]).mean())
            if np.any((severity >= lo) & (severity < hi)) else None
            for lo, hi in [(0, 20), (20, 40), (40, 80), (80, np.inf)]]
    return result


def truth_event_ids(severity, valid, threshold=40., merge_gap=2):
    """Offline truth grouping. IDs are globally unique; invalid periods break events."""
    y = (severity > threshold) & valid[..., None]
    filled = y.copy()
    # Fill at most merge_gap negative policy intervals, without filling reset masks.
    for gap in range(1, merge_gap + 1):
        bridge = y[:-gap-1] & y[gap+1:]
        for offset in range(1, gap+1):
            bridge &= valid[offset:offset+len(bridge), :, None]
        for offset in range(1, gap+1):
            filled[offset:offset+len(bridge)] |= bridge
    previous = np.concatenate((np.zeros_like(filled[:1]), filled[:-1]))
    starts = filled & ~previous
    ids = np.zeros_like(filled, dtype=np.int32)
    next_id = 1
    current = np.zeros(filled.shape[1:], np.int32)
    for t in range(len(filled)):
        count = int(starts[t].sum())
        current[starts[t]] = np.arange(next_id, next_id + count)
        current[~filled[t]] = 0
        ids[t] = current
        next_id += count
    return ids, starts, next_id - 1


def event_metrics(probability, threshold, valid, truth_ids, starts, policy_dt=.02,
                  quiet_steps=5, task='detection', precontact=None, onset_ms=None):
    """Alarm = causal up-crossing after 100 ms with no prior up-crossing.

    Same-foot one-to-one overlap with truth intervals; duplicate alarms count as
    false alarms. This quiet-period rule depends only on previous raw crossings.
    """
    above = (probability >= threshold) & valid[..., None]
    previous = np.concatenate((np.zeros_like(above[:1]), above[:-1]))
    crosses = above & ~previous
    alarm = crosses.copy()
    for lag in range(1, quiet_steps + 1):
        alarm[lag:] &= ~crosses[:-lag]
    when, env, foot = np.where(alarm)
    matched_ids = truth_ids[when, env, foot]
    positive = matched_ids > 0
    unique, first = np.unique(matched_ids[positive], return_index=True)
    matched = len(unique)
    total_events = int(starts.sum())
    false = len(when) - matched
    exposure_min = float(valid.sum() * policy_dt / 60)
    result = {'events': total_events, 'alarms': int(len(when)), 'matched_events': matched,
              'event_recall': matched / total_events if total_events else None,
              'false_alarms': int(false), 'robot_minutes': exposure_min,
              'false_alarms_per_robot_minute': false / exposure_min if exposure_min else None,
              'probability_threshold': float(threshold)}
    if matched:
        positive_indices = np.flatnonzero(positive)[first]
        st, se, sf = np.where(starts)
        start_by_id = np.zeros(int(truth_ids.max()) + 1, np.int32)
        start_by_id[truth_ids[st, se, sf]] = st
        delay_steps = when[positive_indices] - start_by_id[unique]
        result['interval_detection_delay_ms_median'] = float(np.median((delay_steps + (task == 'detection')) * policy_dt * 1000))
    if task == 'forecast' and precontact is not None:
        strict_starts = starts & precontact & np.isfinite(onset_ms)
        strict_ids = truth_ids[strict_starts]
        alarm_pre = positive & precontact[when, env, foot] & np.isfinite(onset_ms[when, env, foot])
        strict_matched = np.intersect1d(matched_ids[alarm_pre], strict_ids)
        result['strict_precontact_events'] = int(len(strict_ids))
        result['strict_precontact_recall'] = len(strict_matched) / len(strict_ids) if len(strict_ids) else None
        leads = onset_ms[when[alarm_pre], env[alarm_pre], foot[alarm_pre]]
        result['precontact_lead_ms_median_before_runtime_delay'] = float(np.median(leads)) if len(leads) else None
    return result

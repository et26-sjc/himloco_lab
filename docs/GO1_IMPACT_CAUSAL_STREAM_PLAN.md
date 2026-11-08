# Go1 causal stream experiment: locally locked protocol

The preceding study showed conditional severity discrimination, with ground-truth landing onset selecting observations. This study asks whether that information survives causal triggering in uninterrupted proprioceptive streams. It does not modify locomotion rewards, PPO, the estimator controlling the policy, or PD gains. It does not revisit replay until there is a useful continuous task to retain.

## Task and terrain

Use 4096 environments, the existing frozen Go1 policy, 400 Hz physics and 50 Hz observations. Seven terrain families share 84 generated tiles: plane, rough ground, ascending/descending slopes, ascending/descending stairs, and held-out boxes. Train new heads on families 0–5 only. Evaluate new trajectories on seen families and boxes, at normal and higher speeds. Boxes are held out from this estimator, not necessarily from the original locomotion policy. Four-second episodes limit travel to each tile; starting platforms remain part of the distribution. Five fresh data seeds and two optimization seeds prevent a single lucky run from supporting the conclusion.

The monitored event is explicitly a qualified landing: at least 40 ms below 5 N, followed by two native samples above 5 N. Its severity is the previous study's filtered loading-rate maximum in the following 40 ms. A strong event exceeds 2000 N/s. This is a simulation normal-force-resultant proxy, not raw force derivative or hardware ground truth, and does not cover all short recontacts or impacts during stance.

## Comparisons and causal timing

1. Privileged, causally confirmed contact gate plus the previous frozen severity head diagnoses the severity limit with contact timing available. Confirmation waits for the second high sample; it is not a deployable method or a mathematical upper bound.
2. A learned contact-state gate plus exactly the same frozen severity head tests a modular stage/severity design. A learned onset-pulse gate is a secondary comparison.
3. A small per-leg linear contact model using q, dq and a PD response proxy tests whether a complex phase network is necessary. The proxy is computed from existing action and joint observations; it is not a torque sensor.
4. Direct continuous risk heads, with both a fresh initialization and a matched old-severity warm start, provide a serious alternative to modular gating. BCE and warm-start MSE use the same recent-landing target semantics. Applying the frozen conditional head everywhere is an additional diagnostic baseline.

Every model sees six 45-D proprioceptive frames available at the actual decision boundary, never future frames or privileged contact. Warm starts retain the original normalization and parameters; fresh Adam state is explicit. New models have a fixed 30000-update budget with batch1024, and the final checkpoint is used. Older severity weights remain frozen. Gated models cost two networks, which must be disclosed rather than claiming equal inference cost.

Evaluate complete valid streams, including background, with a common 100 ms quiet-before-alarm rule. Match a same-foot alarm only in [onset,onset+40ms); pre-onset and late alarms do not count as successful monitoring. Report strong-event recall, false alarms per robot-minute (all four feet combined), timing, per-foot/tile results, and whether an alarm preceded the native force maximum. Evaluation excludes burn-in and intervals whose future label is censored, but the causal state machine runs through those intervals and resets only at a known episode reset. Repeat with a complete observation packet delayed by 20 ms; 5/10 ms sensor delay cannot be represented by this 50 Hz recording.

## Selection and inference

Validation seen-family tiles are divided by tile_id modulo3: role0 selects gate parameters by qualified-onset F1, role1 selects severity thresholds, role2 is untouched verification. Select thresholds by maximum recall under tile-bootstrap95% upper FAR<=0.5/min; include a disabled threshold and report its zero recall honestly. No final-test labels choose parameters. All five replications' model and operating-point hashes are locked before final scoring.

The sole superiority endpoint is normal-speed seen-terrain recall gain of state-gated BCE over warm-start direct BCE at zero delay: paired t95% CI across five data replications after averaging initializations, lower bound >.03, and both methods' test FAR upper<=1/min. Practical usefulness additionally requires normal-seen recall lower>=.7 and FAR upper<=1/min across all four domains. Failure is informative: decompose missed strong landings into missing timely phase candidates and insufficient severity score. Report uncertainty over data seeds and terrain tiles; thousands of robot copies on the same tile are not thousands of independent terrain samples. Bootstrap zero observed false alarms is not proof of zero risk.

## Possible research contribution and limits

A useful result would establish a reproducible, causal phase/severity module with an explicit monitoring budget and timing limit. A negative result would locate why event-conditioned accuracy overstates continuous usability and motivate observation or target changes. The story is falsifiable and does not require classification, modular gating or replay to win. This stage establishes neither control benefit, true pre-impact prediction, hardware transfer, nor replay effectiveness. Keep old methods/data intact. Preserve small complete-input re-inference fixtures and native force audits; delete only newly generated bulk caches after verified archival.

## Prospective secondary-control addition, 2026-11-08 10:23 UTC

Before collecting or scoring any final test, add the same state/oracle gates applied to the already specified warm-start continuous BCE/MSE heads. This requires no new training and separates the effect of the gate from the effect of retaining an event-conditioned frozen severity head. The original sole primary endpoint and training configuration are unchanged. Replication1 training/calibration had completed; only completion logs and file presence were inspected, not its validation metrics. Its operating points are recomputed with these extra controls before the all-model lock. This is an explicitly recorded local amendment, not a claim of an externally preregistered study.

# Potential claims and falsifiable tests

No item below is a current conclusion.  Each row states a possible claim, the
minimum comparison that could falsify it, and evidence that does **not** count.

| Potential claim | Falsifiable experiment | Primary outcomes | Falsification example |
| --- | --- | --- | --- |
| Active candidate gaze improves self-preservation over fixed gaze | Paired exact-state branches on held-out snapshots, identical physical/RNG state and registered gaze cost | capture rate, minimum predator distance, success, safety-efficiency | no improvement, worse capture, or effect disappears after gaze-time cost |
| Looking is selectively useful near occlusion or junctions | Stratified paired evaluation across near-occlusion, junction, peek-location, and control anchors | interaction between source stratum and gaze benefit | equal benefit in controls or no stratum interaction |
| Visual threat memory helps after occlusion | Compare the same vision policy with zero history versus preregistered public-frame history on recently-visible-hidden states | capture, risk calibration, reconfirm latency | history gives no gain or leaks labels rather than public frames |
| VLM risk estimates predict near-horizon danger | Evaluate risk score against held-out H-step capture/minimum-distance outcomes without reusing construction labels | AUROC/AUPRC, Brier score, calibration error | chance discrimination or materially miscalibrated bins |
| The vision-only policy approaches a privileged safe baseline | Paired branches from the same snapshots with policy input restricted to public fields | gap in capture, distance, reward and goal progress | large persistent oracle gap or prompt leakage |
| Active gaze trades efficiency for safety | Compare registered gaze policies at matched starting states and motion budgets | safety, path efficiency, look rate, wall time | safety gain vanishes at matched time/path cost |
| Human gaze anticipates threat rather than merely reacting | Participant-held-out analysis aligned to first predator appearance and occlusion events | time-to-first-look, reaction time, unnecessary-look rate | looks occur only after visible threat or are unrelated to later safety |
| Body-head decoupling is behaviorally meaningful | Compare human/policy episodes with matched threat exposure and route difficulty | decoupled-look rate, capture, path efficiency | decoupling has no outcome association after controls |
| Results generalize beyond sampled states | Hold out seeds, anchor cells, occlusion regions, and eventually worlds | all safety and calibration metrics with confidence intervals | effects confined to construction cells or one seed/world |
| OpenRouter VLM behavior is reproducible enough for comparison | Pin exact model/provider, cache responses, and repeat uncached calls separately | parse success, decision agreement, latency/cost distribution | provider drift or response variance overwhelms policy differences |

## Engineering claims and tests

The following claims may be established by automated tests, but are not
scientific findings:

| Engineering claim | Required test |
| --- | --- |
| Snapshot restore is deterministic | Restore one state twice, apply identical actions, compare public observations, outcomes, physical trace, and RNG-dependent behavior |
| Gaze branches are paired | Hash the stored source state before and after every branch; hashes must match |
| Generation is reproducible | Same normalized config and seed produce the same ordered snapshot IDs in independent output roots |
| Policy input is vision-only | Inspect the constructed request and call log; privileged field names and sentinel values must be absent |
| VLM output is machine-checked | Strict JSON Schema rejects missing, extra, incorrectly typed, out-of-range, and unknown-enum values |
| No-key execution is complete | Remove `OPENROUTER_API_KEY` and run generation, mock open-loop evaluation, and exact-state branch evaluation |
| Results are attributable | Every experiment directory includes resolved config, Git commit, seed, environment/package metadata, raw JSONL, and CSV summary |

## Minimum experimental protocol before paper wording

1. Freeze state-construction and exclusion rules before evaluating policies.
2. Separate generated development snapshots from held-out evaluation snapshots.
3. Group human splits by participant/session; never split frames randomly.
4. Report all state-category counts and every avoidable-by-looking criterion,
   including failures.
5. Compare fixed gaze, each candidate gaze, a vision-only policy, and a
   privileged oracle under the same H-step state/RNG branches.
6. Include gaze duration/action cost and sensitivity to horizon and safety
   thresholds.
7. Report confidence intervals across seeds and anchor regions.
8. Treat P0 outputs only as benchmark validation until those conditions are
   met.

## EXP-00 registered engineering decision rule

`configs/peekbench/exp00.yaml` freezes 75 exact-state snapshots: 60 predator
states and 15 no-predator controls, balanced across the five operational state
categories and three construction sources. Every branch starts from the same
complete state/RNG snapshot. Unlike the P0 instantaneous gaze screen, EXP-00
changes head yaw only through the legal third action dimension.

The primary method table reports safe-success rate, capture rate, unnecessary-
look rate, and path cost for fixed head, random head, coverage scan,
privileged-best legal gaze, and a privileged safe controller. Safe success
means no capture and no registered approach below the configured risk distance
during the horizon. A look is unnecessary when it incurs at least the
registered gaze travel but does not convert a fixed-head safety failure into a
safe result. Path cost is physical prey trajectory length.

The screen returns `GO` only when all registered conditions pass: at least 20%
of predator states fail under fixed head; at least 10% of predator states are
stable legal-gaze recoveries; at least 50% of fixed failures are stable
recoveries; there are at least six such states; and all branches remain legal
and source-immutable. Every snapshot must also match its requested construction
category. “Stable” requires the privileged best gaze to succeed and at least
two distinct non-zero legal target-gaze controllers to succeed. This is a
decision to continue active-gaze research, not verification of a scientific
hypothesis.

## EXP-01 exploratory measurement rule

`configs/peekbench/exp01.yaml` freezes a balanced 15-state pilot. Every static
decision, observation-direction probe, semantic macro candidate, and closed-
loop method starts from the same exact physical/task/RNG state. Policies receive
only current public observation fields and, in the history condition, up to
four prior public binocular frames.

Current threat detection is scored against `predator_pixels_visible`. Danger is
defined by capture or a registered distance-threshold crossing under an
independent 40-step `forward|hold` reference branch. Look direction is scored
only when current predator pixels are absent and at least one legal three-step
look probe reveals them. Macro safety is measured by enumerating all 42
registered motion/look combinations for the same eight-step duration as the
closed-loop decision interval.

Closed-loop public-history capture is paired against fixed continuation, the
same policy's open-loop macro, and its current-only closed loop. The primary gap
is conditioned on states where danger is correctly classified and at least one
safe semantic macro exists; it reports unsafe chosen macros and subsequent
closed-loop captures separately. This pilot is descriptive and exploratory.
No paper claim is allowed from the deterministic mock, the engineering smoke,
or a zero-sized conditional denominator.

## First-person SAC evidence boundary

The binocular SAC run is a training experiment, not evidence that active gaze
caused an improvement. Its registered evaluation therefore uses held-out,
paired seeds and reports the trained active policy beside (1) the same network
with only its head action clamped to zero and (2) random actions. The
head-clamped comparison is an inference-time ablation with distribution shift,
not a separately trained fixed-gaze control. Any causal active-gaze claim still
requires independently trained matched fixed- and active-gaze policies across
multiple seeds. Best, median-ranked, and worst rendered episodes are selected
by a declared task ranking; a visually appealing rollout is never the primary
result.

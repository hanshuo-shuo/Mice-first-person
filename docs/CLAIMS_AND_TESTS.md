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

# EXP-04 — Active Gaze Without Free Compute

## Registered question

Does observation selection improve outcomes when compute and observation budget
are fixed? The comparison includes random gaze, fixed scan, coverage-entropy
maximization, visual saliency, and decision-centric gaze.

## Hard controls

Every method receives one binocular observation per simulator step and makes
one call to the same `public_rgb_threat_features_v1` encoder and one policy
decision. Candidate views are never rendered before selection. Image width,
height, eye count, simulator horizon, time step, and observation duration are
registered in every record. `_assert_equal_budgets` aborts evaluation if any
budget differs.

Different gaze policies can terminate on different steps. For those snapshots,
the evaluator first performs a deterministic preflight from the same exact
state, takes the earliest termination step as a common censoring horizon, and
then restores the source state before rerunning every method to that horizon.
Only the equal-budget reruns enter outcome statistics. Preflight step counts,
the chosen horizon, and the fraction of censored snapshots are saved explicitly;
terminal observations are never padded or fabricated.

All gaze commands use the normal three-dimensional normalized action and remain
subject to the existing +/-60 degree head-yaw contract. Motion uses the same
public-history rule for all methods, so only the gaze-target rule changes.

## Commands and artifacts

```bash
python -m benchmarks.peekbench exp04 --config configs/peekbench/exp04_smoke.yaml
python -m benchmarks.peekbench exp04 --config configs/peekbench/exp04.yaml
```

Outputs are `exp04.jsonl`, `exp04_outcomes.csv`, and `exp04_summary.json` under
`results/peekbench/<experiment_id>/`. Unit tests and smoke outputs establish
contract compliance, not a paper-level performance claim.

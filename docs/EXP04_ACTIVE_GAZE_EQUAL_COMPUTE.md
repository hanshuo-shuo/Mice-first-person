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

## Quest controlled run — 2026-08-22

- Slurm job: `3776261`
- Git commit: `e48682409a43b4609a3dd7424460f61468da8b19`
- Result directory: `exp04_active_gaze_equal_compute_3776261`
- Status: `COMPLETED`, exit code `0:0`, elapsed `00:28:40`
- Snapshots: 200, balanced at 50 per registered visibility category
- Per-method registered budget: 7,292 simulator steps, 7,292 encoder/model
  calls, and 14,584 single-eye frames
- Contract checks: all budgets equal, all gaze actions legal, all source
  snapshots unchanged

| Method | Safe success | Capture | Mean reward |
| --- | ---: | ---: | ---: |
| Random gaze | 0.890 | 0.030 | 0.045 |
| Fixed scan | 0.890 | 0.020 | 0.055 |
| Entropy maximization | 0.895 | 0.020 | 0.065 |
| Visual saliency | 0.820 | 0.035 | 0.050 |
| Decision-centric gaze | 0.885 | 0.015 | 0.070 |

Paired safe-success comparisons against decision-centric gaze found no
advantage over entropy maximization (-1.0 percentage point, exact McNemar
`p=0.625`), fixed scan (-0.5 point, `p=1.0`), or random gaze (-0.5 point,
`p=1.0`). Decision-centric gaze exceeded visual saliency by 6.5 points
(`p=0.0106`; exploratory and not adjusted as a preregistered confirmatory
family). The difficulty was concentrated in `frustum_pixel_occluded`; the
other categories were close to ceiling for most methods.

Common censoring shortened 28/200 snapshots; 15 required an equal-horizon
rerun because methods terminated on different steps, and the minimum common
horizon was one step. Therefore this run verifies the no-free-compute contract
and rejects a simple claim that decision-centric gaze clearly beats the other
coverage baselines, but it does not verify the research hypothesis. A follow-up
should preregister a non-ceiling, fixed-duration state set that remains valid
for every method for the full horizon.

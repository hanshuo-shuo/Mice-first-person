# EXP-05 — Is It Active Gaze or Just Camera Placement?

## Registered comparison

Evaluate the frozen final checkpoint from `sac_cnn_active_gaze_9903898` on the
same 1,000 held-out seeds (`1,000,000`--`1,000,999`) under:

1. learned active gaze;
2. fixed 0 degrees;
3. fixed +60 degrees;
4. fixed -60 degrees;
5. fixed +30 degrees;
6. fixed scan.

The primary comparison is learned active gaze versus fixed +60 degrees. The
reported statistics are the paired clean-success difference with a bootstrap
95% interval, discordant-seed counts, and the exact McNemar p-value.

Fixed placements set the camera yaw before the first policy observation, set
the evaluation environment's head-recenter rate to zero, and replace the third
SAC action component with zero. This holds the requested angle exactly while
leaving the learned forward-velocity and body-yaw outputs, point-mass physics,
collision handling, predator navigation, rewards, and termination unchanged.
The policy receives the resulting public images, proprioception, and applied
previous action.

Fixed scan reuses the EXP-04 public schedule
`[-60, -30, 0, +30, +60, +30, 0, -30]`, with two simulator steps per target.
It uses ordinary normalized head-rate commands and the registered 240 degrees/s
turn-rate limit; it does not teleport the camera.

For a descriptive mapping to the motivating cases, an Active - Fixed +60
success difference of at most 2 percentage points is labeled Case A-like, a
difference of at least 10 points is labeled Case B-like, and values between are
reported as intermediate. The paired interval and discordant outcomes remain
the primary evidence.

## Quest command and artifacts

```bash
bash setup/submit_sac_gaze_ablation_1000.sh
```

The job runs 60 CPU array tasks (six methods by ten seed shards), followed by
an `afterok` aggregation job. Results are written under
`results/sac/sac_cnn_active_gaze_9903898/exp05_gaze_ablation_1000_<ARRAY_JOB_ID>/`.
The aggregate contains per-episode JSONL/CSV, `methods.csv`,
`paired_differences.csv`, `summary.json`, and `REPORT.md`.

## Quest controlled run — 2026-08-23

- Shard array: `3863043`; aggregate: `3863044`
- Git commit: `0a94c50f3f687a8ba6a4bb1413e18be3c1d7e96f`
- Checkpoint SHA-256:
  `7133433da9aceb0d55cb181c1fc42bd2800ec4ba0cbf1e7368c079c6e5a955ec`
- Result directory:
  `results/sac/sac_cnn_active_gaze_9903898/exp05_gaze_ablation_1000_3863043/`
- All 60 rollout shards and the aggregate completed with exit code `0:0`.
- Active and Fixed 0 exactly reproduced the prior audit's 945 and 546 clean
  successes, respectively.

| Method | Clean success (Wilson 95% CI) | Capture | Mean steps | Min distance | Path cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| Learned active gaze | **94.5% [92.9%, 95.8%]** | **5.5%** | **31.52** | **0.258** | **1.284** |
| Fixed 0 degrees | 54.6% [51.5%, 57.7%] | 45.0% | 122.23 | 0.105 | 1.411 |
| Fixed +60 degrees | 90.3% [88.3%, 92.0%] | 9.6% | 44.69 | 0.241 | 1.321 |
| Fixed -60 degrees | 2.6% [1.8%, 3.8%] | 96.9% | 278.13 | 0.020 | 3.275 |
| Fixed +30 degrees | 85.6% [83.3%, 87.6%] | 14.2% | 100.90 | 0.248 | 1.372 |
| Fixed scan | 91.6% [89.7%, 93.2%] | 7.7% | 97.14 | 0.251 | 1.841 |

The primary Active-minus-Fixed-+60 clean-success difference was **+4.2
percentage points**, with paired bootstrap 95% CI `[+2.5, +6.0]`. The paired
table contained 883 both-success, 62 Active-only, 20 Fixed-only, and 35
both-failure seeds; exact McNemar `p=3.71e-6`. Active also used 13.17 fewer
steps per episode, increased minimum predator distance by 0.017, and reduced
path cost by 0.037 relative to Fixed +60; all three paired intervals excluded
zero.

This is an intermediate result rather than the motivating Case A or Case B.
Moving the fixed camera from 0 to +60 recovers 35.7 of the 39.9 percentage
points between Fixed 0 and Active, about 89.5% of that gap. Camera placement is
therefore the dominant explanation. Dynamic control nevertheless contributes
a reproducible residual improvement in success, capture rate, safety margin,
and route efficiency for this frozen checkpoint.

Fixed scan reached 91.6%; Active exceeded it by 2.9 points with paired
bootstrap CI `[+0.6, +5.2]` and exact McNemar `p=0.0149`. Active also used 65.6
fewer steps and 0.557 less path length. This secondary comparison suggests the
remaining value is not indiscriminate scanning. Given the learned policy's
strong sustained +60-degree bias, the next cheapest diagnostic is a
time-scheduled `0 -> +60 -> 0` camera program matched to the learned episode
phase. That would separate closed-loop active sensing from a good pose plus a
simple temporal camera schedule.

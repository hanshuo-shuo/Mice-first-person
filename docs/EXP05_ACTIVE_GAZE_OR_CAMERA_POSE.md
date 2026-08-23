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

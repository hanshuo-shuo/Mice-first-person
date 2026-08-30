# Autoresearch Phase 1 — Frozen-Checkpoint Gaze Schedules

## Status

Phase 1 completed a real bounded development loop and its one-time held-out
confirmation on 2026-08-30.  E0003 passed the registered engineering
confirmation gate.  This confirms the controller improvement for the frozen
checkpoint and task distribution; it does not establish a checkpoint- or
training-independent active-vision claim.

## Frozen run

- Run tag: `gaze_schedule_20260830_v5`
- Foundation commit: `6c4da666d03f67679292f9596df4b864d98283a2`
- Run manifest SHA-256:
  `fb79fb5235dae31c030e1924edee0bb6991fda3c4d2273efce8bef6c89bb2c00`
- EXP-05 checkpoint SHA-256:
  `7133433da9aceb0d55cb181c1fc42bd2800ec4ba0cbf1e7368c079c6e5a955ec`
- Development seeds: `1110000..1110127`, registered before candidate outcomes
- Confirmation seeds: `1200000..1200999`, spent exactly once by `C0001`
- Public inputs: `image_left`, `image_right`, `proprio`, `previous_action`
- Candidate output: one finite normalized `head_yaw_rate`

The trusted evaluator kept SAC locomotion, one model call per executed step,
physics, collision handling, predator navigation, reward callbacks,
termination, and ordered seeds fixed.  Candidate code ran in a killable child
process with per-request timeout and child-only resource limits.

## Development results

| Controller | Clean success | Capture episodes | Mean steps | Mean gaze travel |
| --- | ---: | ---: | ---: | ---: |
| Initial legal scan | 121/128 (94.53%) | 6/128 (4.69%) | 92.22 | 1395.95 deg |
| Pre-positioned fixed +60 reference | 115/128 (89.84%) | 13/128 (10.16%) | 46.65 | 0.00 deg |
| E0002 learned passthrough | 121/128 (94.53%) | 7/128 (5.47%) | 31.57 | 193.47 deg |
| E0003 50/50 learned–scan blend | **125/128 (97.66%)** | **3/128 (2.34%)** | 39.74 | 259.27 deg |

E0002 was discarded: clean success tied the incumbent and capture worsened by
one episode.  Its paired table contained 114 both-success, 7 candidate-only,
7 incumbent-only, and 0 both-failure seeds.

E0003 was kept mechanically.  It improved clean success by 4 paired episodes
(+3.125 percentage points) without worse capture.  Its paired table contained
118 both-success, 7 candidate-only, 3 incumbent-only, and 0 both-failure
seeds.  All cache identity, determinism, source hash, exact coverage, action,
and leak checks passed.  E0003 is commit
`3a7214f0f1fed01baa0db6ba04e504af51b221f4` on branch
`codex/autoresearch-e0003`; only `autoresearch/candidate.py` differs from its
recorded parent incumbent.

These development numbers were used for selection.  They are not confidence
intervals and must not be described as confirmatory evidence.

## One-time confirmation result

| Controller | Clean success | Capture episodes | Mean steps | Mean gaze travel |
| --- | ---: | ---: | ---: | ---: |
| E0003 50/50 learned–scan blend | **977/1000 (97.7%)** | **23/1000 (2.3%)** | 39.63 | 256.38 deg |
| Parent legal scan | 916/1000 (91.6%) | 76/1000 (7.6%) | 97.72 | 1478.51 deg |
| Pre-positioned fixed +60 reference | 895/1000 (89.5%) | 104/1000 (10.4%) | 46.93 | 0.00 deg |

The paired clean-success difference was +6.1 percentage points with a
deterministic bootstrap 95% interval of `[+4.1, +8.1]` points.  Discordant
counts were 82 E0003-only successes and 21 scan-only successes; the exact
McNemar p-value was `1.0696750216702926e-09`.  Capture also improved from 76
to 23 episodes.  The favorable interval, capture non-worsening, authorization,
determinism, identity, complete-record, exact-shard-coverage, source, and leak
gates all passed.

The confirmation records SHA-256 is
`d8dcc5bb4f98498d02bb26b35c6c52d3dafb6238f90fdba1ae9e00a85afb39b0`.
The set is permanently spent; `C0001` is finalized report-only and does not
replace or mutate the E0003 incumbent.

## Operational evidence

The final baseline used Quest array `5201967` and aggregate `5201968`; all 32
shards completed with exit `0:0`.  E0002 used array `5202145` and aggregate
`5202146`; all 16 shards completed.  E0003 used array `5202306` and aggregate
`5202307`; all 16 shards completed.

Earlier run tags preserve four infrastructure failures rather than silently
dropping them: two pre-submission shell-wrapper failures, a compute-node `git`
PATH failure, and an 8 GiB OOM.  No paper outcome was produced from those
runs.  The final scripts use stdin-delivered remote logic, commit-specific
Quest worktrees, no compute-node Git dependency, and 12 GiB rollout shards.
Within v5, E0001 records a local source-symlink path-escape rejection before
smoke; the retry used copied, hash-verified ignored artifacts and became E0002.

Confirmation used original array `5202887`.  Six candidate shard elements
were cancelled by the scheduler on `qnode0476` before producing files; repair
array `5203087` reran only indices 39–44 with that node excluded.  Strict
aggregate `5203088` waited for the original array and successful repair,
verified exact 3 × 1,000 method/seed coverage, and completed with exit `0:0`.

## Acceptance evidence

- Complete pytest suite: `186 passed`
- Autoresearch focused suite after confirmation integration: `115 passed`
- No-key PeekBench `all`: 5 snapshots, 5 open-loop records, 5 branch records
- Two repeated generations produced identical ordered snapshot IDs
- `policy_calls.jsonl`: 175 records; no credential or privileged-field match
- Initial candidate exactly reproduced the historical legal scan on the
  real checkpoint for seed `1100000` across all retained episode metrics

## Claim boundary

The held-out gate supports the engineering statement that the selected legal
blend improves this frozen checkpoint on the registered task distribution.
It does not turn 1,000 evaluation episodes into independent training
replicates.  Broader claims about learned active sensing still require
independently trained checkpoints, with training seed as the inferential unit.

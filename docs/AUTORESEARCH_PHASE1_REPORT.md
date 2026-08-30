# Autoresearch Phase 1 — Frozen-Checkpoint Gaze Schedules

## Status

Phase 1 completed a real bounded development loop on 2026-08-30.  The selected
candidate is an engineering incumbent, not a confirmed scientific result.  The
registered 1,000-seed confirmation set remains unspent and requires a separate
explicit authorization.

## Frozen run

- Run tag: `gaze_schedule_20260830_v5`
- Foundation commit: `6c4da666d03f67679292f9596df4b864d98283a2`
- Run manifest SHA-256:
  `fb79fb5235dae31c030e1924edee0bb6991fda3c4d2273efce8bef6c89bb2c00`
- EXP-05 checkpoint SHA-256:
  `7133433da9aceb0d55cb181c1fc42bd2800ec4ba0cbf1e7368c079c6e5a955ec`
- Development seeds: `1110000..1110127`, registered before candidate outcomes
- Confirmation seeds: `1200000..1200999`, unspent
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

## Acceptance evidence

- Complete pytest suite: `181 passed`
- Autoresearch focused suite after security hardening: `110 passed`
- No-key PeekBench `all`: 5 snapshots, 5 open-loop records, 5 branch records
- Two repeated generations produced identical ordered snapshot IDs
- `policy_calls.jsonl`: 140 records; no credential or privileged-field match
- Initial candidate exactly reproduced the historical legal scan on the
  real checkpoint for seed `1100000` across all retained episode metrics

## Confirmation gate

Do not run confirmation through the normal experiment command.  After explicit
authorization, mark the registered set spent before any rollout, evaluate the
selected E0003 candidate against its parent incumbent and the fixed +60
research reference, then report paired bootstrap intervals, discordant counts,
exact McNemar statistics, and capture non-worsening.  A favorable confirmation
still supports an engineering controller result; broader active-gaze claims
need independent training/checkpoint replication.

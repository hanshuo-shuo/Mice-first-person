# Autoresearch closed-loop handoff

## Mission

Build the smallest safe autonomous experiment loop for this repository.  The
loop must turn one bounded policy change into a reproducible decision:

```text
read incumbent evidence
  -> state one falsifiable hypothesis
  -> change one whitelisted candidate module
  -> run contract checks and a frozen paired evaluator
  -> write durable evidence
  -> keep or discard mechanically
  -> repeat until a finite budget is exhausted
```

The priority is the closed loop itself, not another broad research feature,
dashboard, VLM integration, or large training campaign.  Do not copy the
Karpathy `autoresearch` nanochat sources into this repository.  Reuse its
protocol idea while preserving this repository's simulator, information, and
reproducibility contracts.

Read `AGENTS.md` first.  Its rules override this handoff if the two ever
diverge.

## Handoff snapshot

- Date: 2026-08-30 (Asia/Shanghai).
- Starting branch: `velocity-action-env`.
- Starting commit: `bb75ce1df109a441c090c0653311ee197cb604c8`
  (`Keep simulator tensors off training GPUs`).
- The tracked worktree was clean before this handoff was added.
- The repository already has deterministic PeekBench snapshots, paired state
  restoration, public-observation policy boundaries, matched task manifests,
  SAC training/evaluation entry points, and result provenance.
- The proposed first loop is an engineering search.  It cannot by itself
  verify a scientific active-gaze claim.

### Cleanup completed with this handoff

Only reproducible local clutter was removed: Python/test/Matplotlib caches,
the launcher log, `.DS_Store` files, and old output directories explicitly
named smoke, mock, or smoke acceptance.  Formal/pilot PeekBench results,
speed-sweep results, SAC analysis artifacts, datasets, tracked vendored
bytecode, and the ignored `.env` were preserved.  The deleted generated files
are not recoverable through Git; recreate them with the registered smoke
commands when needed.

## Why the first loop is a frozen-checkpoint gaze controller

`docs/EXP05_ACTIVE_GAZE_OR_CAMERA_POSE.md` reports the frozen checkpoint at
94.5% clean success, fixed +60 degrees at 90.3%, and the registered scan at
91.6% on the historical 1,000-seed evaluation.  Fixed +60 explains most of the
center-to-active gap, while learned gaze retains a smaller residual advantage.
The report already identifies a cheap next diagnostic: a legal temporal camera
program such as `0 -> +60 -> 0`.

That is the right minimum viable autoresearch target because it:

- keeps the SAC checkpoint, locomotion policy, environment, reward, and task
  dynamics fixed;
- has a fast paired rollout evaluator and a clear primary outcome;
- permits a very small mutable surface;
- can exercise resume, crash handling, evidence logging, and keep/discard
  behavior without launching new GPU training;
- directly addresses the current research bottleneck.

Do not start with autonomous SAC hyperparameter or architecture search.  That
becomes phase 2 only after the phase-1 loop passes all acceptance criteria.

## Immediate blocker

The frozen model is not present in this local checkout.  The expected Quest
artifact is:

```text
results/sac/sac_cnn_active_gaze_9903898/
  resolved_config.yaml
  checkpoints/final_model.zip
```

The checkpoint SHA-256 recorded in the EXP-05 report is:

```text
7133433da9aceb0d55cb181c1fc42bd2800ec4ba0cbf1e7368c079c6e5a955ec
```

Implement and test the loop locally with fake policies/small environments, but
do not fabricate a real baseline.  A real rollout must fail clearly when the
checkpoint is absent or its digest differs.  Do not submit a Quest job, copy a
remote artifact, push a branch, or use an external API without explicit user
authorization.

## Required phase-1 layout

Use a small standalone package rather than adding candidate logic to the
trusted EXP-05 evaluator:

```text
autoresearch/
  __init__.py
  __main__.py                 # setup, baseline, experiment, status commands
  candidate.py                # the only agent-editable source after setup
  contract.py                 # public input/output protocol and validation
  evaluator.py                # frozen paired rollout instrument
  guard.py                    # immutable hashes, diff whitelist, leak checks
  ledger.py                   # append-only/atomic experiment records
  program.md                  # instructions consumed by the next agent loop

configs/autoresearch/
  gaze_dev.yaml               # frozen run budget, seed sets, source hashes

tests/
  test_autoresearch_loop.py

results/autoresearch/<run_tag>/
  run.json                    # frozen resolved config and provenance
  incumbent.json              # current accepted candidate identity
  experiments.jsonl           # durable machine-readable ledger
  results.tsv                 # compact human view
  artifacts/E0001/...         # stdout, metrics, checks, diff, failure details
```

Add `results/autoresearch/` to `.gitignore`.  Generated evidence is not source
data and must not be committed.  `program.md`, code, config, and tests are
tracked.

The exact names may change if a repository convention requires it, but keep
the trusted evaluator and mutable candidate visibly separate.

## Candidate contract

The candidate controls only the third normalized action component.  SAC still
produces forward velocity and body yaw from the current public observation.
A suitable interface is:

```python
class CandidateGazeController:
    def reset(self, *, episode_seed: int) -> None: ...

    def head_action(
        self,
        *,
        observation: Mapping[str, np.ndarray],
        public_history: Sequence[Mapping[str, np.ndarray]],
        base_head_action: float,
        step_index: int,
    ) -> float: ...
```

The evaluator must pass defensive copies containing exactly:

- `image_left`
- `image_right`
- `proprio`
- `previous_action`

The result must be one finite scalar in `[-1, 1]`, interpreted as the legal
`head_yaw_rate`.  Head yaw remains constrained to +/-60 degrees by the normal
wrapper.  The candidate must never receive the environment, `info`, reward,
termination reason, snapshot, coordinates, camera labels, future outcome, or
any exact state.

The candidate may use only deterministic local computation and an explicitly
seeded private RNG.  It must not read files, environment variables, network
resources, process state, or provider credentials.  It must not import the
evaluator, environment, simulator, task manifests, or artifact store.  Enforce
this boundary with tests and a narrow import allowlist; do not rely only on
instructions.

## Frozen evaluation contract

Refactor only the minimum reusable mechanics from
`analysis/sac_gaze_ablation.py`.  Preserve its current behavior and tests.
Do not make the candidate a new hard-coded method inside that trusted module.

Every candidate and comparator must use:

- the same frozen checkpoint and resolved config hashes;
- identical ordered episode seeds;
- deterministic SAC inference;
- one binocular observation and one SAC call per executed simulator step;
- the same maximum horizon, normal early termination, physics, reward,
  collision, predator navigation, and goal/capture callbacks;
- legal normalized actions only;
- no candidate-view rendering or privileged lookahead;
- a fresh environment/reset for every paired episode;
- the same public-history policy and maximum history length.

Cache comparator episode records only when the checkpoint, resolved config,
evaluator, seed set, and environment-contract hashes all match.  Content hashes
must be written into `run.json` and every experiment record.

### Seed separation

The historical EXP-05 seeds `1000000..1000999` have already been analyzed and
are not a pristine confirmation set.  Freeze non-overlapping sets in
`gaze_dev.yaml` before the first real candidate is evaluated.  Recommended
initial sizes are:

- smoke: 4 episodes, contract checks only;
- development: 128 paired episodes, reusable for search;
- confirmation: 1,000 paired episodes, explicit one-time gate only.

The automatic loop must have no command that silently runs confirmation.
Require a separate `confirm` command plus explicit user authorization.  Once
confirmation is run, record that the set is spent and never use it for further
selection.  The next agent may choose the exact non-overlapping seed ranges,
but must register them before observing candidate outcomes and explain the
choice in the config.

## Objective and mechanical decision

Keep two baselines distinct:

- **Research reference:** the existing exact fixed +60-degree intervention.
  It is applied before the first observation and answers the camera-placement
  question, but it is not a legal-rate candidate produced by the interface
  above.
- **Search incumbent:** the existing legal-rate fixed scan, re-expressed
  through the candidate interface and starting from the normal zero-degree
  head state.  This is the initial candidate that later candidates must beat.

For each episode, retain candidate, incumbent, and fixed-+60 outcomes on the
same ordered seeds so comparisons remain paired.  Never silently treat the
pre-positioned fixed-+60 reference as if it paid the same gaze-motion cost as a
rate-controlled candidate.

Primary development objective, higher is better:

```text
paired clean-success delta = candidate clean_success - search-incumbent clean_success
```

Hard gates:

1. all contract, hash, determinism, and leak checks pass;
2. no immutable or out-of-whitelist source changed;
3. every action is finite and legal;
4. ordered seed sets and source artifact identities match the run contract;
5. candidate capture-episode rate does not exceed the incumbent rate;
6. development clean success improves by at least two paired episodes;
7. no missing, duplicated, or partially written episode records exist.

For an equal clean-success result, simplicity may be reported but must not
automatically replace the incumbent in phase 1.  Keep the decision rule
mechanical and avoid a hand-tuned composite score.  Record secondary outcomes
without optimizing them: capture, goal reach, steps, minimum predator distance,
path cost, gaze travel, and visible-pixel fraction.

Confirmation is stricter: report paired bootstrap intervals, discordant counts,
and exact McNemar p-values.  A selected candidate is only an engineering win if
the clean-success interval excludes zero in the favorable direction, capture
does not worsen, and every hard contract remains satisfied.  Because the
candidate was selected through repeated development comparisons, do not reuse
development uncertainty as confirmatory evidence.

## Durable state and crash recovery

`experiments.jsonl` is append-only.  Each record needs at least:

```text
experiment_id
parent_incumbent_id
candidate_commit
candidate_sha256
hypothesis
predicted_effect
changed_paths
source_model_sha256
resolved_config_sha256
evaluator_sha256
seed_set_id
started_at / completed_at
status: planned | running | keep | discard | crash | contract_failure
primary_delta and paired counts
secondary metrics
checks
artifact hashes/paths
decision_reason
```

Write artifacts to a temporary experiment directory, fsync/close them, and
rename atomically before appending the final ledger record.  On startup:

- detect a stale `running` record;
- preserve its logs;
- mark it `crash` or resume only an idempotent evaluation;
- never promote a partial candidate;
- reconstruct `incumbent.json` from the last valid `keep` record if needed.

`results.tsv` is a derived view and may be regenerated from JSONL.  It must not
be the source of truth.

## Git isolation and automatic loop

Keep the user's main checkout stable.  Each candidate should be created in a
dedicated experiment worktree from the recorded incumbent commit.

1. Create experiment `E####` and record the falsifiable hypothesis before an
   edit.
2. Create a worktree/branch from the incumbent SHA.
3. Permit changes only to `autoresearch/candidate.py`.
4. Commit the candidate in that worktree.
5. Run static/contract checks, then smoke, then development evaluation.
6. Let the runner emit exactly one machine-readable `keep` or `discard`
   decision.
7. On keep, record that candidate commit as the new incumbent without changing
   the main checkout.
8. On discard/crash, preserve evidence and retire the worktree.
9. Repeat until a finite run budget is reached.
10. A human later reviews and explicitly applies/cherry-picks the final
    incumbent.

Do not use `git reset --hard` on the shared checkout.  Do not automatically
push, merge, delete user branches, install dependencies, submit Slurm jobs, or
call remote model providers.

Default finite budget for the first accepted loop:

```text
max_experiments: 12
max_wall_seconds: 14400
max_consecutive_crashes: 3
on_budget_exhausted: stop_successfully_and_report
```

The upstream `NEVER STOP` behavior is intentionally not adopted.

## CLI acceptance shape

The final CLI may differ internally, but the following workflow must be
possible and documented:

```bash
python -B -m autoresearch setup \
  --config configs/autoresearch/gaze_dev.yaml \
  --run-tag gaze_schedule_YYYYMMDD

python -B -m autoresearch baseline --run-tag gaze_schedule_YYYYMMDD

python -B -m autoresearch experiment \
  --run-tag gaze_schedule_YYYYMMDD \
  --hypothesis-file /path/to/hypothesis.md

python -B -m autoresearch status \
  --run-tag gaze_schedule_YYYYMMDD \
  --json
```

Commands must return nonzero on contract failure and print bounded output.
Full logs belong in the experiment artifact directory.  `status --json` must
be sufficient for an agent to decide the next loop action without scraping a
large log.

## Test-first implementation order

Implement in this order and keep the diff small:

1. Candidate public-input/output contract and test doubles.
2. Immutable hash manifest and changed-path whitelist.
3. Append-only ledger with atomic finalization and crash recovery.
4. Frozen comparator and candidate paired evaluator on tiny fake episodes.
5. Mechanical keep/discard gate.
6. CLI setup/baseline/experiment/status flow.
7. Real frozen-checkpoint adapter, failing safely while the checkpoint is
   absent.
8. `program.md` autonomous protocol and finite-budget loop.
9. Full repository acceptance checks.

Required new tests include:

- candidate receives exactly the four public fields and defensive copies;
- privileged sentinel fields never cross the boundary or enter logs;
- invalid/NaN/out-of-range actions fail before `env.step()`;
- same candidate/seed/source hashes reproduce identical records;
- candidate cannot change seeds, horizon, checkpoint, config, evaluator, or
  immutable source;
- comparator caching invalidates on every relevant hash change;
- pass, discard, crash, timeout, stale-running, and resume paths;
- interrupted writes cannot corrupt the incumbent or ledger;
- confirmation cannot run through the normal automatic experiment command;
- only `autoresearch/candidate.py` may differ in an experiment worktree;
- existing EXP-05 results and tests remain unchanged.

## Repository acceptance

Before handing back any implementation, use the project Conda environment and
disable bytecode writes:

```bash
export PYTHONDONTWRITEBYTECODE=1
export MPLBACKEND=Agg
export PYGAME_HIDE_SUPPORT_PROMPT=1
export SDL_AUDIODRIVER=dummy
export SDL_VIDEODRIVER=dummy

conda run -n Mice-BotEvade pytest -q
conda run -n Mice-BotEvade \
  python -B -m benchmarks.peekbench all \
  --config configs/peekbench/smoke.yaml
```

Also perform the AGENTS.md reproducibility checks: repeated generation must
produce identical ordered snapshot IDs, exact-state branches must replay
deterministically, and generated `policy_calls.jsonl` must contain neither
secrets nor privileged field names.  Do not run large training jobs for
acceptance.

## Phase-1 definition of done

Phase 1 is done only when all of the following are true:

- a clean checkout can initialize a run and record a reproducible baseline;
- one deliberately better fake candidate is kept and one worse candidate is
  discarded without manual ledger editing;
- a crash can be resumed without losing the incumbent or duplicating evidence;
- the only mutable experiment source is the candidate module;
- the real adapter refuses a missing/wrong checkpoint and accepts the recorded
  checkpoint digest when available;
- the finite budget stops the loop without asking for more work or running
  confirmation;
- full tests and the no-key PeekBench smoke pipeline pass;
- the handoff/report explicitly labels all loop findings as engineering
  selection rather than verification of an active-gaze hypothesis.

## Phase 2, explicitly deferred

Only after phase 1 is reviewed may the loop search SAC training choices.  That
phase needs a new candidate hook so the agent cannot modify environment/reward
or evaluation code, fixed transition and hardware budgets, development-only
validation, and repeated training seeds.  The existing matched design in
`configs/sac_matched_generalization.yaml` uses 400,000 transitions and five
training seeds per condition; those five independently trained checkpoints,
not individual evaluation episodes, are the inferential replication unit.

Do not turn phase 2 on merely because phase 1 executes successfully.

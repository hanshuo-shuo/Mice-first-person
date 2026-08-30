# Phase-1 autoresearch protocol: frozen-checkpoint gaze schedules

## Objective

Improve paired clean success over the current legal-rate scan controller while
keeping the trained SAC policy, locomotion commands, simulator, reward,
termination, ordered development seeds, and evaluator frozen.  This is an
engineering search.  It does not verify a scientific active-gaze hypothesis.

## Information boundary

The candidate may use only defensive copies of the current public observation,
up to four prior public observations, the SAC policy's current head command,
its own step index, and an RNG seeded in `reset`.  The only public observation
fields are `image_left`, `image_right`, `proprio`, and `previous_action`.

Never use `info`, rewards, termination reason, simulator objects, exact state,
coordinates, camera labels, future outcomes, files, environment variables,
network access, subprocesses, clocks, or provider calls.  Do not modify the
simulator or evaluator.  `proprio[2]` is the public normalized head yaw and may
be used to implement legal target-seeking schedules.

## One bounded iteration

1. Read `status --json` and the durable artifacts for the incumbent and the
   immediately preceding experiment.
2. State one falsifiable hypothesis and predicted direction before editing.
3. Create a dedicated worktree and branch from the recorded incumbent commit.
4. Change only `autoresearch/candidate.py`; keep the class contract intact.
5. Commit that one-file change.
6. Run `autoresearch experiment` against the shared run directory.  Contract,
   source-hash, leak, smoke, determinism, and development gates precede the
   mechanical keep/discard decision.
7. Read the single final machine record.  A keep becomes the next incumbent;
   a discard, crash, or contract failure does not.
8. Preserve artifacts and retire the worktree without changing the user's
   checkout.

Candidate code executes in a killable spawned worker.  A reset or action call
that exceeds its hard deadline, exits, or exceeds its child-only resource
budget is a contract failure; it must never be retried under another candidate
identity.  Candidate instance state may contain bounded scalars, the physical
head-yaw integrator, and an explicitly seeded private RNG, but not growing
containers or retained observations.

## Quest development evaluation

Quest uses a two-stage ledger lifecycle so a result can never be observed
before its hypothesis is durable:

1. From the foundation checkout, run `baseline --prepare-external`.  This
   records the running baseline and completes the registered four-seed smoke.
2. Push the exact named branch and submit the development comparator shards
   with `setup/submit_autoresearch_gaze.sh baseline`, passing the candidate and
   incumbent commit/SHA identities emitted by the worker context.
3. Copy the self-contained aggregate directory back without editing it, then
   run `baseline --finalize-external ... --experiment-id legal_fixed_scan_v1`.
4. For a candidate, create and commit the one-file experiment worktree first.
   Point the worktree's ignored EXP-05 result path at the registered local
   artifact and invoke the runner with the main checkout's absolute
   `--results-root`.
5. Run `experiment --prepare-external --hypothesis-file ...`; only after it
   returns a running `E####` may the pushed candidate branch be submitted with
   `setup/submit_autoresearch_gaze.sh experiment`.
6. Copy the aggregate directory back and run
   `experiment --finalize-external ... --experiment-id E####`.  The runner
   re-hashes every artifact, validates exact seed/method coverage, and locally
   recomputes the mechanical gate.

After every keep, rebuild the comparator baseline cache under the new
incumbent SHA before evaluating the next candidate.  If Slurm terminates or a
dependency cannot produce a complete aggregate, use the explicit
`abort-external` command to record a crash while preserving staged evidence;
never synthesize or partially aggregate missing shards.

Quest jobs run from commit-specific detached worktrees and share only the
ignored registered source/results directories.  They do not switch the shared
Quest checkout, call a provider, or expose a confirmation option.

Do not tune against the four smoke seeds.  They exist only for contract and
determinism checks.  Do not inspect or run the confirmation set during search.

## Suggested first hypotheses

Prefer small deterministic schedules whose effects are easy to falsify:

- reach +60 degrees from the legal zero start and hold;
- `0 -> +60 -> 0` with one preregistered dwell duration;
- compare short and long +60-degree dwell durations one at a time;
- use public head yaw to avoid oscillation around a target;
- add a single image-derived trigger only after schedule-only candidates have
  established the engineering ceiling.

Do not combine multiple changes in one candidate.  Equal clean success never
replaces the incumbent in phase 1.

## Stop conditions

Stop successfully and report when any configured finite budget is exhausted:
12 experiments, 14,400 wall seconds, or three consecutive crashes.  Also stop
on an immutable hash mismatch, seed/config/checkpoint mismatch, or any sign of
privileged-data leakage.

The normal loop must never run confirmation.  Confirmation requires the
separate command, explicit user authorization, and an unspent-set marker.  A
confirmation result is reportable only with paired bootstrap intervals,
discordant counts, exact McNemar statistics, non-worsened capture, and all
contracts passing.

# EXP-01 VLM Perception--Action Gap

## Status on 2026-08-22

The paired EXP-01 measurement pipeline is implemented and its 15-state pilot
batch is frozen. The complete no-key engineering smoke passed, but the remote
VLM pilot has **not** run. This distinction is deliberate: the registered
pilot config refuses to fall back to the deterministic mock when
`OPENROUTER_API_KEY` is absent.

The external blockers at closeout were:

- `OPENROUTER_API_KEY` was absent from the local process environment;
- `/tmp/quest.sock` was not an active Quest control socket; and
- a non-interactive reconnect to `quest.northwestern.edu` was rejected by SSH
  authentication.

Consequently, this report establishes the benchmark implementation and frozen
pilot inputs, not a result about VLM behavior. In particular, it does not
support the proposed claim that a VLM recognizes static danger but fails under
occlusion, memory, or closed-loop control.

## Research question

Starting from the same exact simulator states, measure the full chain:

1. Does the policy report predator pixels that are actually present?
2. Does its risk score classify registered near-horizon danger?
3. When active observation can reveal a hidden predator, does it choose a
   revealing direction?
4. Does its semantic motion/look recommendation belong to the set of safe
   macro actions available from that state?
5. When decisions are recomputed in a legal closed loop, does the policy reduce
   capture relative to a fixed continuation and its own open-loop action?

The key measurement is not any one accuracy. It is the conditional gap between
correct language-level judgment and unsafe action or capture.

## Frozen exploratory pilot

`configs/peekbench/exp01.yaml` defines 15 constructed snapshots in world
`21_05`, with seed `20260819` and config hash
`7933857233d36dce8e3b5c06a90ced8efcd34a0b32d8b9078831caafa22168a7`.
The batch contains exactly three states from each operational category and five
from each construction source:

| Dimension | Balanced values |
| --- | --- |
| State category | predator visible; geometric outside frustum; frustum/pixel occluded; recently visible then hidden; no-predator control |
| Construction source | near occlusion; junction; mapped peek location |
| Current/history conditions | current binocular observation only; current observation plus up to four prior public binocular frames |

All 15 requested categories were constructed successfully. Every saved state
passed replay determinism. An independent second generation produced the same
ordered snapshot IDs and semantic state hashes.

The exact remote policy route is `openai/gpt-4.1-mini` through the OpenAI
provider, with fallbacks disabled, required-parameter routing enabled, and data
collection denied. The policy prompt declares that `threat_visible` refers to
current eye images, the risk horizon is four seconds, and a selected semantic
macro may run for 0.8 seconds before the next decision.

This is a small exploratory batch. It was not selected by an on-policy
encounter distribution and has no power or confirmatory inference claim.

## Operational definitions

### Static perception and memory

The policy receives only `image_left`, `image_right`, `proprio`,
`previous_action`, and, in the history condition, prior public eye images.
Current threat detection is scored against the canonical
`predator_pixels_visible` camera label. Historical visibility does not make
`threat_visible=true` correct when the current images contain no predator;
memory should instead affect risk and action.

### Danger

Danger is registered independently of the VLM output. From the restored source
state, the evaluator executes the `forward|hold` semantic controller for 40
legal simulator steps. The label is positive if this branch is captured or
crosses the configured minimum predator distance of `0.18`. Risk scores are
thresholded at `0.50` for classification and also retained for Brier scoring.

This is one explicit action-conditional danger construct, not a universal
ground truth for danger.

### Correct observation direction

Six semantic look commands are tested from the same restored state using legal
normalized head-yaw-rate actions: `far_left`, `left`, `center`, `right`,
`far_right`, and `hold`. A state is direction-eligible only when predator pixels
are absent initially and at least one three-step probe reveals them. A VLM look
choice is correct when its paired probe is in the revealing set. States where
no registered probe can reveal the predator are reported but excluded from the
direction denominator.

### Safe macro action

The evaluator enumerates all 42 combinations of seven registered semantic
motion actions and six look actions. Each candidate is executed for eight legal
steps after restoring the exact source state. A candidate is safe when it is
not captured and never crosses the risk-distance threshold. The VLM action is
scored against the complete safe set, and its regret relative to the
future-outcome oracle is retained. The oracle is evaluation-only and is never
sent to the policy.

### Closed loop

Five paired 40-step branches are reported:

| Method | Information and control |
| --- | --- |
| `fixed_continue` | registered `forward|hold` reference |
| `vlm_open_loop_public_history` | one public-history VLM decision held for the full branch |
| `vlm_closed_loop_current_only` | VLM recomputed every eight steps without image history |
| `vlm_closed_loop_public_history` | VLM recomputed every eight steps with up to four prior public frames |
| `privileged_safe_controller` | exact-state evaluator reference, never a policy input |

Every VLM look choice is converted to a target head yaw and reached only through
the normalized third action dimension. Motion still passes through the normal
point-mass integration, collision handling, predator navigation, rewards,
capture/goal events, and termination. The evaluator records capture, safety,
minimum predator distance, goal progress, gaze travel, motion switches, look
switches, conservative actions on controls, latency, cost, cache status, and
parse status.

### Perception--action gap

An actionable state is a registered-danger state with at least one safe
semantic macro candidate. The primary descriptive gap is:

- among actionable states where the public-history danger judgment is correct,
  the fraction whose selected macro is unsafe; and
- among the same states, the fraction captured by the public-history closed
  loop.

Paired capture tables also report captures avoided and introduced relative to
fixed continuation, open loop, and the current-only closed loop. A net rate
alone is not allowed to hide discordant pairs.

## Engineering smoke result

Command:

```bash
env -u OPENROUTER_API_KEY \
  PYTHONDONTWRITEBYTECODE=1 \
  MPLBACKEND=Agg \
  PYGAME_HIDE_SUPPORT_PROMPT=1 \
  SDL_AUDIODRIVER=dummy \
  SDL_VIDEODRIVER=dummy \
  conda run --no-capture-output -n Mice-BotEvade \
  python -B -m benchmarks.peekbench exp01 \
  --config configs/peekbench/exp01_smoke.yaml
```

Artifacts are under
`results/peekbench/exp01_perception_action_gap_smoke_20260822/`.

| Smoke check | Observed |
| --- | ---: |
| Snapshots | 5, one per category |
| Deterministic mock decisions | 20 |
| Remote VLM calls | 0 |
| Parse success | 20 / 20 |
| Current predator-pixel detection | 5 / 5 |
| Eligible look-direction choices | 1 / 3 |
| Safe macro selection when available | 5 / 5 |
| Predator captures in any reported public method | 0 / 4 |
| Source snapshots unchanged | yes |
| Legal action/head-yaw contract | yes |
| Privileged/credential policy-log leak scan | no matches |

The short smoke horizon produced no registered-danger state and no capture.
The gap denominator is therefore zero. The mock's perfect pixel detection is a
known color-threshold property, not VLM evidence; its one-in-three look result
and closed-loop switching are also engineering diagnostics only.

## Verification

- Complete repository suite: **52 passed in 138.93 seconds**.
- Standard no-key PeekBench `all` pipeline: 5 snapshots, 5 open-loop records,
  and 5 paired branch records completed through the mock backend.
- EXP-01 same-config repeat: identical ordered snapshot IDs and state hashes.
- Frozen 15-state pilot repeat: identical ordered snapshot IDs and state
  hashes.
- All smoke probes, macro candidates, and closed-loop branches reported legal
  actions and immutable source snapshots.
- `policy_calls.jsonl` scan found no API-key marker, authorization header,
  predator/prey coordinates, geometric-LOS field, privileged label, source
  state, or exact-state dictionary marker.

No files under `cellworld_game-main/` and no base environment, reward, physics,
or observation/action contract were modified.

## Artifacts

Each completed EXP-01 evaluation writes:

- `exp01.jsonl`: full per-snapshot static labels, look probes, all macro
  candidates, closed-loop traces, and outcomes;
- `exp01_measurements.csv`: flat perception, danger, look, and macro scores;
- `exp01_closed_loop.csv`: one outcome row per snapshot and method;
- `exp01_methods.csv`: method-level descriptive table;
- `exp01_summary.json`: gap, memory, paired-capture, cost, parse, and integrity
  summaries; and
- `policy_calls.jsonl` plus `response_cache/`: sanitized attributable model
  telemetry and cached provider responses.

Generated result directories are not source evidence and remain ignored by
Git. A paper analysis should archive an immutable copy with the code commit,
resolved config, and provider metadata.

## Running the remote pilot

Local execution, after supplying the credential only through the process
environment:

```bash
python -B -m benchmarks.peekbench exp01 \
  --config configs/peekbench/exp01.yaml
```

Quest submission, after the branch is committed and pushed and an authenticated
control socket is active:

```bash
bash setup/submit_quest.sh setup/peekbench_exp01.sbatch
```

`setup/peekbench_exp01.sbatch` exits before evaluation if the Quest job
environment does not contain `OPENROUTER_API_KEY`. The key must never be placed
in the repository, YAML, shell command arguments, Slurm logs, or result files.

After a remote run, acceptance requires re-running the leakage scan, confirming
all parses and source/action contracts, checking reported provider/model and
cost, and then replacing this status section with the exact result directory
and descriptive tables. Until then, EXP-01 is implemented and frozen but the
remote VLM measurement is incomplete.

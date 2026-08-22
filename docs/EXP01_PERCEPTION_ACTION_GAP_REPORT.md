# EXP-01 VLM Perception--Action Gap

## Status on 2026-08-22

The paired pipeline, no-key engineering smoke, and remote exploratory VLM
pilot are complete. Quest Slurm job `3766897` finished with state `COMPLETED`
and exit code `0:0` in 7 minutes 53 seconds on commit
`e28c426a9fab6b881217d3c18d9caf89e8c63211`.

The run used `openai/gpt-4.1-mini` through the pinned OpenAI provider route. It
produced 144 valid decisions: 114 uncached remote calls and 30 cache hits, with
no parse failures and a reported total cost of `$0.034586`. All source-state,
action-legality, and policy-log leakage checks passed.

The principal exploratory result is more basic than the motivating
expectation. This model did **not** first establish strong static perception
and then fail only at action. It detected one of three current visible threats,
classified one of six registered-danger states, chose zero of six revealing
look directions, and assigned exactly the same risk under current-only and
public-history inputs for all 15 states. Public-history closed loop improved on
the model's own open-loop macro, but did not reduce capture relative to the
fixed continuation. These are descriptive results on a small constructed
pilot, not a verified general VLM claim or paper-ready inference.

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

All 15 requested categories were constructed successfully and every saved
state passed replay determinism. A second Quest compute-node generation
(Slurm job `3767276`) produced the same ordered snapshot IDs. Snapshot IDs
cryptographically include the semantic state hash, so this verifies repeat
generation on the reference Quest platform.

A separate macOS generation with the same config and seed differed on 6 of 15
snapshot IDs, although all source cells, requested categories, and construction
success flags matched. The environments differed in platform and numerical
packages: macOS used NumPy `2.4.3`, pygame `2.5.2`, and Torch `2.10.0`; Quest
used NumPy `2.4.6`, pygame `2.6.1`, and Torch `2.5.1`. Inspection localized the
first state divergence to post-settle predator navigation values. The cause is
not proven, but numerical nearest-cell/path selection is a plausible source.
The current reproducibility claim is therefore Quest-within-platform, not
cross-platform. This does not break within-run pairing because every Quest
branch restored the same stored Quest source state, but a confirmatory study
must distribute frozen snapshot artifacts or lock the numerical runtime.

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

## Remote exploratory VLM result

### Static measurement chain

| Measurement | Public-history result |
| --- | ---: |
| Current predator-pixel detection | 12 / 15 correct overall; positive recall 1 / 3 |
| Registered danger classification | 9 / 15 correct overall; positive recall 1 / 6 |
| Danger Brier score | 0.372 |
| Revealing look direction | 0 / 6 eligible states |
| Safe macro available | 15 / 15 states |
| Safe macro selected | 10 / 15 overall; 1 / 6 registered-danger states |
| Exact future-outcome oracle macro selected | 2 / 15 |

Overall accuracy hides the positive-class failure: twelve current images had
no predator pixels and nine states were negative under the registered danger
definition. The relevant recalls are therefore `1/3` for current visible
threats and `1/6` for danger, rather than the superficially larger 80% and 60%
overall accuracies.

The three recently-visible-hidden states received mean risk `0.0` both with
and without public history. Risk scores agreed exactly on all 15 states, and
the complete static decision agreed on 14 of 15. The one changed decision did
not change the recently-hidden capture count: current-only and public-history
closed loops were each captured on 1 of 3 such states. This pilot shows no
useful static threat-memory signal from the supplied history.

For active observation, all six eligible hidden/off-frustum states had at
least one legal revealing probe, but the model selected none of those
directions. Fourteen of 15 static look recommendations were `center`; the only
other recommendation was `left` on a currently visible state. The failure is
therefore observable at direction selection before any head-control execution
issue.

### Paired closed-loop outcome

Capture rates below use only the 12 predator states:

| Method | Captured | Capture rate | Safe-success rate |
| --- | ---: | ---: | ---: |
| Fixed `forward|hold` continuation | 6 / 12 | 50.0% | 50.0% |
| VLM open-loop public-history macro | 9 / 12 | 75.0% | 25.0% |
| VLM closed loop, current only | 7 / 12 | 58.3% | 41.7% |
| VLM closed loop, public history | 6 / 12 | 50.0% | 50.0% |
| Privileged safe reference | 0 / 12 | 0.0% | 100.0% |

The paired comparisons are more informative than the marginal rates:

- relative to fixed continuation, public-history closed loop avoided one
  capture and introduced one capture: net `0/12`;
- relative to the model's open-loop macro, it avoided three and introduced
  none: net `3/12`; and
- relative to current-only closed loop, it avoided one and introduced none:
  net `1/12`.

Thus, feedback helped repair some consequences of holding one VLM macro for
four seconds, but this did not establish a safety improvement over the simple
fixed reference. Five of six registered-danger states still ended in capture
under the public-history closed loop, and it introduced one capture among the
six predator states that were safe under fixed continuation.

Across public-history closed-loop decisions, the mean semantic motion-switch
rate was 20.6% and the mean look-switch rate was 33.9%; on no-predator controls,
33.3% of decision updates were non-forward. These diagnose switching and
conservatism, but no threshold was preregistered for “jitter” or “overly
conservative,” so the pilot does not attach those qualitative labels.

### What happened to the registered gap metric

All six registered-danger states had at least one safe semantic macro, but only
one was correctly classified as dangerous. That one state also received a safe
macro and was not captured by the public-history closed loop. The registered
conditional gap is therefore `0/1`, which is not evidence that the gap is
absent; its denominator collapsed because perception/risk failed first.

The more useful outcome is a failure cascade:

1. danger was recognized in `1/6` actionable danger states;
2. a safe macro was selected in `1/6` danger states; and
3. public-history closed loop avoided capture in `1/6` danger states.

This supports developing EXP-01 as a measurement benchmark, but the present
pilot points to a perception--risk--observation failure rather than isolating a
language-to-action translation gap. A larger, held-out, multi-model study must
retain both the preregistered conditional metric and this stage-wise attrition
analysis.

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

- Complete repository suite: **53 passed in 139.77 seconds**.
- Standard no-key PeekBench `all` pipeline: 5 snapshots, 5 open-loop records,
  and 5 paired branch records completed through the mock backend.
- EXP-01 smoke same-config repeat: identical ordered snapshot IDs and state
  hashes on macOS.
- Frozen 15-state pilot repeat: identical ordered snapshot IDs on two Quest
  compute-node generations; 6 of 15 IDs differed between Quest and macOS as
  documented above.
- All smoke and remote-pilot probes, macro candidates, and closed-loop branches
  reported legal actions and immutable source snapshots.
- All 144 remote outputs parsed under the strict schema. Telemetry records the
  exact model and provider route, 30 cache hits, 114 uncached calls, and cost.
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

The completed remote pilot is stored locally and on Quest under
`results/peekbench/exp01_perception_action_gap_3766897/`. Key artifact SHA-256
digests are:

- `exp01_summary.json`:
  `6002eec5bfcc6ead627adb5ada45259c92b6c2b6ced25e4d23dad691da204e16`;
- `exp01.jsonl`:
  `f359b659e9bbda0e03a613e3f8703298cbf6e37c9ab295ec99890121ec12ebf6`;
- `policy_calls.jsonl`:
  `d9964f08378af93064631a09b2ca974024a42225adc4adcaf769ead4808dede8`.

## Re-running the remote pilot

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

Acceptance requires the same leakage scan, strict parse checks, source/action
contracts, exact provider/model audit, cost accounting, and reference-platform
snapshot comparison used for job `3766897`. New uncached calls are a separate
provider sample and must not silently replace this result; cached replay should
retain and report its cache-hit status.

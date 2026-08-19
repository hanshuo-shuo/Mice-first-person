# P0 PeekBench implementation report

## Status

PeekBench P0 is implemented as a reproducible **engineering MVP** for
BotEvade.  It is not evidence that active gaze, threat memory, or a VLM policy
improves self-preservation.  No large-scale training was run.

Validated implementation commit: `167ab87a4eb9b91c5cfa2a47ca7ce0c95310cd65`.

## Implemented components

- Repository operating rules in `AGENTS.md`.
- Research/information-boundary audit in `docs/RESEARCH_AUDIT.md`.
- Falsifiable claim registry in `docs/CLAIMS_AND_TESTS.md`.
- Versioned PeekBench configs in `configs/peekbench/`.
- Deterministic BotEvade anchor sampling near occlusions, operational
  junctions, and mapped peek locations.
- Complete, type-preserving `get_state_dict()` artifacts plus separate public
  binocular observations and evaluation-only privileged labels.
- Five operational state classes and exact gaze candidates
  `[-60, -30, 0, 30, 60]` degrees.
- Fixed-gaze, candidate-gaze, and privileged-safe H-step branches restored
  from the same source state before every branch.
- Per-criterion `avoidable_by_looking_candidate` screening with explicit
  failure reasons and a permanent P0 disclaimer that the paper definition is
  not established.
- Vision-only policy dataclasses, deterministic mock backend, and OpenRouter
  adapter with strict JSON Schema, exact model/provider config, timeout,
  bounded retry, atomic response cache, hashes, latency, usage, cost, parse
  status, and sanitized raw-response logging.
- Human-demonstration validation, participant/session grouping, descriptive
  look/safety/efficiency metrics, and a successful `no_data` path.
- Unified `generate`, `open-loop`, `branches`, and `all` commands.

No changes were made to `cellworld_game-main`, base physics, collision,
reward, capture/goal events, termination, or the first-person public
observation/action contract.

## Test results

Command:

```bash
env -u OPENROUTER_API_KEY \
  PYTHONDONTWRITEBYTECODE=1 \
  MPLBACKEND=Agg \
  PYGAME_HIDE_SUPPORT_PROMPT=1 \
  SDL_AUDIODRIVER=dummy \
  SDL_VIDEODRIVER=dummy \
  conda run --no-capture-output -n Mice-BotEvade \
  python -m pytest -q
```

Result: **33 passed in 63.88 seconds**.

Coverage includes:

- all 20 pre-existing environment, renderer, recorder, reward, and
  determinism tests;
- state restore replay equality;
- source snapshot immutability under gaze branches;
- identical snapshot IDs/state hashes across independent same-seed/config
  generations;
- all five state-category definitions;
- strict schema rejection of missing, extra, malformed, and out-of-range
  output;
- no-key mock fallback;
- bounded retry, response cache, usage/cost capture, and exact provider config;
- absence of privileged fields and credentials from provider request/log
  surfaces;
- clear no-human-data behavior.

## Generated P0 data

Command:

```bash
env -u OPENROUTER_API_KEY \
  PYTHONDONTWRITEBYTECODE=1 \
  MPLBACKEND=Agg \
  PYGAME_HIDE_SUPPORT_PROMPT=1 \
  SDL_AUDIODRIVER=dummy \
  SDL_VIDEODRIVER=dummy \
  conda run --no-capture-output -n Mice-BotEvade \
  python -B -m benchmarks.peekbench all \
  --config configs/peekbench/smoke.yaml
```

Output: `results/peekbench/peekbench_p0_smoke/`.

Run metadata records seed `23`, config hash
`87a5433289c01237ea20cc70a5b87ee93685c34fa803c55604524929a7167581`,
commit `167ab87a4eb9b91c5cfa2a47ca7ce0c95310cd65`, and `git_dirty=false`.

| Artifact / evaluation | Count |
| --- | ---: |
| Complete snapshots | 5 |
| Public current binocular observations | 5 |
| Public history frames | 1 |
| State categories | 5 (one per required category) |
| Anchor sources | 2 near-occlusion, 2 junction, 1 peek-location |
| Gaze candidates per state | 5 |
| Mock open-loop decisions | 5 |
| Fixed-gaze H-step branches | 5 |
| Candidate-gaze H-step branches | 25 |
| Privileged-safe H-step branches | 5 |
| Total branch trajectories | 35 |
| Candidates passing every avoidable-by-looking filter | 0 |

Every generated snapshot passed construction and replay-determinism checks.
All branch records report the source snapshot unchanged.  A scan of
`open_loop.jsonl` and `policy_calls.jsonl` found no API-key markers,
authorization headers, predator coordinates, geometric-LOS fields, or
privileged-state fields.

## What P0 validates

P0 validates only the following engineering statements:

1. A complete BotEvade state can be serialized, restored, and replayed with
   identical public observations, outcomes, physical state, task state, and
   RNG-dependent behavior.  Semantic hashes normalize only the vendored
   host-wall-clock field `model.last_step`; the full artifact still stores it.
2. Same seed and normalized config produce the same ordered snapshot IDs and
   semantic state hashes across independent runs.
3. Fixed and candidate gaze can be evaluated from paired source states without
   mutating the stored source.
4. All five requested operational classes can be constructed in the smoke
   configuration.
5. The full pipeline runs without an API key through the deterministic mock
   backend.
6. The policy boundary excludes privileged records by construction and by
   automated request/log tests.

## What P0 does not validate

- No state passed all preliminary avoidable-by-looking criteria.  In this
  three-step smoke run, branches generally did not cross the registered
  adverse-risk threshold or show the required minimum-distance/capture
  improvement.  This is a useful filter result, not evidence against active
  looking.
- Active gaze has not been shown to improve capture, survival, reward, route
  efficiency, or calibration.
- Public-frame history has not been shown to improve hidden-threat decisions.
- The privileged-safe heuristic is not proven optimal.
- No real OpenRouter request was made, so remote parse rate, latency, cost,
  provider stability, and model quality are unmeasured.
- No human demonstration sessions were available for behavioral conclusions.
- No cross-world, cross-region, participant-held-out, or large-seed
  generalization result exists.

## Next minimum experiment

1. Freeze a development/evaluation split by seed, anchor cell, and occlusion
   region; generate at least 20 held-out states per class.
2. Register 10- and 20-step horizons, risk thresholds, and a legal head-turn
   duration/cost before viewing policy outcomes.
3. Compare fixed gaze, every candidate gaze, mock vision, and the privileged
   baseline on identical branches; report all filter failures and bootstrap
   confidence intervals by anchor region.
4. Only after that engineering check, run a small cached OpenRouter evaluation
   with one exact model/provider configuration and a hard budget.  Do not train
   a large model in that round.
5. Collect pilot human sessions with explicit participant IDs, then run only
   participant/session-grouped descriptive analysis before proposing an
   inferential human claim.

## Known limitations

- P0 targets `FirstPersonBotEvadeEnv` and world `21_05`; Oasis lacks the full
  task/RNG snapshot contract needed for paired branches.
- States are deliberately constructed near selected anchors and are not an
  on-policy encounter distribution.
- “Junction” currently means an open Cellworld location with at least three
  nearest-neighbor free cells, not a validated behavioral intersection label.
- Instantaneous gaze intervention does not yet charge time or enforce a legal
  sequence of head-turn commands.
- Recent visibility is saved as public history but the constructed history is
  not yet a natural long rollout.
- The mock backend is a deterministic color heuristic, not a scientific model
  baseline.
- The smoke config uses 32x24 images, five states, and a three-step horizon.
- The privileged-safe controller is a transparent heuristic rather than an
  optimal control solution.
- Environment YAML is versioned but is not a platform-specific lockfile.
- Generated result directories are intentionally Git-ignored; source configs,
  code, tests, and this report are the versioned record.

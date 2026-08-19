# Research and reproducibility audit

## Scope

This audit describes the repository before and during the PeekBench P0 work.
It distinguishes existing simulator capability, new engineering validation,
and research questions that remain empirically unverified.

## Existing capabilities

### Environment and embodiment

- `BotEvadeEnv` and `OasisEnv` expose Gymnasium-compatible point-mass tasks.
  The base tasks own physics, collision, predator navigation, rewards,
  capture/goal events, and episode termination.
- `FirstPersonVisionWrapper` renders two mouse-centred RGB eyes from the same
  arena and occlusion polygons used by the simulator.
- The public mouse observation is a dictionary containing two `uint8` HWC
  images, normalized three-value proprioception, and the previous policy
  action.
- Active gaze uses a separate head-yaw state while physical body heading
  remains in `AgentState.body_heading`.
- Camera diagnostics already separate geometric line of sight, left/right
  frustum membership, rendered-pixel visibility, detection range, and believed
  visibility.

### Determinism and data

- `BotEvadeEnv.get_state_dict()` stores model/task state, predator navigation,
  frame stack, counters, event/reward bookkeeping, and Python, NumPy, and
  Gymnasium RNG state.
- PeekBench explicitly seeds the saved global Python and NumPy streams for
  every constructed sample in addition to Gymnasium/model RNG seeding.
- The first-person wrapper adds head yaw, previous action, cached observation,
  and camera visibility to that snapshot.
- Existing tests establish same-seed replay and one-step snapshot round trips.
- Human demonstrations are compressed NPZ files without pickle objects.  They
  record `observation_t -> action_t -> reward_t/done_t`, canonical camera
  labels, events, and a separate privileged-state array.

## Missing or incomplete capability

- There was no versioned benchmark dataset generator, snapshot artifact
  format, state taxonomy, or exact-state multi-branch evaluator.
- There was no policy abstraction that kept provider prompts structurally
  separate from privileged evaluation records.
- There was no strict semantic response schema, retry/cache telemetry, or
  deterministic no-key backend for VLM evaluation.
- Human data had no standalone alignment audit, participant/session split, or
  active-looking/safety-efficiency summaries.
- `OasisEnv` does not currently implement the complete `get_state_dict()` /
  `set_state_dict()` interface.  PeekBench P0 therefore targets BotEvade
  rather than adding an unreviewed partial snapshot implementation to Oasis.
- The Conda specification did not explicitly include `pytest` or
  `jsonschema`, although both are required by the acceptance workflow.
- Pytest's default filename pattern also collected the vendored
  `cellworld_game-main/model_test.py`, an import-time network demo rather than
  a test.  Root `pytest.ini` now limits collection to the reviewed
  `tests/test_*.py` suite without changing vendored code.
- There is no large-scale, held-out empirical evidence for active-gaze,
  memory, calibration, or transfer claims.

## Information boundary

| Information | Vision-only policy | Offline evaluator |
| --- | --- | --- |
| `image_left`, `image_right` | allowed | allowed |
| optional prior public image frames | allowed | allowed |
| normalized `proprio` | allowed | allowed |
| `previous_action` | allowed | allowed |
| camera image hashes / prompt hash | allowed for cache/log metadata | allowed |
| predator pixels inferred from images | allowed | allowed |
| `predator_pixels_visible` ground truth | forbidden | allowed |
| geometric line of sight / frustum ground truth | forbidden | allowed |
| prey or predator world coordinates | forbidden | allowed |
| simulator/task state and RNG state | forbidden | allowed |
| capture, future reward, future minimum distance | forbidden | allowed |
| human `privileged_state` array | forbidden | allowed |

The policy API accepts only the public observation fields and optional public
history.  Benchmark records keep privileged labels beside, not inside, the
policy input.  Model-call logs contain hashes, configuration, timing, usage,
cost, parse status, and raw model response; they do not serialize prompts,
images, API credentials, or privileged records.

## Leakage and reproducibility risks

1. `info` dictionaries and human NPZ files contain both public and privileged
   fields.  Passing a whole `info`, snapshot record, or NPZ row to a policy
   would leak predator position and visibility ground truth.
2. Frame-level random splitting of human demonstrations would put adjacent,
   nearly identical frames from one participant/session into train and test.
   Splits must group by participant and session.
3. Snapshot construction near hand-selected locations is not an on-policy
   state distribution.  Results must be reported by construction source and
   must not be generalized to natural encounter frequency.
4. A state labelled “avoidable-by-looking” by engineering filters is only a
   candidate.  It is not the paper construct until visibility, action
   feasibility, paired outcome, horizon, and policy-independence criteria are
   preregistered and tested on held-out states.
5. Provider model aliases and provider fallback can silently change behavior.
   Model and provider routing configuration must be recorded exactly.
   PeekBench sets an explicit provider order, disables fallbacks, requires
   structured-output parameter support, and requests providers that deny data
   collection; deployments must still review the selected provider's policy.
6. Remote responses are nondeterministic even with cached inputs.  Raw
   response, parse status, hashes, model/provider, and cache hit status must be
   retained.  Reproducible reruns should use the response cache.
7. Package resolution is not a lockfile.  Environment and package versions,
   Git commit, seed, and complete config must accompany every result.
8. The vendored simulator contains tracked historical bytecode.  Running
   Python without `PYTHONDONTWRITEBYTECODE=1` can create irrelevant binary
   diffs.
9. Human metadata currently does not require a participant identifier.  An
   unknown participant group prevents meaningful participant-held-out claims.
10. A raw VLM response can contain arbitrary text.  It is logged as model
    output only and is never executed or used to mutate simulator state.

## Exact-state counterfactual interface

The suitable P0 interface is:

1. Create `FirstPersonBotEvadeEnv` with a fixed, recorded configuration.
2. Reset with a fixed seed and obtain a complete `get_state_dict()` snapshot.
3. Save that snapshot with a type-preserving codec plus the current public
   observation and separate privileged labels.
4. Before every branch, deep-copy and restore the same source snapshot.
5. Change only snapshot-owned first-person gaze state for an instantaneous
   gaze counterfactual, render a public observation without advancing physics,
   then run legal actions through `env.step()` for the registered horizon.
6. Restore again before the next fixed-gaze, candidate-gaze, or privileged
   oracle branch.
7. Compare capture, reward, minimum predator distance, goal progress, and
   termination while retaining every branch's decision and failure criteria.

The full artifact retains `model.last_step`, but semantic replay hashes
normalize that one host wall-clock value.  The vendored model overwrites it on
every step even with `real_time=False`; it does not affect physics, tasks,
observations, RNG, rewards, or termination.  No other state field is excluded.

Instantaneous gaze intervention is an evaluation abstraction, not yet a claim
about biomechanical action cost.  A later benchmark should add legal-duration
head-turn branches and preregister their time cost.  Oasis should enter paired
counterfactual evaluation only after its full task/RNG snapshot contract is
implemented and tested independently.

## Operational PeekBench state classes

- `predator_visible`: predator contributes pixels to at least one current eye.
- `geometric_outside_frustum`: unoccluded 360-degree geometric LOS is true but
  both current eye frusta exclude the predator.
- `frustum_pixel_occluded`: at least one eye frustum contains the projected
  predator but the wall depth test removes all predator pixels.
- `recently_visible_hidden`: current predator pixels are absent and the
  benchmark's recorded public visibility history contains a recent visible
  observation within the configured horizon.
- `no_predator_control`: the environment was constructed with
  `use_predator=False`.

These are deterministic engineering labels.  Their ecological validity and
frequency in natural trajectories remain unverified.

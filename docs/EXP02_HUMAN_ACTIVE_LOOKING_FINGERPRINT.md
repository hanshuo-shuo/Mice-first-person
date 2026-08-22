# EXP-02 Human Active-Looking Fingerprint

## Status on 2026-08-22

The engineering scaffold is implemented as
`analysis.human_active_looking_fingerprint`.  It is an offline descriptive
analysis for recorded Mouse First-Person Lab sessions.  It does not train a
policy, does not call a remote model, and does not require `OPENROUTER_API_KEY`.

Local verification passed with the complete pytest suite: `55 passed` in 149.89
seconds.  The no-key PeekBench smoke pipeline used the deterministic `mock`
backend, produced five snapshot IDs, repeated generation with identical IDs,
and reran paired branches with an unchanged `branches.csv` SHA-256.

Quest Slurm job `3769183` completed successfully with state `COMPLETED`, exit
code `0:0`, and elapsed time `00:00:15` on commit
`c71d6e477d9c0823df7fae3ea629b94addd14bfc`.  The job found no human
demonstration sessions under `datasets/human_demos`, wrote a valid `no_data`
summary, and created the expected artifact files under
`results/human_active_looking_fingerprint/exp02_3769183/`.

With no recorded sessions, the command exits successfully with a `no_data`
summary.  With data, it writes episode, session, participant, split, and risk
bin artifacts under `results/human_active_looking_fingerprint/`.

## Research Question

EXP-02 asks whether human active-looking behavior has a reproducible structure
that is invisible in success-rate-only comparisons.  The target is a
fingerprint of behavior, not a claim that any one proxy fully explains human
strategy.

## Required Split

The split manifest is grouped by:

| Field | Purpose |
| --- | --- |
| `participant_id` | Prevents subject identity from being silently ignored. |
| `session` | Keeps one continuous recording session in one split. |
| `world_name` | Keeps environment layout attached to the split key. |

The `group_key` is `participant/session/world`, and the split unit is recorded
as `participant/session/world; never frame`.  Adjacent frames from the same
video are never randomized into different train/test partitions.

## Fingerprint Metrics

| User-facing question | Implemented metric |
| --- | --- |
| First active observation occurs how far from occlusion? | `first_active_look_distance_to_occlusion` from `prey_x/prey_y` to nearest Cellworld occlusion polygon. |
| Does the participant slow before a dangerous junction? | `pre_danger_deceleration` before a rising risk-context event near an occlusion. |
| Does head turn precede body turn? | `head_turn_before_body_fraction` and `mean_head_lead_seconds` for same-direction onsets. |
| After predator disappearance, how long before checking again? | `reconfirm_action_latency_after_loss` after `predator_pixels_visible` falling edges. |
| Are left/right looks tied to information value? | `look_information_value_agreement`, using predator bearing or asymmetric frustum labels as descriptive value proxies. |
| Does looking change the route? | `route_change_probability_after_look` based on post-look body-heading change. |
| With no predator context, are unnecessary looks reduced? | `no_predator_context_look_rate`, `unnecessary_look_rate`, and `look_suppression_when_no_predator`. |
| As risk rises, how are safety and efficiency traded? | Low/medium/high risk-bin summaries plus `high_risk_forward_suppression`, `high_risk_look_increase`, and `safety_efficiency`. |

Risk context is a descriptive proxy from pixel visibility, geometric LOS,
detection range, and predator distance.  Privileged fields remain analysis-only
and must not be used as policy inputs.

## Outputs

Run locally:

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n Mice-BotEvade \
  python -B -m analysis.human_active_looking_fingerprint \
  --data-root datasets/human_demos
```

Primary artifacts:

| File | Contents |
| --- | --- |
| `exp02_summary.json` | Status, aggregate fingerprint metrics, split counts, and ethics reminder. |
| `episode_fingerprint.csv` | One row per episode with all strategy-structure metrics. |
| `session_summary.csv` | Session-level aggregate grouped by participant/session/world/split. |
| `participant_summary.csv` | Participant-level aggregate for quick inspection. |
| `risk_bin_summary.csv` | Low/medium/high risk frame summaries. |
| `split_manifest.csv` | Hash split by participant/session/world; never frame. |
| `metric_definitions.json` | Machine-readable metric definitions. |
| `ETHICS_NOTE.md` | Human-data governance reminder. |

## Quest

After committing and pushing the intended code/data-path setup, submit:

```bash
bash setup/submit_quest.sh setup/exp02_human_active_looking.sbatch
```

Optional environment overrides:

```bash
EXP02_DATA_ROOT=datasets/human_demos \
EXP02_OUTPUT_DIR=results/human_active_looking_fingerprint/exp02_manual \
bash setup/submit_quest.sh setup/exp02_human_active_looking.sbatch
```

Reference no-data smoke:

| Field | Value |
| --- | --- |
| Slurm job | `3769183` |
| Commit | `c71d6e477d9c0823df7fae3ea629b94addd14bfc` |
| State | `COMPLETED` |
| Exit code | `0:0` |
| Output | `results/human_active_looking_fingerprint/exp02_3769183/` |
| Summary status | `no_data` |

## Ethics And Data Governance

Before using multi-participant data in a paper, follow the relevant
institutional process for informed consent, data management, privacy
protection, and any required ethics or IRB review.  Engineering tests, smoke
runs, generated summaries, and descriptive fingerprints are not evidence that
human-subjects data are approved for publication.

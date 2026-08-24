# Matched Training and Task Generalization

## P0 blocker

Inference ablation of one checkpoint does not estimate training-procedure
stability. The next comparison therefore trains four embodiments from scratch
with five shared training seeds per condition:

| Condition | Policy action | Camera controller |
| --- | --- | --- |
| Active | `[forward_velocity, body_yaw_rate, head_yaw_rate]` | learned |
| Fixed center | `[forward_velocity, body_yaw_rate]` | exact 0 degrees |
| Fixed +60 | `[forward_velocity, body_yaw_rate]` | exact +60 degrees from the first frame |
| Fixed scan | `[forward_velocity, body_yaw_rate]` | registered rate-limited scan |

All conditions share the binocular renderer, proprioception, locomotion
physics, predator navigation, reward, termination, network architecture,
training budget, and exact task banks. Only Active can issue head actions.
Simulator navigation/visibility geometry tensors are forced to CPU at the
environment factory boundary; only the SAC networks use the assigned GPU.
This keeps behavior and memory use consistent across Quest's 40GB and 80GB
A100 nodes.

The registered training seeds are `2026082401`--`2026082405`. Each checkpoint
receives 400,000 training transitions, a budget selected to fit Quest's hard
48-hour `gengpu` walltime using the measured throughput of the earlier 300k
run. Final inference treats the five
training seeds, not the 1,000 tasks within one checkpoint, as the replication
unit. Per-condition reports include the mean, sample standard deviation,
range, and a t interval across the five independently trained checkpoints.

## Versioned task distribution

The old fixed `(0.05, 0.5) -> (1.0, 0.5)` route is not used. Version 1 contains:

| Split | Tasks | Registered holdout |
| --- | ---: | --- |
| Train | 4,096 | five directed region-pair families across NW/NE/SW |
| Validation | 256 | unseen directed pair `northwest -> northeast`, disjoint cell pool |
| Test | 1,000 | every route crosses held-out southeast occlusion cells, disjoint exact pairs |

Every task fixes a start cell, goal cell, predator cell, prey and predator
headings, and predator speed ratio. Predator speed is sampled from
`{0.10, 0.15, 0.20, 0.25}`. Exact start-goal pairs do not overlap across
splits. Train and validation paths never enter southeast; every test path does
and crosses at least one of its 39 registered near-occlusion cells.
The simulator navigation graph must contain an 8--36-cell route, and predator
spawns must remain separated from both endpoints.

Versioned files and SHA-256 identities:

- `train.jsonl`:
  `8d6ce323fd2ba4dc62ed2386aa9dc548b09237a2208e125bb3392994a2d1039e`
- `validation.jsonl`:
  `fb5d82477097f58181b2d07ce16abb88e0c86b379581e9661c4ef033d3e4fb43`
- `test.jsonl`:
  `19c34a7e32ed5a3d2b0d8a2e5c69d50ad57ec16abcf76fcdfcdac9c6dd7efc19`

Training samples the finite train bank with the environment's seeded private
RNG. Validation and test enumerate exact task indices. Privileged task fields
remain in evaluator info/artifacts and never enter policy observations.

## Remaining generalization boundary

This repository currently has the complete cached first-person physics and
snapshot resources only for world `21_05`. The southeast test is therefore an
unseen spatial/occlusion region in the same world, not an unseen occlusion
layout or world. A cross-layout/world claim remains blocked until a second
fully cached world has compatible navigation, collision, visibility, spawn,
and first-person rendering resources. The code does not synthesize an altered
occlusion mask on top of an incompatible navigation graph.

## Commands and artifacts

Regenerate the task banks deterministically:

```bash
python -m training.generate_task_manifests \
  --output-dir configs/tasksets/matched_v1 \
  --train-count 4096 \
  --validation-count 256 \
  --test-count 1000
```

Run one local gradient/checkpoint smoke:

```bash
python train_first_person_sac.py \
  --config configs/sac_matched_generalization.yaml \
  --condition fixed_p60 \
  --smoke
```

Submit the full Quest dependency chain:

```bash
bash setup/submit_matched_training.sh
```

Before the full matrix, the four conditions can be exercised sequentially on
one 40GB A100 (`pcie` constraint) with four environment workers and one
held-out test task each:

```bash
bash setup/submit_quest.sh setup/matched_sac_acceptance.sbatch
```

To queue the full chain behind that acceptance job:

```bash
MATCHED_ACCEPTANCE_JOB_ID=<ACCEPTANCE_JOB_ID> \
  bash setup/submit_matched_training.sh
```

The chain contains 20 A100 training tasks, 100 CPU test shards after all
training succeeds, and one aggregate job. The aggregate writes per-checkpoint,
per-condition, and paired training-seed tables under
`results/sac_matched/matched_v1_matrix_<TRAIN_ARRAY_ID>/test_<TEST_ARRAY_ID>/aggregate/`.

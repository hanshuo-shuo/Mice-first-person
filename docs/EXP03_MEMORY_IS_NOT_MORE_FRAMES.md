# EXP-03 — Memory Is Not More Frames

## Registered question

When the complete current public observation is identical, can a method use a
threat sighting outside a short frame stack to choose a different action? This
tests persistence of an occluded threat, not current-frame object detection.

## Pair construction

Each pair starts from the same prey position, body heading, head yaw, and public
sensor state. In the threat condition a stationary predator is initially
visible; in the control condition no predator exists. Both environments then
receive the same legal normalized body-turn actions. A pair is kept only when:

1. `image_left`, `image_right`, `proprio`, and `previous_action` are byte-identical;
2. the threat was visibly encoded earlier in the public history;
3. it is absent from the last `frame_stack_k` public frames;
4. it is not visible at the endpoint; and
5. threat and control target actions differ.

The predator candidate must be aligned with the goal at history start. After a
legal 180-degree turn, `forward` moves away from the persistent rear threat,
whereas `backward` moves toward the goal in the no-predator control. This makes
the differing target actions a consequence of task geometry, not an arbitrary
history label.

Privileged threat presence is available only to the evaluator and the
explicitly named privileged reference.

## Methods

- `single_frame_reactive`: current public RGB only;
- `frame_stacking`: current plus the last K frames;
- `gru_belief`: fixed-gate recurrent diagnostic over registered history;
- `transformer_history`: fixed full-history attention diagnostic;
- `privileged_belief`: exact-state upper reference;
- `vlm_textual_memory`: public visual detections compressed to a registered
  text state (`threat_last_seen_left/right` or `no_threat_seen`).

The GRU, transformer, and textual-memory entries are deterministic diagnostic
probes, not trained-model results. A paper comparison must replace or augment
them with preregistered trained models on disjoint train/validation/test pairs.

## Commands and artifacts

```bash
python -m benchmarks.peekbench exp03 --config configs/peekbench/exp03_smoke.yaml
python -m benchmarks.peekbench exp03 --config configs/peekbench/exp03.yaml
```

Outputs are `exp03.jsonl`, `exp03_predictions.csv`, and `exp03_summary.json`
under `results/peekbench/<experiment_id>/`. Smoke results are engineering
evidence only and never verify the research hypothesis.

## Quest controlled run — 2026-08-22

- Slurm job: `3773769`
- Git commit: `15209e0fa69d9fdc14c12fd9c06ce3f68f3be606`
- Result directory: `exp03_memory_is_not_more_frames_3773769`
- Status: `COMPLETED`, exit code `0:0`, elapsed `00:03:01`
- Pairs: 100 unique exact-state pairs from 100 near-occlusion cells
- Contract checks: all current public observations byte-identical; all paired
  target actions different; all threat/control state-hash pairs unique

| Method | Condition accuracy | Both conditions correct |
| --- | ---: | ---: |
| Single-frame reactive | 0.50 | 0.00 |
| Four-frame stack | 0.50 | 0.00 |
| Fixed-gate GRU belief probe | 1.00 | 1.00 |
| Full-history attention probe | 1.00 | 1.00 |
| Privileged belief reference | 1.00 | 1.00 |
| Registered textual-memory probe | 1.00 | 1.00 |

This controlled construction demonstrates that current-frame detection and a
short stack are insufficient for these pairs, while registered persistent
state is sufficient. It does **not** establish learned GRU, transformer, or VLM
generalization: those entries remain deterministic diagnostic probes, and all
pairs come from the near-occlusion anchor set.

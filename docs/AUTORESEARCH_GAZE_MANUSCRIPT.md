# Bounded Autoresearch for Legal Active-Gaze Control in Predator Evasion

## Draft status

This is a development-stage manuscript outline.  Bracketed confirmation text
must remain unresolved until the one-time held-out experiment is explicitly
authorized and completed.  The current evidence supports an engineering
selection result only.

## Abstract

Autonomous research loops can make rapid progress while also amplifying
evaluation leakage, irreproducible selection, and simulator shortcuts.  We
implemented a bounded autoresearch protocol for first-person predator evasion
that permits changes to one legal-rate gaze controller while freezing a
trained binocular SAC policy, locomotion outputs, task dynamics, rewards,
termination, ordered seeds, and evaluator.  Candidate code receives only four
public observation fields and executes in a time- and memory-bounded process;
all experiment state is append-only and content-addressed.  On a registered
128-seed development set, the initial legal scan achieved 121 clean successes
with 6 capture episodes.  A learned-gaze passthrough tied success and was
discarded for one additional capture.  A preregistered 50/50 blend of learned
and scan head-rate commands achieved 125 clean successes with 3 capture
episodes and was kept mechanically.  [Held-out 1,000-seed confirmation is
unspent; no confirmatory effect is claimed.]  The result demonstrates a
falsifiable autonomous engineering loop and identifies a candidate for
confirmation, rather than verifying a general active-vision hypothesis.

## 1. Introduction

Active vision couples perception and action: where an agent looks changes the
observations that determine its later motion.  This makes camera-control
experiments especially vulnerable to hidden state access, unequal compute,
and interventions that bypass physical rate limits.  It also makes them a
useful test bed for disciplined autonomous experimentation.

We ask an engineering question: with a trained locomotion policy and simulator
frozen, can a bounded loop improve a legal head-rate controller using only
public first-person observations?  The contribution is the combination of a
narrow mutable surface, paired registered evaluation, mechanical decisions,
crash-safe evidence, and an explicit separation between development selection
and one-time confirmation.

## 2. Methods

### 2.1 Environment and public contract

The task uses the Cellworld-based first-person predator-evasion environment.
The physical body heading remains canonical; head yaw is separate and limited
to ±60 degrees.  Actions are normalized forward velocity, body yaw rate, and
head yaw rate.  Candidates receive defensive copies of exactly `image_left`,
`image_right`, `proprio`, and `previous_action`, plus at most four public
history frames and the frozen SAC head command.  Rewards, event labels,
coordinates, exact state, and simulator objects never cross the boundary.

### 2.2 Frozen policy and comparators

All runs use the final EXP-05 checkpoint with SHA-256
`7133433da9aceb0d55cb181c1fc42bd2800ec4ba0cbf1e7368c079c6e5a955ec`.
SAC controls locomotion on every step.  The search incumbent is the historical
legal symmetric scan starting from zero head yaw.  A camera pre-positioned at
+60 degrees is retained as a research reference but is not treated as a
rate-controlled candidate.

### 2.3 Bounded autoresearch protocol

The loop records a falsifiable hypothesis before editing, creates a candidate
commit from the current incumbent, permits only `autoresearch/candidate.py` to
change, runs contract and deterministic smoke checks, evaluates 128 paired
development seeds, and emits one keep/discard decision.  A keep requires all
hard checks, at least two additional clean-success episodes, and no increase
in capture episodes.  Equal success never replaces the incumbent.

Candidate code runs in a spawned process with a hard per-call deadline and
child-only CPU/memory limits.  Static checks prohibit file, environment,
network, process, clock, unseeded randomness, retained observation containers,
and privileged imports.  Quest shards repeat every episode and aggregate only
complete, ordered, identity-matched records.  The local runner independently
recomputes the gate from copied aggregate records.

### 2.4 Registered data splits

The four-seed smoke set begins at `1100000`; the reusable development set is
`1110000..1110127`; the one-time confirmation set is
`1200000..1200999`.  All are disjoint from the historical EXP-05 range
`1000000..1000999` and were frozen before candidate outcomes.

## 3. Development results

The legal scan attained 121/128 clean successes and 6/128 capture episodes.
The learned SAC head command alone also attained 121/128 clean successes but
had 7 captures, so it was discarded.  It nevertheless exposed complementary
paired behavior: seven candidate-only and seven incumbent-only successes.

Motivated by that complementarity, the next hypothesis changed one quantity:
the head rate became an equal-weight average of the learned command and legal
scan command.  The blend attained 125/128 clean successes and 3/128 captures.
Relative to scan, it produced seven candidate-only versus three
incumbent-only successes, for a +4/128 paired development improvement.  It
also reduced mean episode length from 92.22 to 39.74 steps and gaze travel from
1395.95 to 259.27 degrees; these were secondary, non-optimized outcomes.

## 4. Confirmation analysis plan

[Unresolved.]  Spend the registered confirmation set once.  Report the paired
clean-success mean difference and deterministic bootstrap 95% interval,
candidate-only and incumbent-only counts, exact McNemar p-value, capture
non-worsening, and all contract checks.  The engineering gate passes only if
the interval excludes zero favorably and capture does not worsen.

## 5. Limitations

The development set was reused for selection, so its effect size is optimistic
and not inferential evidence.  The experiment freezes one trained checkpoint;
evaluation episodes are not independent training replicates.  The fixed +60
reference receives a pre-positioned camera and therefore does not pay the same
motion cost as a legal-rate controller.  The selected blend is deliberately
simple and may be specific to this renderer, policy, speed regime, and task
distribution.  Scientific claims about learned active sensing require the
held-out gate and, ultimately, independently trained checkpoints.

## 6. Reproducibility and artifacts

The frozen run is `gaze_schedule_20260830_v5`.  The selected candidate is
commit `3a7214f0f1fed01baa0db6ba04e504af51b221f4`; its source SHA-256 is
`706f7512b41e02064a174c50c08e597281997da73e822f94e4cdb3d613bc731d`.
The append-only ledger, candidate sources, hypotheses, smoke results,
development records, summaries, checks, and artifact hashes are stored below
`results/autoresearch/gaze_schedule_20260830_v5/` and are intentionally not
version-controlled source data.

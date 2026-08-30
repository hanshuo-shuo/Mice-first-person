"""Phase-1 search incumbent: the registered legal-rate EXP-05 scan.

This is the only source file an autoresearch experiment may edit.  The
controller deliberately depends only on the public contract passed to
``head_action``.  It integrates the actually applied float32 head command from
``previous_action`` rather than reading an environment object or privileged
state.
"""


class CandidateGazeController:
    """Track the historical fixed-scan targets with legal rate commands."""

    _TARGETS_DEGREES = (-60.0, -30.0, 0.0, 30.0, 60.0, 30.0, 0.0, -30.0)
    _DWELL_STEPS = 2
    _HEAD_YAW_LIMIT_DEGREES = 60.0
    _MAXIMUM_HEAD_DELTA_DEGREES = 24.0
    _RECENTER_DELTA_DEGREES = 9.0
    _TOLERANCE_DEGREES = 2.0
    _MINIMUM_ACTIVE_COMMAND = 0.051

    def reset(self, *, episode_seed):
        # The incumbent is deterministic and needs no RNG.  Retaining the
        # explicit seed makes episode initialization visible and auditable.
        self._episode_seed = int(episode_seed)
        self._head_yaw_degrees = 0.0

    def head_action(
        self,
        *,
        observation,
        public_history,
        base_head_action,
        step_index,
    ):
        del public_history, base_head_action
        target_index = (int(step_index) // self._DWELL_STEPS) % len(
            self._TARGETS_DEGREES,
        )
        target = self._TARGETS_DEGREES[target_index]
        # At step n>0 this is the exact float32 command consumed by the
        # wrapper at step n-1.  Replaying the public dynamics avoids the tiny
        # float32 round-trip error that would come from proprio[2].
        if int(step_index) > 0:
            previous_command = float(observation["previous_action"][2])
            if abs(previous_command) > 0.05:
                self._head_yaw_degrees += (
                    previous_command * self._MAXIMUM_HEAD_DELTA_DEGREES
                )
            elif abs(self._head_yaw_degrees) <= self._RECENTER_DELTA_DEGREES:
                self._head_yaw_degrees = 0.0
            elif self._head_yaw_degrees > 0.0:
                self._head_yaw_degrees -= self._RECENTER_DELTA_DEGREES
            else:
                self._head_yaw_degrees += self._RECENTER_DELTA_DEGREES
            self._head_yaw_degrees = max(
                -self._HEAD_YAW_LIMIT_DEGREES,
                min(self._HEAD_YAW_LIMIT_DEGREES, self._head_yaw_degrees),
            )

        error = target - self._head_yaw_degrees

        if abs(target) <= self._TOLERANCE_DEGREES and abs(
            error,
        ) <= self._TOLERANCE_DEGREES:
            return 0.0

        command = error / self._MAXIMUM_HEAD_DELTA_DEGREES
        if abs(error) <= self._TOLERANCE_DEGREES:
            direction = error if abs(error) > 0.1 else target
            command = (
                self._MINIMUM_ACTIVE_COMMAND
                if direction >= 0.0
                else -self._MINIMUM_ACTIVE_COMMAND
            )
        elif abs(command) <= 0.05:
            command = (
                self._MINIMUM_ACTIVE_COMMAND
                if error >= 0.0
                else -self._MINIMUM_ACTIVE_COMMAND
            )
        return max(-1.0, min(1.0, command))

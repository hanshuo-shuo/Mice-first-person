"""E0003: blend learned active gaze with the legal scan incumbent."""


class CandidateGazeController:
    """Average learned and scan head-rate commands with equal weight."""

    _TARGETS_DEGREES = (-60.0, -30.0, 0.0, 30.0, 60.0, 30.0, 0.0, -30.0)
    _DWELL_STEPS = 2
    _HEAD_YAW_LIMIT_DEGREES = 60.0
    _MAXIMUM_HEAD_DELTA_DEGREES = 24.0
    _RECENTER_DELTA_DEGREES = 9.0
    _TOLERANCE_DEGREES = 2.0
    _MINIMUM_ACTIVE_COMMAND = 0.051
    _LEARNED_WEIGHT = 0.5

    def reset(self, *, episode_seed):
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
        del public_history
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

        target_index = (int(step_index) // self._DWELL_STEPS) % len(
            self._TARGETS_DEGREES,
        )
        target = self._TARGETS_DEGREES[target_index]
        error = target - self._head_yaw_degrees
        if abs(target) <= self._TOLERANCE_DEGREES and abs(
            error,
        ) <= self._TOLERANCE_DEGREES:
            scan_command = 0.0
        else:
            scan_command = error / self._MAXIMUM_HEAD_DELTA_DEGREES
            if abs(error) <= self._TOLERANCE_DEGREES:
                direction = error if abs(error) > 0.1 else target
                scan_command = (
                    self._MINIMUM_ACTIVE_COMMAND
                    if direction >= 0.0
                    else -self._MINIMUM_ACTIVE_COMMAND
                )
            elif abs(scan_command) <= 0.05:
                scan_command = (
                    self._MINIMUM_ACTIVE_COMMAND
                    if error >= 0.0
                    else -self._MINIMUM_ACTIVE_COMMAND
                )
            scan_command = max(-1.0, min(1.0, scan_command))

        command = (
            self._LEARNED_WEIGHT * float(base_head_action)
            + (1.0 - self._LEARNED_WEIGHT) * scan_command
        )
        return max(-1.0, min(1.0, command))

"""E0001: pass through the frozen SAC policy's learned head command."""


class CandidateGazeController:
    """Use image-conditioned active gaze while leaving locomotion untouched."""

    def reset(self, *, episode_seed):
        self._episode_seed = int(episode_seed)

    def head_action(
        self,
        *,
        observation,
        public_history,
        base_head_action,
        step_index,
    ):
        del observation, public_history, step_index
        return float(base_head_action)

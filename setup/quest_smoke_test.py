"""Small compute-node smoke test for the Quest Python environment."""

import gymnasium
import pygame
import pulsekit
import shapely
import stable_baselines3
import torch
import yaml

from botevade_gym import BotEvadeEnv


def main() -> None:
    env = BotEvadeEnv(
        world_name="21_05",
        use_lppos=False,
        use_predator=False,
        render=False,
        real_time=False,
        action_type=BotEvadeEnv.ActionType.CONTINUOUS,
    )
    observation, _ = env.reset(seed=0)
    assert env.observation_space.contains(observation)
    env.close()

    print(
        "Quest compute-node smoke test passed "
        f"(torch={torch.__version__}, device=cpu)."
    )


if __name__ == "__main__":
    main()

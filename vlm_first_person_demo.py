"""Minimal binocular-observation/action-out loop for a VLM/VLA policy.

Examples:
    conda run -n Mice-BotEvade python vlm_first_person_demo.py --save /tmp/prey_view.png
    conda run -n Mice-BotEvade python vlm_first_person_demo.py --human
    conda run -n Mice-BotEvade python vlm_first_person_demo.py --env oasis --human
"""

import argparse

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=("botevade", "oasis"), default="botevade")
    parser.add_argument("--world", default="21_05")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--width", type=int, default=192, help="Width of each eye image")
    parser.add_argument("--height", type=int, default=128, help="Height of each eye image")
    parser.add_argument("--fov", type=float, default=120.0, help="Horizontal FOV per eye")
    parser.add_argument(
        "--active-gaze",
        action="store_true",
        help="Use [forward velocity, body yaw rate, head yaw rate] actions",
    )
    parser.add_argument(
        "--single-rgb",
        action="store_true",
        help="Return one centred RGB image instead of the binocular VLA dict",
    )
    parser.add_argument("--human", action="store_true")
    parser.add_argument("--save", help="Optional path for the final RGB frame")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def make_env(args):
    vision_kwargs = {
        "vision_width": args.width,
        "vision_height": args.height,
        "vision_fov": args.fov,
        "observation_mode": "single_rgb" if args.single_rgb else "mouse",
        "action_mode": (
            "egocentric_velocity_head"
            if args.active_gaze
            else "egocentric_velocity"
        ),
        "render_mode": "human" if args.human else "rgb_array",
    }
    if args.env == "botevade":
        from botevade_gym import BotEvadeEnv, FirstPersonBotEvadeEnv

        return FirstPersonBotEvadeEnv(
            world_name=args.world,
            use_lppos=False,
            use_predator=True,
            action_type=BotEvadeEnv.ActionType.CONTINUOUS,
            **vision_kwargs,
        )
    from oasis_gym import FirstPersonOasisEnv, OasisEnv

    return FirstPersonOasisEnv(
        world_name=args.world,
        use_predator=True,
        action_type=OasisEnv.ActionType.CONTINUOUS,
        **vision_kwargs,
    )


def vlm_policy(first_person_obs, action_space, rng: np.random.Generator) -> np.ndarray:
    """Replace this body with the model call.

    In mouse mode, pass ``image_left``, ``image_right``, ``proprio``, and
    ``previous_action`` to a VLA.  A plain VLM can receive the two images and
    have its symbolic decision converted to this same numeric action schema.
    """

    if isinstance(first_person_obs, dict):
        assert first_person_obs["image_left"].shape[-1] == 3
        assert first_person_obs["image_right"].shape[-1] == 3
    else:
        assert first_person_obs.ndim == 3 and first_person_obs.shape[-1] == 3
    return rng.uniform(action_space.low, action_space.high).astype(np.float32)


def save_rgb(path: str, frame: np.ndarray) -> None:
    import pygame

    surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
    pygame.image.save(surface, path)


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    env = make_env(args)
    try:
        obs, info = env.reset(seed=args.seed)
        if not env.observation_space.contains(obs):
            raise RuntimeError("First-person reset observation does not match observation_space")
        if isinstance(obs, dict):
            observation_description = {
                key: value.shape for key, value in obs.items()
            }
        else:
            observation_description = obs.shape
        print(
            f"first-person obs={observation_description}; "
            f"action_space={env.action_space}",
        )
        for _ in range(args.steps):
            action = vlm_policy(obs, env.action_space, rng)
            obs, reward, terminated, truncated, info = env.step(action)
            if not env.observation_space.contains(obs):
                raise RuntimeError("First-person step observation does not match observation_space")
            if terminated or truncated:
                obs, info = env.reset()
        if args.save:
            frame = env.render_first_person("rgb_array")
            save_rgb(args.save, frame)
            print(f"saved {args.save}")
    finally:
        env.close()


if __name__ == "__main__":
    main()

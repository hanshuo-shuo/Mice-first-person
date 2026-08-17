"""Evaluate a random policy on the Oasis task.

Usage:
    python eval_oasis.py [--episodes 20] [--render] [--predator-ratio 0.15]
"""
import argparse
import numpy as np
from oasis_gym import OasisEnv


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--render", default=True, action="store_true")
    p.add_argument("--predator-ratio", type=float, default=0.3,
                   help="Predator forward speed as a ratio of prey max forward speed")
    p.add_argument("--turning-ratio", type=float, default=0.175,
                   help="Predator turning speed as a ratio of prey max turning speed")
    p.add_argument("--max-step", type=int, default=600)
    p.add_argument("--no-predator", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    env = OasisEnv(
        world_name="oasis_island7_02",
        use_predator=not args.no_predator,
        predator_prey_forward_speed_ratio=args.predator_ratio,
        predator_prey_turning_speed_ratio=args.turning_ratio,
        max_step=args.max_step,
        render=args.render,
        real_time=args.render,  # only pace to real-time when rendering
    )

    results = []
    for ep in range(args.episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        capture_events = 0
        goal_events = 0
        camera_visible_steps = 0
        geometric_visible_steps = 0
        minimum_distance = np.inf
        done = truncated = False

        while not (done or truncated):
            action = rng.uniform(-1.0, 1.0, size=(2,)).astype(np.float32)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward

            events = info["transition_events"]
            capture_events += int(events["capture_event"])
            goal_events += int(events["goal_event"])
            # These are intentionally different labels.  The base Oasis
            # environment has no first-person renderer, so its pixel label is
            # false; use a first-person environment when measuring camera
            # visibility rather than falling back to simulator LOS.
            camera_visible_steps += int(events["predator_pixels_visible"])
            geometric_visible_steps += int(events["predator_geometric_los"])
            minimum_distance = min(minimum_distance, float(events["minimum_distance"]))

        episode_metrics = info["episode_metrics"]
        survived = int(bool(episode_metrics["survived"]))
        result = {
            "reward": ep_reward,
            "capture_events": capture_events,
            "goal_events": goal_events,
            "camera_visible_steps": camera_visible_steps,
            "geometric_visible_steps": geometric_visible_steps,
            "minimum_distance": minimum_distance,
            "survived": survived,
        }
        results.append(result)
        print(
            f"  ep {ep+1:3d}  reward={ep_reward:7.2f} "
            f"capture_events={capture_events} goal_events={goal_events} "
            f"survived={survived}"
        )

    rewards   = [r["reward"]   for r in results]
    captures  = [r["capture_events"] for r in results]
    goals     = [r["goal_events"] for r in results]
    survivals = [r["survived"] for r in results]
    min_distances = [r["minimum_distance"] for r in results]
    print("\n--- Summary ---")
    print(f"  episodes      : {args.episodes}")
    print(f"  predator ratio: {args.predator_ratio}")
    print(f"  reward   mean={np.mean(rewards):.2f}  std={np.std(rewards):.2f}")
    print(f"  captures mean={np.mean(captures):.2f}  std={np.std(captures):.2f}")
    print(f"  goals    mean={np.mean(goals):.2f}  std={np.std(goals):.2f}")
    print(f"  min dist mean={np.nanmean(min_distances):.4f}")
    print(f"  survival rate: {np.mean(survivals)*100:.1f}%")

    env.close()


if __name__ == "__main__":
    main()

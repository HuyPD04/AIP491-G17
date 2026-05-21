from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.visdrone import build_processed_index
from src.rl.action import action_count, enabled_action_values
from src.rl.env import AdaptiveSlicingEnv
from src.rl.policy import DQNPolicy
from src.rl.replay_buffer import ReplayBuffer
from src.rl.state import state_dim
from src.rl.trainer import epsilon_by_step, linear_decay, optimize_dqn, select_action
from src.utils.config import load_config
from src.utils.setting import CHECKPOINT_DIR, PROCESSED_DATA_DIR, RL_CONFIG


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(int(cfg.get("seed", 42)))

    dataset = build_processed_index(PROCESSED_DATA_DIR, split=args.split)
    if bool(cfg.get("train_only_hard", True)):
        dataset = [sample for sample in dataset if Path(str(sample["hard_region_path"])).exists()]
    sample_limit = cfg.get("sample_limit")
    if sample_limit is not None:
        dataset = dataset[: int(sample_limit)]
    if not dataset:
        raise RuntimeError("No training samples found. Run detect_full_image.py and hard_region.py first.")

    import torch

    device = torch.device(str(cfg.get("device", "cpu")))
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

    env = AdaptiveSlicingEnv(cfg, dataset)
    policy = DQNPolicy(state_dim(cfg), action_count(), int(cfg.get("hidden_dim", 128))).to(device)
    target_policy = DQNPolicy(state_dim(cfg), action_count(), int(cfg.get("hidden_dim", 128))).to(device)
    target_policy.load_state_dict(policy.state_dict())
    if args.resume:
        state_dict = torch.load(args.resume, map_location=device)
        policy.load_state_dict(state_dict)
        target_policy.load_state_dict(policy.state_dict())

    optimizer = torch.optim.Adam(policy.parameters(), lr=float(cfg.get("lr", cfg.get("learning_rate", 0.001))))
    replay = ReplayBuffer(int(cfg.get("replay_buffer_size", 10000)))
    batch_size = int(cfg.get("batch_size", 32))
    target_update_interval = int(cfg.get("target_update_interval", 100))
    save_interval = int(cfg.get("save_interval", max(len(dataset), 1)))
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log_path)

    total_episodes = cfg.get("num_episodes")
    if total_episodes is None:
        total_episodes = int(cfg.get("epochs", 1) or 1) * len(dataset)
    total_episodes = int(total_episodes)
    global_step = 0
    last_epsilon = float(cfg.get("epsilon_start", 1.0))
    progress = tqdm(range(total_episodes), desc="RL training")
    for episode in progress:
        if episode % len(dataset) == 0:
            random.shuffle(dataset)
        sample = dataset[episode % len(dataset)]
        state = env.reset(str(sample["image_id"]))
        done = False
        episode_reward = 0.0
        losses: list[float] = []
        while not done:
            if "epsilon_decay_steps" in cfg:
                epsilon = linear_decay(
                    float(cfg.get("epsilon_start", 1.0)),
                    float(cfg.get("epsilon_end", 0.1)),
                    global_step,
                    int(cfg.get("epsilon_decay_steps", 1)),
                )
            else:
                epsilon = epsilon_by_step(
                    float(cfg.get("epsilon_start", 1.0)),
                    float(cfg.get("epsilon_end", 0.1)),
                    global_step,
                    float(cfg.get("epsilon_decay", 0.999)),
                )
            guided_prob = linear_decay(
                float(cfg.get("guided_prob_start", cfg.get("guided_prob", 0.8))),
                float(cfg.get("guided_prob_end", cfg.get("guided_prob", 0.8))),
                global_step,
                int(cfg.get("guided_decay_steps", 1)),
            )
            last_epsilon = epsilon
            action = select_action(state, policy, epsilon, guided_prob, env)
            next_state, reward, done, _ = env.step(action)
            replay.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            global_step += 1

            if len(replay) >= batch_size:
                batch = replay.sample(batch_size)
                losses.append(
                    optimize_dqn(
                        batch,
                        policy,
                        target_policy,
                        optimizer,
                        float(cfg.get("gamma", 0.9)),
                        allowed_actions=enabled_action_values(cfg),
                    )
                )

        if episode % target_update_interval == 0:
            target_policy.load_state_dict(policy.state_dict())
        if episode > 0 and episode % save_interval == 0:
            torch.save(policy.state_dict(), checkpoint_dir / f"episode_{episode}.pt")

        mean_loss = sum(losses) / max(len(losses), 1)
        append_csv_row(
            log_path,
            {
                "episode": episode,
                "image_id": sample["image_id"],
                "reward": episode_reward,
                "steps": env.step_idx,
                "num_slices": len(env.visited_rois),
                "covered_hard_objects": len(env.covered_object_ids),
                "epsilon": epsilon,
                "guided_prob": guided_prob,
                "loss": mean_loss,
            },
        )
        progress.set_postfix(reward=f"{episode_reward:.3f}", eps=f"{epsilon:.3f}", loss=f"{mean_loss:.4f}")

    final_path = checkpoint_dir / "final.pt"
    torch.save(policy.state_dict(), final_path)
    print(f"trained_episodes={total_episodes}")
    print(f"checkpoint={final_path}")
    print(f"log={log_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the backup RL adaptive slicer.")
    parser.add_argument("--config", type=Path, default=RL_CONFIG)
    parser.add_argument("--split", default="train")
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument("--log-path", type=Path, default=ROOT / "runs" / "logs" / "train_rl.csv")
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def append_csv_row(path: str | Path, row: dict[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    main()

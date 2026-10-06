""" Speed of the Python and Rust game engines: raw env steps/sec, then real training steps/sec
with train.py's settings. The model is loaded from a file but never saved.

    python -m tools.bench_engine                  # both, env and training
    python -m tools.bench_engine --only env
"""
import argparse
import time

import numpy as np

from cribbage.env import CribbageEnv
from cribbage.paths import MODEL_PATH
from cribbage.rust_env import RustCribbageEnv

ENGINES = {"python": CribbageEnv, "rust": RustCribbageEnv}


def bench_env(cls, seconds):
    """ One env, random legal actions vs the smart opponent. """
    env = cls("smart")
    rng = np.random.default_rng(0)
    env.reset(seed=0)
    steps, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        for _ in range(1000):
            _, _, done, trunc, _ = env.step(int(rng.choice(np.flatnonzero(env.action_masks()))))
            steps += 1
            if done or trunc:
                env.reset()
    return steps / (time.perf_counter() - t0)


def bench_train(cls, model_path, steps):
    import torch
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from sb3_contrib import MaskablePPO
    import train

    class Timer(BaseCallback):
        """ Splits wall time into playing games (rollouts) and updating the network. """
        def __init__(self):
            super().__init__()
            self.rollout = self.update = 0.0
            self.mark = None

        def _on_rollout_start(self):
            if self.mark is not None:
                self.update += time.perf_counter() - self.mark
            self.mark = time.perf_counter()

        def _on_rollout_end(self):
            self.rollout += time.perf_counter() - self.mark
            self.mark = time.perf_counter()

        def _on_training_end(self):
            self.update += time.perf_counter() - self.mark

        def _on_step(self):
            return True

    def make_env():
        from stable_baselines3.common.monitor import Monitor
        from sb3_contrib.common.wrappers import ActionMasker
        return ActionMasker(Monitor(cls("smart")), lambda e: e.unwrapped.action_masks())

    torch.set_num_threads(train.TORCH_THREADS)
    env = SubprocVecEnv([make_env for _ in range(train.N_ENVS)])
    model = MaskablePPO.load(model_path, env=env, device="cpu")
    model.batch_size = train.BATCH_SIZE
    timer = Timer()
    t0 = time.perf_counter()
    model.learn(total_timesteps=steps, callback=timer)
    elapsed = time.perf_counter() - t0
    env.close()
    cycles = model.num_timesteps and steps / (model.n_steps * train.N_ENVS)
    return steps / elapsed, timer.rollout / cycles, timer.update / cycles


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["env", "train"])
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--env-seconds", type=float, default=20)
    parser.add_argument("--train-steps", type=int, default=14 * 1024 * 6)
    args = parser.parse_args()

    if args.only != "train":
        for name, cls in ENGINES.items():
            print(f"env   {name:6}: {bench_env(cls, args.env_seconds):8,.0f} steps/s (1 env, 1 core)", flush=True)
    if args.only != "env":
        for name, cls in ENGINES.items():
            sps, rollout, update = bench_train(cls, args.model, args.train_steps)
            print(f"train {name:6}: {sps:8,.0f} steps/s | per cycle: {rollout:5.1f}s playing, "
                  f"{update:5.1f}s updating", flush=True)

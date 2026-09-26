import os
import time
import json
import argparse
import torch
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from cribbage import CribbageEnv
from cribbage.paths import MODEL_PATH, STATE_PATH, LOG_DIR
from evaluate import evaluate, EVAL_GAMES

N_ENVS = 14
TRAIN_OPPONENT = "smart"
CHUNK_STEPS = 200_000            # train this many steps, then check the clock
NET_ARCH = [256, 256]            # separate policy and value nets of this shape


def make_env(opponent):
    def _init():
        env = Monitor(CribbageEnv(opponent=opponent))
        env = ActionMasker(env, lambda e: e.unwrapped.action_masks())
        return env
    return _init


def load_state():
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            return json.load(f)
    return {"total_steps": 0, "sessions": 0, "history": []}


def save_state(state):
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=float, default=1.0, help="How long to train, in hours")
    args = parser.parse_args()

    torch.set_num_threads(1)
    os.makedirs(LOG_DIR, exist_ok=True)

    state = load_state()
    state["sessions"] += 1

    env = SubprocVecEnv([make_env(TRAIN_OPPONENT) for _ in range(N_ENVS)])

    if os.path.exists(MODEL_PATH + ".zip"):
        print(f"Resuming from {MODEL_PATH}.zip (lifetime steps so far: {state['total_steps']:,})")
        model = MaskablePPO.load(MODEL_PATH, env=env, device="cpu")
    else:
        print("No existing model found — starting fresh.")
        model = MaskablePPO("MlpPolicy", env, verbose=0, device="cpu",
                             n_steps=1024, batch_size=256, n_epochs=8,
                             gamma=0.999, learning_rate=0.0003,
                             policy_kwargs=dict(net_arch=NET_ARCH))

    model.set_logger(configure(LOG_DIR, ["stdout", "csv"]))

    deadline = time.time() + args.hours * 3600
    print(f"Training for {args.hours} hours (until {time.strftime('%H:%M:%S', time.localtime(deadline))})...")

    session_steps = 0
    t0 = time.time()
    while time.time() < deadline:
        model.learn(total_timesteps=CHUNK_STEPS, reset_num_timesteps=False)
        session_steps += CHUNK_STEPS
        model.save(MODEL_PATH)   # save after every chunk — safe to Ctrl+C any time
        remaining = deadline - time.time()
        print(f"  ...checkpoint saved | session steps so far: {session_steps:,} | "
              f"time left: {remaining / 60:.0f} min")

    elapsed = time.time() - t0
    state["total_steps"] += session_steps
    print(f"\nSession done: {session_steps:,} steps in {elapsed / 60:.1f} min "
          f"({session_steps / elapsed:,.0f} steps/sec)")
    print(f"Lifetime total steps: {state['total_steps']:,}")

    print(f"\n========== EVAL vs SMART ({EVAL_GAMES} games) ==========")
    winrate = evaluate(model, "smart")
    print(f"Win rate: {winrate:.1f}%")

    state["history"].append({
        "session": state["sessions"],
        "session_steps": session_steps,
        "lifetime_steps": state["total_steps"],
        "winrate_vs_smart": winrate,
    })
    save_state(state)
    env.close()

    print(f"\nModel saved to {MODEL_PATH}.zip -- run this script again anytime to keep training it.")
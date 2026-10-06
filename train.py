import glob
import os
import time
import json
import argparse
import sys
import torch
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from cribbage import CribbageEnv, ENGINE
from cribbage.paths import MODEL_PATH, MODELS_DIR, STATE_PATH, LOG_DIR, POOL_DIR, BASELINE_NET
from evaluate import evaluate, EVAL_GAMES
from tools.export_opponent import export
from tools.session_note import write_note
from tools.vault_charts import write_charts
import metrics

N_ENVS = 14
CHUNK_STEPS = 200_000            # train this many steps, then check the clock
NET_ARCH = [256, 256]            # separate policy and value nets of this shape
TORCH_THREADS = 8                # for the network update; measured fastest with 14 envs on 16 cores
BATCH_SIZE = 1024                # minibatch per gradient step (was 256; fewer, steadier updates)
POOL_KEEP = 8                    # self-play: the newest snapshots of itself in the draw (files are kept)


def make_env(opponent, pool=(), p_smart=0.5):
    def _init():
        env = Monitor(CribbageEnv(opponent=opponent, pool=pool, p_smart=p_smart))
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


def pool_files():
    """ Opponents for self-play: every fixed network in models/pool (b54.npz, ...) plus the newest
    POOL_KEEP snapshots of itself (sp_<lifetime steps>.npz). """
    snaps = sorted(glob.glob(os.path.join(POOL_DIR, "sp_*.npz")))[-POOL_KEEP:]
    fixed = sorted(p for p in glob.glob(os.path.join(POOL_DIR, "*.npz")) if not os.path.basename(p).startswith("sp_"))
    return fixed + snaps


def add_snapshot(model):
    return export(model, os.path.join(POOL_DIR, f"sp_{model.num_timesteps:013d}.npz"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=float, default=1.0, help="How long to train, in hours")
    parser.add_argument("--opponent", default="smart", choices=["smart", "pool"],
                        help="pool = self-play: the heuristic (--p-smart of games) or a network from models/pool")
    parser.add_argument("--p-smart", type=float, default=0.5, help="share of pool games against the heuristic")
    parser.add_argument("--pool-every", type=int, default=20_000_000,
                        help="add a snapshot of itself to the pool every this many steps")
    parser.add_argument("--metrics-games", type=int, default=2000,
                        help="games for the end-of-session metrics and vault note (0 = skip both)")
    args = parser.parse_args()

    torch.set_num_threads(TORCH_THREADS)
    os.makedirs(LOG_DIR, exist_ok=True)

    state = load_state()
    state["sessions"] += 1

    pool = []
    if args.opponent == "pool":
        os.makedirs(POOL_DIR, exist_ok=True)
        if os.path.exists(MODEL_PATH + ".zip"):
            add_snapshot(MaskablePPO.load(MODEL_PATH, device="cpu"))
        pool = pool_files()
        print(f"Self-play: {args.p_smart:.0%} of games vs the heuristic, the rest vs "
              f"{', '.join(os.path.basename(p) for p in pool)}")
    env = SubprocVecEnv([make_env(args.opponent, pool, args.p_smart) for _ in range(N_ENVS)])

    if os.path.exists(MODEL_PATH + ".zip"):
        print(f"Resuming from {MODEL_PATH}.zip (lifetime steps so far: {state['total_steps']:,})")
        model = MaskablePPO.load(MODEL_PATH, env=env, device="cpu")
        model.batch_size = BATCH_SIZE       # a loaded model keeps the batch size it was saved with
    else:
        print("No existing model found — starting fresh.")
        model = MaskablePPO("MlpPolicy", env, verbose=0, device="cpu",
                             n_steps=1024, batch_size=BATCH_SIZE, n_epochs=8,
                             gamma=0.999, learning_rate=0.0003,
                             policy_kwargs=dict(net_arch=NET_ARCH))

    model.set_logger(configure(LOG_DIR, ["stdout", "csv"]))

    deadline = time.time() + args.hours * 3600
    print(f"Game engine: {ENGINE}")
    print(f"Training for {args.hours} hours (until {time.strftime('%H:%M:%S', time.localtime(deadline))})...")

    session_steps = 0
    t0 = time.time()
    try:
        while time.time() < deadline:
            model.learn(total_timesteps=CHUNK_STEPS, reset_num_timesteps=False)
            session_steps += CHUNK_STEPS
            model.save(MODEL_PATH)   # save after every chunk — safe to Ctrl+C any time
            if pool and session_steps % args.pool_every == 0:
                add_snapshot(model)
                pool = pool_files()
                env.env_method("set_pool", pool)
                print(f"  ...pool now {len(pool)} networks (added {os.path.basename(pool[-1])})")
            remaining = deadline - time.time()
            print(f"  ...checkpoint saved | session steps so far: {session_steps:,} | "
                  f"time left: {remaining / 60:.0f} min")
    except KeyboardInterrupt:
        # The chunk in progress is dropped; everything up to the last checkpoint is kept and recorded.
        print(f"\nStopped early. Keeping the last checkpoint ({session_steps:,} session steps).")
        model = MaskablePPO.load(MODEL_PATH, device="cpu") if session_steps else model

    elapsed = time.time() - t0
    state["total_steps"] += session_steps
    print(f"\nSession done: {session_steps:,} steps in {elapsed / 60:.1f} min "
          f"({session_steps / elapsed:,.0f} steps/sec)")
    print(f"Lifetime total steps: {state['total_steps']:,}")

    print(f"\n========== EVAL vs SMART ({EVAL_GAMES} games) ==========")
    winrate = evaluate(model, "smart")
    print(f"Win rate: {winrate:.1f}%")
    entry = {
        "session": state["sessions"],
        "session_steps": session_steps,
        "lifetime_steps": state["total_steps"],
        "opponent": args.opponent if args.opponent == "smart" else f"pool (p_smart {args.p_smart})",
        "winrate_vs_smart": winrate,
        "date": time.strftime("%Y-%m-%d"),
        "engine": ENGINE,
        "hours": round(elapsed / 3600, 2),
        "steps_per_sec": round(session_steps / elapsed),
        "obs": int(model.observation_space.shape[0]),
        "command": f"CRIBBAGE_ENGINE={ENGINE} python " + " ".join(sys.argv),
    }
    if os.path.exists(BASELINE_NET):
        entry["winrate_vs_baseline"] = evaluate(model, vs=BASELINE_NET)
        print(f"Win rate vs the frozen 54% model ({os.path.basename(BASELINE_NET)}): {entry['winrate_vs_baseline']:.1f}%")

    state["history"].append(entry)
    save_state(state)
    try:
        env.close()
    except (OSError, EOFError):              # the workers may already have gone after a Ctrl+C
        pass

    if args.metrics_games:
        # The session is already recorded above; a failure here only loses the metrics and the note.
        try:
            n = state["sessions"]
            print(f"\n========== METRICS ({args.metrics_games} games) ==========")
            m = metrics.measure(MODEL_PATH, args.metrics_games, N_ENVS, vs_random=False,
                                out=os.path.join(MODELS_DIR, f"metrics_s{n}.json"))
            print(f"Discards: {m['vs_smart']['discard_ai']['avg_points_lost']:.2f} pts lost/deal "
                  f"(bot {m['vs_smart']['discard_opp']['avg_points_lost']:.2f})")
            prev = next((h for h in reversed(state["history"][:-1]) if h.get("winrate_vs_smart") is not None), None)
            note = write_note(entry, m, prev)
            print(f"Vault note: {note}" if note else "No Obsidian vault found; no note written.")
            if note:
                write_charts()
        except Exception as e:
            print(f"Metrics or vault note failed ({e!r}); the session itself is saved.")

    print(f"\nModel saved to {MODEL_PATH}.zip -- run this script again anytime to keep training it.")
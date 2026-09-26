import os
import sys
import time
import pandas as pd
import plotext as plt
from cribbage.paths import CSV_PATH
from cribbage.ui import clear_screen

X = "time/total_timesteps"


def load():
    if not os.path.exists(CSV_PATH):
        print(f"No CSV at {CSV_PATH}\nStart training first (python train.py).")
        return None
    try:
        df = pd.read_csv(CSV_PATH, on_bad_lines="skip")
    except Exception as e:
        print(f"Could not read CSV yet ({e}). Try again in a few seconds.")
        return None
    if X not in df.columns or len(df) < 2:
        print("Not enough data yet. Give it a minute.")
        return None
    return df


def chart(df, col, title, color, smooth=1):
    if col not in df.columns:
        print(f"[{title}] column '{col}' not in CSV yet")
        return
    d = df.dropna(subset=[X, col])
    if len(d) < 2:
        print(f"[{title}] waiting for data...")
        return
    y = d[col]
    if smooth > 1:
        y = y.rolling(smooth, min_periods=1).mean()
    plt.clf()
    plt.plotsize(90, 14)
    plt.plot(d[X].tolist(), y.tolist(), color=color)
    plt.title(title)
    plt.show()


def last(df, col):
    if col not in df.columns:
        return None
    s = df[col].dropna()
    return s.iloc[-1] if len(s) else None


def snapshot():
    df = load()
    if df is None:
        return

    # ep_rew_mean is now the sum of scaled per-point rewards plus the
    # +-1.0 win/loss bonus across a WHOLE game (many deals), not one deal.
    chart(df, "rollout/ep_rew_mean", "Avg reward per full game (higher = better)", "yellow", smooth=5)
    chart(df, "train/entropy_loss", "Negative entropy (rises toward 0 as the AI settles)", "cyan")
    chart(df, "rollout/ep_len_mean", "Avg decisions per full game", "green", smooth=5)

    steps = last(df, X)
    fps = last(df, "time/fps")
    rew = last(df, "rollout/ep_rew_mean")
    elen = last(df, "rollout/ep_len_mean")
    ev = last(df, "train/explained_variance")

    print("=" * 60)
    if steps is not None:
        print(f" STEPS:            {int(steps):,}")
    if fps is not None:
        print(f" SPEED:            {int(fps):,} steps/sec")
    if elen is not None and steps is not None:
        # rough games/min estimate using the current avg decisions/game
        if elen > 0 and fps:
            print(f" EST GAMES/MIN:    {int((fps * 60) / elen):,}  (based on ~{elen:.0f} decisions/game)")
        print(f" AVG DECISIONS/GAME: {elen:.1f}")
    if rew is not None:
        print(f" AVG REWARD/GAME:  {rew:.3f}  (win bonus is +-1.0, points are scaled x0.05)")
    if ev is not None:
        print(f" EXPLAINED VAR:    {ev:.2f}  (closer to 1 = the AI predicts outcomes better)")
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "watch":
        while True:
            clear_screen()
            snapshot()
            print("\n(refreshing every 10s, Ctrl+C to stop)")
            time.sleep(10)
    else:
        snapshot()
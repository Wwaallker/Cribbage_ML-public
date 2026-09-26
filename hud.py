import os
import sys
import time
import pandas as pd
from cribbage.paths import CSV_PATH
from cribbage.ui import BANNER, RED, GREEN, YELLOW, CYAN, DIM, BOLD, RESET, bar, clear_screen
from play import play_game, ReturnToMenu

X = "time/total_timesteps"


# ======================================================================
# MAIN MENU
# ======================================================================
def main_menu():
    while True:
        clear_screen()
        print(BANNER)
        print(f" {CYAN}{BOLD}MAIN MENU{RESET}")
        print(f" {'-' * 40}")
        print(f"  {BOLD}[1]{RESET} Play vs AI")
        print(f"  {BOLD}[2]{RESET} Training Dashboard (live)")
        print(f"  {BOLD}[Q]{RESET} Quit")
        print()
        choice = input(f" {CYAN}Select: {RESET}").strip().lower()
        if choice == "1":
            try:
                play_game()
            except ReturnToMenu:
                pass
        elif choice == "2":
            training_dashboard()
        elif choice == "q":
            clear_screen()
            sys.exit(0)


# ======================================================================
# TRAINING DASHBOARD
# ======================================================================
def sparkline(series, width=50):
    if len(series) < 2:
        return ""
    vals = series.tail(width * 4).tolist()
    step = max(1, len(vals) // width)
    vals = vals[::step][:width]
    lo, hi = min(vals), max(vals)
    blocks = " ▁▂▃▄▅▆▇█"
    if hi == lo:
        return blocks[4] * len(vals)
    out = ""
    for v in vals:
        idx = int((v - lo) / (hi - lo) * (len(blocks) - 1))
        out += blocks[idx]
    return out


def load_csv():
    if not os.path.exists(CSV_PATH):
        return None
    try:
        df = pd.read_csv(CSV_PATH, on_bad_lines="skip")
    except Exception:
        return None
    if X not in df.columns or len(df) < 2:
        return None
    return df


def last(df, col):
    if col not in df.columns:
        return None
    s = df[col].dropna()
    return s.iloc[-1] if len(s) else None


def render_dashboard():
    df = load_csv()
    print(BANNER)

    if df is None:
        print(f"{YELLOW}  >> NO SIGNAL -- waiting for models/logs/progress.csv{RESET}")
        print(f"{DIM}  (start training with train.py in another terminal){RESET}\n")
        return

    steps, fps = last(df, X), last(df, "time/fps")
    rew, elen = last(df, "rollout/ep_rew_mean"), last(df, "rollout/ep_len_mean")
    ev, ent, kl = last(df, "train/explained_variance"), last(df, "train/entropy_loss"), last(df, "train/approx_kl")

    print(f" {CYAN}{BOLD}TRAINING STATUS{RESET}                              {DIM}(refreshing...){RESET}")
    print(f" {'-' * 68}")
    if steps is not None:
        print(f" {BOLD}STEPS{RESET}     {steps:>14,.0f}   {DIM}lifetime timesteps trained{RESET}")
    if fps is not None:
        print(f" {BOLD}SPEED{RESET}     {fps:>14,.0f}   {DIM}steps/sec{RESET}")
    print()

    if rew is not None:
        c = GREEN if rew > -0.3 else (YELLOW if rew > -1.0 else RED)
        print(f" {BOLD}REWARD{RESET}    {bar(rew, -3.0, 0.5, color=c)}  {c}{rew:+.3f}{RESET}")
    if ent is not None:
        print(f" {BOLD}ENTROPY{RESET}   {bar(ent, -1.1, 0.0, color=CYAN)}  {CYAN}{ent:+.3f}{RESET}")
    if ev is not None:
        print(f" {BOLD}EV     {RESET}   {bar(ev, 0.0, 1.0, color=YELLOW)}  {YELLOW}{ev:.3f}{RESET}")
    print()

    if elen is not None:
        print(f" {BOLD}GAME LEN{RESET}  {elen:>6.1f} decisions/game")
    if kl is not None:
        print(f" {BOLD}KL DIV{RESET}    {kl:>8.5f}   {DIM}(policy still adjusting if > ~0.01){RESET}")
    print()

    if "rollout/ep_rew_mean" in df.columns:
        print(f" {BOLD}REWARD TREND{RESET}")
        print(f" {GREEN}{sparkline(df['rollout/ep_rew_mean'].dropna())}{RESET}")
    if "train/entropy_loss" in df.columns:
        print(f" {BOLD}ENTROPY TREND{RESET}")
        print(f" {CYAN}{sparkline(df['train/entropy_loss'].dropna())}{RESET}")
    print()
    print(f"{DIM} {'-' * 68}{RESET}")
    print(f"{DIM} press ctrl+c to return to menu{RESET}")


def training_dashboard():
    try:
        while True:
            clear_screen()
            render_dashboard()
            time.sleep(5)
    except KeyboardInterrupt:
        return


# ======================================================================
if __name__ == "__main__":
    main_menu()
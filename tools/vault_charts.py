""" Draws the training charts into the Obsidian vault (Charts/*.png) and the note that shows
them (Charts.md). The data is the vault's own notes: the properties of Sessions/ and
Checkpoints/, so a value corrected by hand in a note shows up in the charts too.

train.py runs this after writing the session note. By hand:

    python -m tools.vault_charts                 # dark charts (Obsidian's default theme)
    python -m tools.vault_charts --theme light
"""
import argparse
import glob
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cribbage.paths import VAULT_DIR
from tools.session_note import read_note

THEMES = {     # surface, ink, muted ink, grid, then categorical slots 1-3 (blue, orange, aqua)
    "dark": dict(surface="#1a1a19", ink="#ffffff", muted="#c3c2b7", grid="#383835",
                 series=["#3987e5", "#d95926", "#199e70"]),
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", muted="#52514e", grid="#e4e3df",
                  series=["#2a78d6", "#eb6834", "#1baf7a"]),
}
# Where the model changed shape, in lifetime steps (M): marked on the step charts.
# Each label sits at the foot of its line, on the side given.
MODEL_CHANGES = [(166, "pegging inputs", "left"), (1256, "discard inputs + teacher", "right")]


def load(folder, vault):
    rows = []
    for path in glob.glob(os.path.join(vault, folder, "*.md")):
        props = read_note(path)[0]
        if props.get("lineage"):              # an older model line (v1): not on these charts
            continue
        props["name"] = os.path.splitext(os.path.basename(path))[0]
        rows.append(props)
    return rows


def style(ax, t, title, ylabel):
    ax.set_facecolor(t["surface"])
    ax.set_title(title, loc="left", color=t["ink"], fontsize=13, pad=26)
    ax.set_ylabel(ylabel, color=t["muted"])
    ax.grid(axis="y", color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["grid"])
    ax.tick_params(colors=t["muted"], length=0)


def mark_model_changes(ax, t):
    bottom = ax.get_ylim()[0]
    for x, label, side in MODEL_CHANGES:
        ax.axvline(x, color=t["muted"], linestyle=":", linewidth=1)
        ax.text(x, bottom, f" {label} " , color=t["muted"], fontsize=8, va="bottom", ha=side)


def legend(ax, t):
    # Above the plot, under the title, so it never covers data.
    leg = ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, fontsize=9,
                    borderaxespad=0.2)
    for text in leg.get_texts():
        text.set_color(t["muted"])


def new_figure(t):
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=160)
    fig.patch.set_facecolor(t["surface"])
    return fig, ax


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


def win_rate(sessions, checkpoints, t, path):
    fig, ax = new_figure(t)
    cps = sorted((c for c in checkpoints if c.get("vs_smart") is not None), key=lambda c: c["steps_M"])
    ax.scatter([c["steps_M"] for c in cps], [c["vs_smart"] for c in cps], s=14, color=t["muted"],
               alpha=0.55, linewidths=0, label="checkpoint (500 games, ±4%)", zorder=2)
    ss = sorted((s for s in sessions if s.get("vs_smart") is not None), key=lambda s: s["steps_M"])
    xs, ys = [s["steps_M"] for s in ss], [s["vs_smart"] for s in ss]
    ax.plot(xs, ys, color=t["series"][0], linewidth=2, marker="o", markersize=6,
            markeredgecolor=t["surface"], markeredgewidth=1.5, label="end of session (2,000 games)", zorder=3)
    ax.axhline(50, color=t["ink"], linewidth=1, alpha=0.6)
    ax.text(0.5, 50.4, "even with the smart bot", transform=ax.get_yaxis_transform(),
            ha="center", color=t["muted"], fontsize=8)
    if ss:                                      # label the latest point only
        ax.annotate(f"{ss[-1]['name']}\n{ys[-1]:.1f}%", (xs[-1], ys[-1]), xytext=(8, 0),
                    textcoords="offset points", ha="left", va="center", color=t["ink"], fontsize=9)
        ax.set_xlim(right=xs[-1] * 1.12)     # room for that label
    style(ax, t, "Win % against the smart bot", "win %")
    ax.set_xlabel("lifetime training steps (millions)", color=t["muted"])
    mark_model_changes(ax, t)
    legend(ax, t)
    save(fig, path)


def discards(sessions, checkpoints, t, path):
    fig, ax = new_figure(t)
    pts = sorted((r for r in sessions + checkpoints if r.get("discards_lost") is not None),
                 key=lambda r: r["steps_M"])
    xs = [r["steps_M"] for r in pts]
    ax.plot(xs, [r["discards_lost"] for r in pts], color=t["series"][0], linewidth=2,
            marker="o", markersize=4, label="AI", zorder=3)
    bot = [(r["steps_M"], r["bot_discards_lost"]) for r in pts if r.get("bot_discards_lost") is not None]
    ax.plot([b[0] for b in bot], [b[1] for b in bot], color=t["series"][1], linewidth=2, label="smart bot", zorder=2)
    ax.set_ylim(bottom=0)
    if pts:
        ax.annotate(f"{pts[-1]['discards_lost']:.2f}", (xs[-1], pts[-1]["discards_lost"]), xytext=(6, 4),
                    textcoords="offset points", color=t["ink"], fontsize=9)
    style(ax, t, "Discard quality: points given up per deal (lower is better)", "points lost per deal")
    ax.set_xlabel("lifetime training steps (millions)", color=t["muted"])
    mark_model_changes(ax, t)
    legend(ax, t)
    save(fig, path)


def point_gaps(sessions, t, path):
    """ Points per deal minus the bot's, by source, for each measured session. """
    ss = sorted((s for s in sessions if s.get("hand_gap") is not None), key=lambda s: s["session"])
    fig, ax = new_figure(t)
    sources = [("pegging_gap", "pegging"), ("hand_gap", "hand"), ("crib_gap", "crib")]
    width = 0.26
    for k, (key, label) in enumerate(sources):
        xs = [i + (k - 1) * width for i in range(len(ss))]
        ax.bar(xs, [s.get(key) or 0 for s in ss], width=width - 0.03, color=t["series"][k], label=label, zorder=3)
    ax.axhline(0, color=t["ink"], linewidth=1, alpha=0.6, zorder=4)
    ax.set_xticks(range(len(ss)), [f"S{s['session']}" for s in ss])
    style(ax, t, "Points per deal against the bot, by source (above 0 = ahead)", "AI minus bot, points per deal")
    legend(ax, t)
    save(fig, path)


def write_charts(vault=VAULT_DIR, theme="dark"):
    """ Draws every chart and writes Charts.md. Returns the note's path, or None without a vault. """
    if not os.path.isdir(vault):
        return None
    t = THEMES[theme]
    sessions = [s for s in load("Sessions", vault) if s.get("type") == "session" and s.get("steps_M") is not None]
    checkpoints = [c for c in load("Checkpoints", vault) if c.get("steps_M") is not None]
    out = os.path.join(vault, "Charts")
    os.makedirs(out, exist_ok=True)
    win_rate(sessions, checkpoints, t, os.path.join(out, "win_rate.png"))
    discards(sessions, checkpoints, t, os.path.join(out, "discards.png"))
    point_gaps(sessions, t, os.path.join(out, "point_gaps.png"))

    note = os.path.join(vault, "Charts.md")
    with open(note, "w", encoding="utf-8") as f:
        f.write(f"""---
tags: [reference]
updated: {time.strftime('%Y-%m-%d %H:%M')}
---
# Charts

Drawn from the properties of the notes in `Sessions/` and `Checkpoints/` by `tools/vault_charts.py`, after every training run. Do not edit this note: it is rewritten each time.

## Win rate
Each blue point is the end of a session; grey dots are checkpoints part-way through. Dotted lines mark changes to the model's inputs.
![[win_rate.png]]

## Discards
The gap between the two lines is points per deal the discards win or lose against the bot.
![[discards.png]]

## Where the points come from
One group per measured session. A bar below zero is a leak against the bot.
![[point_gaps.png]]
""")
    return note


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw the training charts into the Obsidian vault.")
    parser.add_argument("--theme", choices=THEMES, default="dark")
    parser.add_argument("--vault", default=VAULT_DIR)
    args = parser.parse_args()
    print(write_charts(args.vault, args.theme) or f"No vault at {args.vault}")

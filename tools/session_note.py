""" Writes a training session's note into the Obsidian vault (cribbage.paths.VAULT_DIR):
Sessions/Session NN.md, with the numbers as properties (so the vault's Bases can filter,
group and sort sessions) and the same numbers as tables underneath for reading.

train.py calls write_note() at the end of every session. From the command line it (re)writes
notes from models/training_state.json and models/metrics_s<N>.json:

    python -m tools.session_note                 # the latest session
    python -m tools.session_note --session 12    # one session
    python -m tools.session_note --all           # every session in the history
    python -m tools.session_note --session 5 --checkpoint models/metrics_d2_t2024.json
                                                 # a metrics file from part-way through session 5

A checkpoint becomes its own note in Checkpoints/, and its session's note lists them in an
embedded base (filtered on `session == this.session`).

Rewriting a note keeps everything under its "## My notes" heading, and keeps any property
the new data has no value for (a date or commit typed in by hand, say).
"""
import argparse
import glob
import json
import os
import re

from cribbage.paths import MODELS_DIR, STATE_PATH, VAULT_DIR

SESSIONS = "Sessions"
CHECKPOINTS = "Checkpoints"
MY_NOTES = "## My notes"
DRIFT = 0.2            # discard points lost per deal above which the teacher should run again
SOURCES = ("pegging", "hand", "crib", "his heels")


# --- frontmatter -------------------------------------------------------------------------------
# The notes only use flat "key: value" properties, so a small reader and writer do instead of PyYAML.

def _parse_value(v):
    v = v.strip()
    if v == "":
        return None
    if v.startswith("[") and v.endswith("]"):
        return [x.strip().strip('"') for x in v[1:-1].split(",") if x.strip()]
    if v.startswith('"') and v.endswith('"'):
        return v[1:-1]
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v


def _format_value(v):
    if v is None:
        return ""
    if isinstance(v, list):
        return "[" + ", ".join(v) + "]"
    if isinstance(v, float):
        return f"{v:g}"
    if isinstance(v, str) and (v.startswith("[[") or ":" in v or v.startswith(("#", "-", "+"))):
        return f'"{v}"'
    return str(v)


def read_note(path):
    """ (properties, body) of a note; ({}, "") if it does not exist. """
    if not os.path.exists(path):
        return {}, ""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    m = re.match(r"---\n(.*?)\n---\n?(.*)", text, re.S)
    if not m:
        return {}, text
    props = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            props[k.strip()] = _parse_value(v)
    return props, m.group(2)


def my_notes(body):
    """ The hand-written part of an existing note. A note written before this tool (no
    "## My notes" heading) keeps everything under its title instead, so nothing is lost. """
    if MY_NOTES in body:
        return body.split(MY_NOTES, 1)[1].strip("\n")
    return re.sub(r"\A#[^\n]*\n", "", body.strip()).strip("\n")


# --- the note ----------------------------------------------------------------------------------

def note_path(session, vault=VAULT_DIR):
    return os.path.join(vault, SESSIONS, f"Session {session:02d}.md")


def model_link(obs, vault=VAULT_DIR):
    """ A link to the vault's model note with this observation size, if there is one. """
    for path in glob.glob(os.path.join(vault, "Models", "*.md")):
        if read_note(path)[0].get("obs") == obs:
            return f"[[{os.path.splitext(os.path.basename(path))[0]}]]"
    return None


def _r(x, digits=2):
    return None if x is None else round(float(x), digits)


def metric_properties(metrics):
    """ The properties every measured note carries, from a metrics.py result. """
    smart = (metrics or {}).get("vs_smart", {})
    ppd = smart.get("points_per_deal", {})
    ai_d, opp_d = smart.get("discard_ai", {}), smart.get("discard_opp", {})
    gap = lambda src: _r(ppd[src]["ai"] - ppd[src]["opp"]) if src in ppd else None
    return {
        "margin": _r(smart.get("avg_margin")),
        "discards_lost": _r(ai_d.get("avg_points_lost")),
        "bot_discards_lost": _r(opp_d.get("avg_points_lost")),
        "best_discard_pct": _r(ai_d.get("best_pct"), 1),
        "blunder_pct": _r(ai_d.get("blunder_pct"), 1),
        "pegging_gap": gap("pegging"),
        "hand_gap": gap("hand"),
        "crib_gap": gap("crib"),
        "skunks_given_pct": _r(smart.get("skunks_given_pct"), 1),
        "skunked_pct": _r(smart.get("skunked_pct"), 1),
    }


def properties(entry, metrics, prev, vault):
    win = entry.get("winrate_vs_smart")
    opponent = entry.get("opponent", "smart")         # sessions before self-play had no other
    return {
        "type": "session",
        "session": entry["session"],
        "date": entry.get("date"),
        "engine": entry.get("engine"),
        "opponent": opponent,
        "hours": _r(entry.get("hours")),
        "session_steps_M": _r(entry["session_steps"] / 1e6, 1),
        "steps_M": _r(entry["lifetime_steps"] / 1e6, 1),
        "steps_per_sec": entry.get("steps_per_sec"),
        "vs_smart": _r(win),
        "vs_smart_change": _r(win - prev["winrate_vs_smart"]) if win is not None and prev else None,
        "vs_b54": _r(entry.get("winrate_vs_baseline")),
        **metric_properties(metrics),
        "model": model_link(entry["obs"], vault) if entry.get("obs") else None,
        "previous": f"[[Session {prev['session']:02d}]]" if prev else None,
        "commit": entry.get("commit"),
        "tags": ["session", f"opponent/{'pool' if opponent and opponent.startswith('pool') else 'smart'}"],
    }


def _fmt(x, spec="+.2f"):
    return "" if x is None else format(x, spec)


CHECKPOINT_BASE = """```base
filters:
  and:
    - type == "checkpoint"
    - session == this.session
views:
  - type: table
    name: Checkpoints
    order:
      - file.name
      - games
      - vs_smart
      - margin
      - discards_lost
      - pegging_gap
      - hand_gap
      - crib_gap
    sort:
      - property: steps_M
        direction: ASC
```"""


def has_checkpoints(session, vault):
    return any(read_note(path)[0].get("session") == session
               for path in glob.glob(os.path.join(vault, CHECKPOINTS, "*.md")))


def body(p, entry, metrics, prev, vault=VAULT_DIR, prev_discards=None):
    n = p["session"]
    lines = [f"# Training session {n}", ""]

    # The headline, as a callout.
    win = p.get("vs_smart")
    if win is not None:
        change = win - prev["winrate_vs_smart"] if prev else None
        ci = (metrics or {}).get("vs_smart", {}).get("win_pct_ci95")
        lines.append("> [!summary] " + f"{win:.1f}% vs smart" + (f" (±{ci:.1f})" if ci else "")
                     + (f", {change:+.1f} on session {prev['session']}" if change is not None else ""))
    else:
        lines.append("> [!summary] Not evaluated")
    run = [f"{p['session_steps_M']:,.1f}M steps"]
    if p.get("hours"):
        run.append(f"in {p['hours']:g} h")
    if p.get("engine"):
        run.append(f"on the {p['engine']} engine")
    if p.get("opponent"):
        run.append(f"against {p['opponent']}")
    total = f"{p['steps_M'] / 1000:.2f}B" if p["steps_M"] >= 1000 else f"{p['steps_M']:,.0f}M"
    lines += ["> " + " ".join(run) + f"; {total} steps in all.", ""]
    # Only a drift if the session before had good (taught) discards.
    if p.get("discards_lost") is not None and p["discards_lost"] > DRIFT and prev_discards is not None \
            and prev_discards <= DRIFT:
        lines += [f"> [!warning] Discards have drifted to {p['discards_lost']:.2f} points lost per deal "
                  f"(above {DRIFT}). Run the discard teacher again.", ""]

    # Results against the previous session.
    rows = [("Win % vs smart", p.get("vs_smart"), prev and prev.get("winrate_vs_smart"), ".1f"),
            ("Win % vs b54 (frozen 54% model)", p.get("vs_b54"), prev and prev.get("winrate_vs_baseline"), ".1f"),
            ("Lifetime steps (M)", p["steps_M"], prev and prev["lifetime_steps"] / 1e6, ",.1f")]
    lines += ["## Results", "", "| | This session | Previous | Change |", "| --- | --- | --- | --- |"]
    for name, now, before, spec in rows:
        if now is None:
            continue
        change = "" if before is None else format(now - before, "+" + spec)
        lines.append(f"| {name} | {format(now, spec)} | {'' if before is None else format(before, spec)} | {change} |")
    lines.append("")

    smart = (metrics or {}).get("vs_smart")
    if smart:
        lines += [f"Measured over {smart['games']:,} seeded games against the smart bot.", "",
                  "| | Value |", "| --- | --- |",
                  f"| Average margin (points) | {smart['avg_margin']:+.1f} |",
                  f"| Deals per game | {smart['avg_deals_per_game']:.1f} |",
                  f"| Skunks given / taken | {smart['skunks_given_pct']:.1f}% / {smart['skunked_pct']:.1f}% |", ""]

        ppd = smart["points_per_deal"]
        lines += ["## Points per deal", "", "Where the margin comes from. A negative gap is a leak.", "",
                  "| Source | AI | Smart bot | Gap |", "| --- | --- | --- | --- |"]
        for src in SOURCES:
            if src in ppd:
                ai, opp = ppd[src]["ai"], ppd[src]["opp"]
                lines.append(f"| {src.capitalize()} | {ai:.2f} | {opp:.2f} | {ai - opp:+.2f} |")
        lines.append("")

        ai_d, opp_d = smart.get("discard_ai"), smart.get("discard_opp")
        if ai_d and opp_d:
            lines += ["## Discards", "",
                      "Each discard rated against the best of the 15 (kept hand over every starter, plus or minus the crib).", "",
                      "| | AI | Smart bot |", "| --- | --- | --- |",
                      f"| Points lost per deal | {ai_d['avg_points_lost']:.2f} | {opp_d['avg_points_lost']:.2f} |",
                      f"| Best pick | {ai_d['best_pct']:.0f}% | {opp_d['best_pct']:.0f}% |",
                      f"| Top-3 pick | {ai_d['top3_pct']:.0f}% | {opp_d['top3_pct']:.0f}% |",
                      f"| Blunders (2+ points) | {ai_d['blunder_pct']:.1f}% | {opp_d['blunder_pct']:.1f}% |", ""]

    if has_checkpoints(n, vault):
        lines += ["## Checkpoints", "", "Snapshots measured part-way through the session (500 games each: about ±4%).", "",
                  CHECKPOINT_BASE, ""]

    run_lines = []
    if entry.get("command"):
        run_lines.append(f"- Command: `{entry['command']}`")
    if p.get("steps_per_sec"):
        run_lines.append(f"- Speed: {p['steps_per_sec']:,} steps/s")
    if metrics:
        run_lines.append(f"- Metrics: `models/metrics_s{n}.json`")
    if prev:
        run_lines.append(f"- Previous: [[Session {prev['session']:02d}]]")
    if run_lines:
        lines += ["## Run", ""] + run_lines + [""]
    return "\n".join(lines)


def write_note(entry, metrics=None, prev=None, vault=VAULT_DIR):
    """ Writes (or rewrites) the note for one history entry of training_state.json.
    metrics: the dict metrics.py writes, if the session was measured. prev: the previous
    history entry, for the changes. Returns the note's path, or None if there is no vault. """
    if not os.path.isdir(vault):
        return None
    path = note_path(entry["session"], vault)
    old_props, old_body = read_note(path)

    props = properties(entry, metrics, prev, vault)
    for k, v in old_props.items():            # keep what was typed in by hand
        if props.get(k) is None:
            props[k] = v

    front = "\n".join(f"{k}: {_format_value(v)}" for k, v in props.items() if v is not None)
    prev_discards = read_note(note_path(prev["session"], vault))[0].get("discards_lost") if prev else None
    text = (f"---\n{front}\n---\n" + body(props, entry, metrics, prev, vault, prev_discards)
            + f"\n{MY_NOTES}\n\n" + (my_notes(old_body) or "-") + "\n")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def write_checkpoint(metrics, session, source, vault=VAULT_DIR):
    """ A note in Checkpoints/ for a metrics file measured part-way through a session.
    The note is all properties; the session's note shows them as a table. """
    if not os.path.isdir(vault):
        return None
    smart = metrics["vs_smart"]
    steps_m = metrics["steps"] / 1e6
    props = {
        "type": "checkpoint",
        "session": session,
        "session_note": f"[[Session {session:02d}]]",
        "steps_M": _r(steps_m, 1),
        "games": smart["games"],
        "vs_smart": _r(smart["win_pct"]),
        "vs_smart_ci": _r(smart["win_pct_ci95"]),
        **metric_properties(metrics),
        "measured": metrics.get("created"),
        "source": os.path.basename(source),
        "tags": ["checkpoint"],
    }
    path = os.path.join(vault, CHECKPOINTS, f"Checkpoint {steps_m:06.1f}M.md")
    front = "\n".join(f"{k}: {_format_value(v)}" for k, v in props.items() if v is not None)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"---\n{front}\n---\nThe model at {steps_m:,.1f}M steps, during [[Session {session:02d}]]: "
                f"{smart['win_pct']:.1f}% vs smart over {smart['games']:,} games.\n")
    return path


def load_metrics(session, models_dir=MODELS_DIR):
    path = os.path.join(models_dir, f"metrics_s{session}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def write_from_state(sessions=None, models_dir=MODELS_DIR, vault=VAULT_DIR):
    """ Writes the notes for these session numbers (default: the latest) from training_state.json. """
    with open(os.path.join(models_dir, os.path.basename(STATE_PATH))) as f:
        history = json.load(f)["history"]
    sessions = sessions or [history[-1]["session"]]
    paths = []
    for i, entry in enumerate(history):
        if entry["session"] in sessions:
            prev = next((h for h in reversed(history[:i]) if h.get("winrate_vs_smart") is not None), None)
            paths.append(write_note(entry, load_metrics(entry["session"], models_dir), prev, vault))
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Write training session notes into the Obsidian vault.")
    parser.add_argument("--session", type=int, nargs="*", help="session numbers (default: the latest)")
    parser.add_argument("--all", action="store_true", help="every session in the history")
    parser.add_argument("--checkpoint", nargs="*", help="metrics files from part-way through --session")
    parser.add_argument("--models", default=MODELS_DIR, help="the models folder holding training_state.json")
    parser.add_argument("--vault", default=VAULT_DIR)
    args = parser.parse_args()

    if not os.path.isdir(args.vault):
        raise SystemExit(f"No vault at {args.vault} (set CRIBBAGE_VAULT or pass --vault)")
    if args.checkpoint:
        if not args.session or len(args.session) != 1:
            raise SystemExit("--checkpoint needs exactly one --session")
        for src in args.checkpoint:
            with open(src) as f:
                print(f"wrote {write_checkpoint(json.load(f), args.session[0], src, args.vault)}")
    with open(os.path.join(args.models, os.path.basename(STATE_PATH))) as f:
        everything = [h["session"] for h in json.load(f)["history"]]
    for path in write_from_state(everything if args.all else args.session, args.models, args.vault):
        print(f"wrote {path}")
    from tools.vault_charts import write_charts      # the charts read the notes just written
    print(f"wrote {write_charts(args.vault)}")

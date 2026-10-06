""" Rebuilds a game played in play.py from its replay file, and prints the position.

    python -m tools.replay_game replays/K7QM2XPA.json             # the position at the end
    python -m tools.replay_game replays/K7QM2XPA.json --upto 23   # just before move 23

The cards come from the seed key and every move from the file (the AI's included), so the
game comes out exactly as it was played, whatever has happened to the model since. It uses
the Python engine: your moves stand in for the computer's, which the Rust engine cannot do.
Every score is listed with its reason, deal by deal, as the board replay showed it.
"""
import argparse
import json

from cribbage.env import CribbageEnv, DISCARD_PAIRS, N_CARDS
from cribbage.seeds import key_to_seed
from cribbage.ui import card_str
from play import deal_counts, record_scores


class EndOfMoves(Exception):
    """ Raised where the replay runs out of moves: the position to show. """


def cards(cs):
    return " ".join(card_str(c) for c in cs) or "-"


def rebuild(replay, upto=None):
    """ Plays the recorded moves (all, or those before move number `upto`, counting from 1).
    Returns (env, log, moves played, what the game is waiting for). """
    moves = replay["moves"] if upto is None else replay["moves"][:upto - 1]
    env = CribbageEnv("smart")
    env.reset(seed=key_to_seed(replay["seed"]))
    yours = [m for m in moves if m["by"] == "you"]
    ai = [m for m in moves if m["by"] == "ai"]

    def next_human(kind):
        if not yours:
            raise EndOfMoves(kind)
        return yours.pop(0)["cards"]

    env._comp_discard = lambda: next_human("your discard")
    env._comp_pick = lambda legal: next_human("your play")[0]

    # Score log with reasons, using play.py's own bookkeeping.
    state = {"events": [], "count_labels": []}
    log = []
    count_hands = env._count_hands

    def count_and_log():
        counts = deal_counts(env)
        state["count_labels"] = ["crib" if "CRIB" in label else "hand" for label, *_ in counts]
        r = count_hands()
        state["count_labels"] = []
        flush(f"deal {env.deals} counts: " + " | ".join(
            f"{label} {cards(cs)} = {pts}" for label, cs, pts, *_ in counts))
        return r

    def flush(note=None):
        for player, pts, reason in state["events"]:
            log.append(f"  {'AI ' if player == 0 else 'YOU'} +{pts:<2} {reason}")
        state["events"] = []
        if note:
            log.append(note)

    env._count_hands = count_and_log
    record_scores(env, state)

    ai_played = 0
    waiting = "game over" if upto is None else None
    try:
        for m in ai:
            a = m["action"]
            if env.phase == 0:
                i, j = DISCARD_PAIRS[a - N_CARDS]
                log.append(f"deal {env.deals}: AI discards {cards([env.hand[i], env.hand[j]])}")
            ai_played += 1
            _, _, done, trunc, info = env.step(a)
            if info:
                raise SystemExit(f"AI move {ai_played}: the recorded move was rejected: {info}")
            if done or trunc:
                break
        else:
            waiting = "the AI's move"
    except EndOfMoves as e:
        waiting = str(e)
    flush()
    yours_played = sum(m["by"] == "you" for m in moves) - len(yours)
    return env, log, ai_played + yours_played, waiting


def show(env, log, played, waiting, replay):
    print(f"seed {replay['seed']} | model {replay.get('model')} ({replay.get('model_steps', 0):,} steps) | "
          f"{played} moves replayed | waiting for: {waiting}")
    print(f"deal {env.deals}, {'discard' if env.phase == 0 else 'pegging'}, dealer: "
          f"{'AI' if env.is_ai_dealer else 'YOU'} | score AI {env.p1_score}, YOU {env.comp_score}"
          + (f" | {'AI' if env.winner == 0 else 'YOU'} won" if env.game_over else ""))
    print(f"starter {card_str(env.starter_card) if env.starter_card != -1 else '-'} | "
          f"count {env.current_peg_sum}: {cards(env.pegging_table)} | pegged this deal: {cards(env.peg_history)}")
    print(f"AI hand {cards(env.hand)} | your hand {cards(env.comp_hand)} | crib {cards(env.crib)}")
    print("\nscoring log:")
    print("\n".join(log))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rebuild a play.py game from its replay file.")
    parser.add_argument("replay")
    parser.add_argument("--upto", type=int, help="stop just before this move number")
    args = parser.parse_args()
    with open(args.replay) as f:
        replay = json.load(f)
    show(*rebuild(replay, args.upto), replay)

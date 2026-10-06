import random
import numpy as np
from cribbage.env import CribbageEnv     # Python engine: _comp_pick is swapped out below

# Isolates the pegging change: the computer plays with the old 1-ply pick vs the new
# 2-ply pick, against the same 1-ply "AI" stand-in, over identical seeded deals.
# Discards use the smart heuristic on both sides of the comparison, so only pegging differs.

GAMES = 2000


def one_ply(env, legal, table, peg_sum):
    """ The pre-2-ply _comp_pick heuristic, kept here as the baseline. """
    best_card, best_val = None, -99.0
    for c in legal:
        new_sum = peg_sum + env._card_value(c)
        val = env._peg_points(table + [c]) + random.random() * 0.5
        if new_sum in (5, 10, 21):
            val -= 1.0
        if val > best_val:
            best_card, best_val = c, val
    return best_card


def run(comp_mode, games=GAMES):
    env = CribbageEnv(opponent="smart")
    if comp_mode == "1-ply":
        env._comp_pick = lambda legal: one_ply(env, legal, env.pegging_table, env.current_peg_sum)

    # Split pegging points from hand/crib points by snapshotting scores before hands are counted.
    peg = {"ai": 0, "comp": 0}
    count_hands = env._count_hands

    def counted():
        peg["ai"] += env.p1_score - start[0]
        peg["comp"] += env.comp_score - start[1]
        return count_hands()
    env._count_hands = counted

    comp_wins, deals = 0, 0
    for g in range(games):
        random.seed(g)
        env.reset(seed=g)
        done = trunc = False
        start = (env.p1_score, env.comp_score)
        while not (done or trunc):
            if env.phase == 0:
                start = (env.p1_score, env.comp_score)
                a = env.discard_action(ai_discard(env))
            else:
                legal = [c for c in env.hand if env.current_peg_sum + env._card_value(c) <= 31]
                a = one_ply(env, legal, env.pegging_table, env.current_peg_sum)
            _, _, done, trunc, _ = env.step(a)
        comp_wins += env.winner == 1
        deals += env.deals
    return 100.0 * comp_wins / games, (peg["comp"] - peg["ai"]) / deals


def ai_discard(env):
    """ AI stand-in discards with the same smart heuristic as the computer, from its own seat. """
    env.comp_hand, env.hand = env.hand, env.comp_hand
    env.is_ai_dealer = not env.is_ai_dealer
    pair = CribbageEnv._comp_discard(env)
    env.comp_hand, env.hand = env.hand, env.comp_hand
    env.is_ai_dealer = not env.is_ai_dealer
    return list(pair)


if __name__ == "__main__":
    for mode in ("1-ply", "2-ply"):
        win, margin = run(mode)
        print(f"computer {mode}: wins {win:5.1f}% vs 1-ply AI stand-in | "
              f"pegging margin {margin:+.2f} pts/deal")

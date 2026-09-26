import numpy as np
from cribbage import CribbageEnv, OBS_SIZE


def run(opp, games=1000):
    env = CribbageEnv(opponent=opp)
    lens, deals, wins = [], [], 0
    for _ in range(games):
        obs, _ = env.reset()
        assert obs.shape == (OBS_SIZE,)
        done = trunc = False
        n = 0
        while not (done or trunc):
            a = int(np.random.choice(np.flatnonzero(env.action_masks())))
            obs, r, done, trunc, info = env.step(a)
            assert "error" not in info, info
            n += 1
        assert not trunc, "timeout hit"
        assert env.game_over and max(env.p1_score, env.comp_score) >= 121
        assert min(env.p1_score, env.comp_score) < 121
        lens.append(n)
        deals.append(env.deals)
        wins += env.winner == 0
    print(f"vs {opp:6s}: avg decisions/game {np.mean(lens):.1f} | "
          f"avg deals/game {np.mean(deals):.1f} | random player wins {100.0 * wins / games:.1f}%")


def check_peg_breakdown(trials=20000):
    """ peg_breakdown must always itemise exactly what _peg_points scores. """
    env = CribbageEnv()
    rng = np.random.default_rng(0)
    for _ in range(trials):
        deck = rng.permutation(52)
        seq, total = [], 0
        for c in deck:
            if total + env._card_value(int(c)) > 31 or len(seq) == 8:
                break
            seq.append(int(c))
            total += env._card_value(int(c))
            assert sum(p for _, p in env.peg_breakdown(seq)) == env._peg_points(seq), seq
    print("peg_breakdown matches _peg_points.")


def check_hand_breakdown(trials=20000):
    """ hand_breakdown must always itemise exactly what score_hand scores. """
    env = CribbageEnv()
    rng = np.random.default_rng(0)
    for _ in range(trials):
        deck = [int(c) for c in rng.permutation(52)[:5]]
        hand, starter = deck[:4], deck[4]
        for crib in (False, True):
            items = env.hand_breakdown(hand, starter, is_crib=crib)
            assert sum(p for _, p in items) == env.score_hand(hand, starter, is_crib=crib), (hand, starter)
    print("hand_breakdown matches score_hand.")


if __name__ == "__main__":
    run("random")
    run("smart")
    print("All games ended cleanly.")
    check_peg_breakdown()
    check_hand_breakdown()

""" Detailed metrics for a saved model: win rates, skunks, where the points come from,
and how good its crib discards are. Writes models/metrics.json.

Discard quality: for every discard, all 15 ways to split the 6 cards are rated by expected
points -- the kept hand over every possible starter, plus (as dealer) or minus (as pone) the
crib, whose other 2 cards are sampled from the unseen cards. "Points lost" is how far the
chosen discard falls short of the best one. The heuristic opponent is rated the same way,
as a yardstick. Pegging is ignored, so this measures the counting side of the discard only.
"""
import argparse
import itertools
import json
import os
import random
import time
from multiprocessing import Pool

import numpy as np

from cribbage import CribbageEnv
from cribbage.env import DISCARD_PAIRS
from cribbage.paths import MODEL_PATH, MODELS_DIR
from evaluate import EVAL_SEED

CRIB_SAMPLES = 10            # random pairs of opponent crib cards per starter
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")


def discard_values(env, six, is_dealer, rng):
    """ Expected points of each of the 15 discards from `six`, keyed by the discarded pair
    (frozenset). The same crib samples are used for every option, so they compare fairly. """
    unseen = [c for c in range(52) if c not in six]
    samples = {st: [rng.sample([c for c in unseen if c != st], 2) for _ in range(CRIB_SAMPLES)]
               for st in unseen}
    values = {}
    for pair in itertools.combinations(six, 2):
        keep = [c for c in six if c not in pair]
        hand = crib = 0.0
        for st in unseen:
            hand += env.score_hand(keep, st)
            for other in samples[st]:
                crib += env.score_hand(list(pair) + other, st, is_crib=True)
        hand /= len(unseen)
        crib /= len(unseen) * CRIB_SAMPLES
        values[frozenset(pair)] = hand + (crib if is_dealer else -crib)
    return values


def rate(values, chosen):
    """ (points lost vs the best discard, rank 1-15 of the chosen one). """
    ranked = sorted(values.values(), reverse=True)
    v = values[frozenset(chosen)]
    return ranked[0] - v, ranked.index(v) + 1


def play_games(args):
    """ Plays a block of seeded games vs `opponent`, recording everything. Runs in a worker. """
    model_path, opponent, first, count, rate_discards = args
    from sb3_contrib import MaskablePPO
    import torch
    torch.set_num_threads(1)
    model = MaskablePPO.load(model_path, device="cpu")
    env = CribbageEnv(opponent=opponent)
    rng = random.Random(first)

    games, discards = [], {"ai": [], "opp": []}
    points = {src: [0, 0] for src in ("pegging", "hand", "crib", "his heels")}   # [ai, opp]
    source = {"now": None, "counts": []}

    award = env._award

    def award_tracked(player, pts):
        if source["counts"]:
            src = source["counts"].pop(0)
        elif env.phase == 0:
            src = "his heels"
        else:
            src = "pegging"
        if pts and not env.game_over:
            points[src][player] += pts
        return award(player, pts)
    env._award = award_tracked

    count_hands = env._count_hands

    def count_tracked():
        source["counts"] = ["hand", "hand", "crib"]
        r = count_hands()
        source["counts"] = []
        return r
    env._count_hands = count_tracked

    comp_discard = env._comp_discard

    def comp_discard_rated():
        pair = comp_discard()
        if rate_discards:
            vals = discard_values(env, list(env.comp_hand), not env.is_ai_dealer, rng)
            discards["opp"].append(rate(vals, pair))
        return pair
    env._comp_discard = comp_discard_rated

    for g in range(first, first + count):
        obs, _ = env.reset(seed=EVAL_SEED + g)
        done = trunc = False
        while not (done or trunc):
            a, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=True)
            a = int(a)
            if env.phase == 0 and rate_discards:
                i, j = DISCARD_PAIRS[a - 52]
                chosen = [env.hand[i], env.hand[j]]
                vals = discard_values(env, list(env.hand), env.is_ai_dealer, rng)
                discards["ai"].append(rate(vals, chosen))
            obs, r, done, trunc, _ = env.step(a)
        ai, opp = min(env.p1_score, 121), min(env.comp_score, 121)
        games.append({"won": env.winner == 0, "ai": ai, "opp": opp, "deals": env.deals})
    return games, discards, points


def summarize(games, discards, points):
    n = len(games)
    wins = sum(g["won"] for g in games)
    p = wins / n
    deals = sum(g["deals"] for g in games)
    losers = [(g["won"], g["opp"] if g["won"] else g["ai"]) for g in games]
    out = {
        "games": n,
        "win_pct": 100 * p,
        "win_pct_ci95": 100 * 1.96 * (p * (1 - p) / n) ** 0.5,
        "avg_deals_per_game": deals / n,
        "avg_margin": float(np.mean([g["ai"] - g["opp"] for g in games])),
        "skunks_given_pct": 100 * sum(w and 60 < s <= 90 for w, s in losers) / n,
        "double_skunks_given_pct": 100 * sum(w and s <= 60 for w, s in losers) / n,
        "skunked_pct": 100 * sum((not w) and 60 < s <= 90 for w, s in losers) / n,
        "double_skunked_pct": 100 * sum((not w) and s <= 60 for w, s in losers) / n,
        "points_per_deal": {src: {"ai": v[0] / deals, "opp": v[1] / deals} for src, v in points.items()},
    }
    for who, rows in discards.items():
        if rows:
            lost = np.array([r[0] for r in rows])
            ranks = np.array([r[1] for r in rows])
            out[f"discard_{who}"] = {
                "n": len(rows),
                "avg_points_lost": float(lost.mean()),
                "best_pct": float(100 * (ranks == 1).mean()),
                "top3_pct": float(100 * (ranks <= 3).mean()),
                "blunder_pct": float(100 * (lost >= 2).mean()),     # gave up 2+ points
                "rank_counts": [int((ranks == k).sum()) for k in range(1, 16)],
            }
    return out


def run(model_path, opponent, games, rate_discards, workers):
    block = -(-games // workers)
    jobs = [(model_path, opponent, s, min(block, games - s), rate_discards)
            for s in range(0, games, block)]
    with Pool(len(jobs)) as pool:
        results = pool.map(play_games, jobs)
    all_games, all_discards = [], {"ai": [], "opp": []}
    all_points = {src: [0, 0] for src in results[0][2]}
    for g, d, p in results:
        all_games += g
        for k in d:
            all_discards[k] += d[k]
        for src in p:
            all_points[src][0] += p[src][0]
            all_points[src][1] += p[src][1]
    return summarize(all_games, all_discards, all_points)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detailed metrics for a saved model.")
    parser.add_argument("model", nargs="?", default=MODEL_PATH)
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--no-random", action="store_true", help="skip the games vs the random player")
    parser.add_argument("--out", default=METRICS_PATH)
    args = parser.parse_args()

    from sb3_contrib import MaskablePPO
    steps = MaskablePPO.load(args.model, device="cpu").num_timesteps
    t0 = time.time()
    metrics = {
        "model": os.path.basename(args.model), "steps": int(steps),
        "created": time.strftime("%Y-%m-%d %H:%M"),
        "vs_smart": run(args.model, "smart", args.games, True, args.workers),
    }
    if not args.no_random:
        metrics["vs_random"] = run(args.model, "random", args.games, False, args.workers)
    with open(args.out, "w") as f:
        json.dump(metrics, f, indent=2)

    s = metrics["vs_smart"]
    print(f"{steps:,} steps | {time.time() - t0:.0f}s")
    vs_random = f" | vs random: {metrics['vs_random']['win_pct']:.1f}%" if "vs_random" in metrics else ""
    print(f"vs smart : {s['win_pct']:.1f}% ±{s['win_pct_ci95']:.1f}{vs_random}")
    for who in ("ai", "opp"):
        d = s[f"discard_{who}"]
        print(f"discards ({who:3}): {d['avg_points_lost']:.2f} pts lost/deal | best {d['best_pct']:.0f}% | "
              f"blunders {d['blunder_pct']:.0f}%")
    print(f"written to {args.out}")

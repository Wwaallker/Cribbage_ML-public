""" The Rust engine (cribbage/rust_env.py) must match the Python engine (cribbage/env.py) exactly.

The Rust engine carries a copy of CPython's random generator, so beyond identical scoring on
random inputs, seeded games are compared move for move: every observation, mask, reward and
score, and the opponent's every choice (its random tie-break noise included).

    python -m tests.test_rust_engine            # everything, a few minutes
    python -m tests.test_rust_engine --quick    # fewer games, and no model evaluation

The model evaluation uses models/current.zip, or the model named by CRIBBAGE_MODEL.
"""
import os
import random
import sys
import tempfile

import numpy as np

from cribbage.env import CribbageEnv, N_ACTIONS, OBS_SIZE
from cribbage.paths import MODEL_PATH
from cribbage.rust_env import RustCribbageEnv, STATE_FIELDS

QUICK = "--quick" in sys.argv
N_INPUTS = 100_000


def _pegging_seq(rng, max_len=8):
    """ A random legal count: cards in play order, total at most 31. """
    seq, total = [], 0
    for c in rng.permutation(52):
        c = int(c)
        v = 10 if c % 13 >= 9 else c % 13 + 1
        if total + v > 31 or len(seq) == max_len:
            break
        seq.append(c)
        total += v
    return seq


def test_score_hand(trials=N_INPUTS):
    py, rs = CribbageEnv(), RustCribbageEnv()
    rng = np.random.default_rng(1)
    for _ in range(trials):
        cards = [int(c) for c in rng.permutation(52)[:5]]
        hand, starter = cards[:4], cards[4]
        for st in (starter, None):
            for crib in (False, True):
                assert py.score_hand(hand, st, crib) == rs.score_hand(hand, st, crib), (hand, st, crib)
    print(f"score_hand identical on {trials:,} hands (with and without starter, hand and crib)")


def test_peg_points(trials=N_INPUTS):
    py, rs = CribbageEnv(), RustCribbageEnv()
    rng = np.random.default_rng(2)
    for _ in range(trials):
        seq = _pegging_seq(rng)
        for k in range(1, len(seq) + 1):
            assert py._peg_points(seq[:k]) == rs._peg_points(seq[:k]), seq[:k]
    print(f"_peg_points identical on {trials:,} counts (every prefix)")


def test_breakdowns_match_rust_scoring(trials=20_000):
    """ The display breakdowns are shared Python code; they must still add up to Rust's scores. """
    rs = RustCribbageEnv()
    rng = np.random.default_rng(3)
    for _ in range(trials):
        cards = [int(c) for c in rng.permutation(52)[:5]]
        for crib in (False, True):
            assert sum(p for _, p in rs.hand_breakdown(cards[:4], cards[4], crib)) == \
                rs.score_hand(cards[:4], cards[4], crib)
        seq = _pegging_seq(rng)
        assert sum(p for _, p in rs.peg_breakdown(seq)) == rs._peg_points(seq)
    print(f"hand_breakdown / peg_breakdown add up to the Rust scores on {trials:,} inputs")


def test_discard_values(trials=2_000):
    """ The discard_value inputs: the 15 discard values of a hand, as dealer and as pone. """
    py, rs = CribbageEnv(), RustCribbageEnv()
    rng = np.random.default_rng(6)
    for _ in range(trials):
        six = sorted((int(c) for c in rng.permutation(52)[:6]), key=lambda c: (c % 13, c // 13))
        for owns_crib in (False, True):
            a, b = py.discard_values(six, owns_crib), rs.discard_values(six, owns_crib)
            assert a == b, (six, owns_crib, a, b)
    print(f"discard_values bit-identical on {trials:,} hands (as dealer and as pone)")


def _random_pegging_state(rng):
    """ A plausible mid-pegging position: history, a live count, a starter, a hand, discards. """
    deck = [int(c) for c in rng.permutation(52)]
    history = _pegging_seq(rng, max_len=int(rng.integers(0, 8)))
    rest = [c for c in deck if c not in history]
    cut = int(rng.integers(0, len(history) + 1))
    table = history[cut:]                  # the live count is the tail of the history
    starter = rest.pop() if rng.random() < 0.9 else -1
    own_hand = [rest.pop() for _ in range(int(rng.integers(0, 5)))]
    own_discards = [rest.pop() for _ in range(2)]
    return history, table, starter, own_hand, own_discards, rest


def test_expected_best_reply(trials=N_INPUTS):
    py, rs = CribbageEnv(), RustCribbageEnv()
    rng = np.random.default_rng(4)
    nonzero = 0
    for _ in range(trials):
        history, table, starter, own_hand, own_discards, rest = _random_pegging_state(rng)
        for env in (py, rs):
            env.peg_history, env.pegging_table, env.starter_card = history, table, starter
            env.current_peg_sum = sum(10 if c % 13 >= 9 else c % 13 + 1 for c in table)
        card = own_hand[0] if own_hand and rng.random() < 0.8 else rest.pop()
        n_opp = int(rng.integers(0, 5))
        a = py._expected_best_reply(card, own_hand, own_discards, n_opp)
        b = rs._expected_best_reply(card, own_hand, own_discards, n_opp)
        assert a == b, (history, table, starter, own_hand, own_discards, card, n_opp, a, b)
        nonzero += a != 0
    print(f"_expected_best_reply bit-identical on {trials:,} positions ({nonzero:,} non-zero)")


def _copy_state(src, dst):
    for name in STATE_FIELDS:
        value = getattr(src, name)
        setattr(dst, name, list(value) if isinstance(value, list) else value)


def test_obs_and_masks_for_same_state(positions=20_000):
    """ Positions reached by the Python engine, copied into the Rust engine. """
    py, rs = CribbageEnv("smart"), RustCribbageEnv("smart")
    rng = np.random.default_rng(5)
    random.seed(5)
    checked = 0
    while checked < positions:
        py.reset()
        done = trunc = False
        while not (done or trunc) and checked < positions:
            _copy_state(py, rs)
            assert np.array_equal(py._get_obs(), rs._get_obs()), (checked, py.phase)
            assert np.array_equal(py._get_obs(1), rs._get_obs(1)), (checked, py.phase)
            assert np.array_equal(py.action_masks(), rs.action_masks()), checked
            checked += 1
            _, _, done, trunc, _ = py.step(int(rng.choice(np.flatnonzero(py.action_masks()))))
    print(f"observations (both seats) and masks identical on {positions:,} copied positions")


def _random_net(path, seed=7):
    """ An opponent network file with random weights, shaped like tools/export_opponent.py's. """
    rng = np.random.default_rng(seed)
    shapes = [(256, OBS_SIZE), (256, 256), (N_ACTIONS, 256)]
    arrays = {}
    for k, (rows, cols) in enumerate(shapes):
        arrays[f"w{k}"] = (rng.standard_normal((rows, cols)) / cols ** 0.5).astype(np.float32)
        arrays[f"b{k}"] = (rng.standard_normal(rows) * 0.1).astype(np.float32)
    np.savez(path, **arrays)
    return path


def test_opponent_net(tmpdir, positions=2_000):
    """ An opponent network gives the same logits on both engines. """
    path = _random_net(os.path.join(tmpdir, "net.npz"))
    py = CribbageEnv("pool", pool=[path])
    rs = RustCribbageEnv("pool", pool=[path])
    rng = np.random.default_rng(8)
    worst = 0.0
    for _ in range(positions):
        obs = (rng.random(OBS_SIZE) < 0.05).astype(np.float32)
        worst = max(worst, float(np.abs(py.pool[0].logits(obs) - np.array(rs._core.net_logits(0, list(obs)))).max()))
    assert worst < 1e-9, worst
    print(f"opponent network logits match on {positions:,} inputs (largest difference {worst:.1e})")
    return path


def _python_points_tally(env):
    """ Points by source on the Python engine, tallied the way metrics.py does. """
    points = {src: [0, 0] for src in ("pegging", "hand", "crib", "his heels")}
    queue = []
    award, count_hands = env._award, env._count_hands

    def award_tracked(player, pts):
        src = queue.pop(0) if queue else ("his heels" if env.phase == 0 else "pegging")
        if pts and not env.game_over:
            points[src][player] += pts
        return award(player, pts)

    def count_tracked():
        queue[:] = ["hand", "hand", "crib"]
        r = count_hands()
        queue.clear()
        return r

    env._award, env._count_hands = award_tracked, count_tracked
    return points


def lockstep(opponent, games, seed0=0, **kwargs):
    """ Plays the same seeded games on both engines with the same (random legal) actions,
    asserting they stay identical. Returns summary statistics, which are then equal too. """
    py, rs = CribbageEnv(opponent, **kwargs), RustCribbageEnv(opponent, **kwargs)
    py_points = _python_points_tally(py)
    rs_points = {src: [0, 0] for src in py_points}
    steps = wins = deals = 0
    for g in range(games):
        o1, _ = py.reset(seed=seed0 + g)
        o2, _ = rs.reset(seed=seed0 + g)
        assert py.opp_smart == rs.opp_smart
        assert (py.opp_net is None) == (rs.opp_net is None)
        rng = np.random.default_rng(seed0 + g)
        done = trunc = False
        while True:
            assert np.array_equal(o1, o2), (opponent, g, steps)
            m1, m2 = py.action_masks(), rs.action_masks()
            assert np.array_equal(m1, m2), (opponent, g, steps)
            if done or trunc:
                break
            a = int(rng.choice(np.flatnonzero(m1)))
            o1, r1, d1, t1, i1 = py.step(a)
            o2, r2, d2, t2, i2 = rs.step(a)
            assert (r1, d1, t1, i1) == (r2, d2, t2, i2), (opponent, g, steps, r1, r2)
            assert (py.p1_score, py.comp_score, py.crib) == (rs.p1_score, rs.comp_score, rs.crib)
            done, trunc = d1, t1
            steps += 1
        assert (py.deals, py.winner) == (rs.deals, rs.winner)
        for src, (ai, opp) in rs.points_by_source().items():
            rs_points[src][0] += ai
            rs_points[src][1] += opp
        wins += py.winner == 0
        deals += py.deals
    assert py_points == rs_points, (py_points, rs_points)
    per_deal = {src: [round(v / deals, 3) for v in pts] for src, pts in rs_points.items()}
    print(f"vs {opponent:6}: {games:,} seeded games, {steps:,} steps identical | random player wins "
          f"{100 * wins / games:.1f}% | {deals / games:.2f} deals/game | points/deal [ai, opp] {per_deal}")


def test_lockstep_games(net_path):
    lockstep("smart", 500 if QUICK else 5000)
    lockstep("random", 200 if QUICK else 2000, seed0=100_000)
    lockstep("mixed", 200 if QUICK else 2000, seed0=200_000)
    lockstep("pool", 200 if QUICK else 2000, seed0=300_000, pool=[net_path, net_path], p_smart=0.3)


def test_bad_actions():
    """ Illegal moves and the step limit give the same penalties on both engines. """
    for bad in (-1, 0, 67, 51):
        py, rs = CribbageEnv(), RustCribbageEnv()
        py.reset(seed=9)
        rs.reset(seed=9)
        a = py.step(bad)
        b = rs.step(bad)
        assert np.array_equal(a[0], b[0]) and a[1:] == b[1:] and a[4] == {"error": "illegal_move"}, (bad, a[1:], b[1:])
    py.steps = rs.steps = 400
    a, b = py.step(52), rs.step(52)
    assert a[1:] == b[1:] and a[3] and a[4] == {"error": "timeout"}
    print("illegal moves and timeouts handled identically")


def test_model_evaluation(games=2000):
    """ The trained model's seeded evaluation (evaluate.py) on both engines. """
    path = os.environ.get("CRIBBAGE_MODEL", MODEL_PATH)
    if QUICK or not os.path.exists(path.removesuffix(".zip") + ".zip"):
        print(f"model evaluation skipped ({'--quick' if QUICK else 'no model at ' + path})")
        return
    import evaluate
    from sb3_contrib import MaskablePPO
    model = MaskablePPO.load(path, device="cpu")
    rates = {}
    for name, cls in (("python", CribbageEnv), ("rust", RustCribbageEnv)):
        evaluate.CribbageEnv = cls
        rates[name] = evaluate.evaluate(model, "smart", games)
    p = rates["python"] / 100
    margin = 100 * 1.96 * (p * (1 - p) / games) ** 0.5
    print(f"model vs smart, {games} seeded games: python {rates['python']:.2f}% | "
          f"rust {rates['rust']:.2f}% (95% margin ±{margin:.1f})")
    assert abs(rates["python"] - rates["rust"]) <= margin


if __name__ == "__main__":
    n = 10_000 if QUICK else N_INPUTS
    test_score_hand(n)
    test_peg_points(n)
    test_breakdowns_match_rust_scoring()
    test_expected_best_reply(n)
    test_discard_values()
    test_obs_and_masks_for_same_state(2_000 if QUICK else 20_000)
    test_bad_actions()
    with tempfile.TemporaryDirectory() as tmp:
        net_path = test_opponent_net(tmp)
        test_lockstep_games(net_path)
    test_model_evaluation()
    print("Rust engine matches the Python engine.")

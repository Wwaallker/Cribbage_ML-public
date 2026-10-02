""" Teaches a model's crib discards directly (supervised), instead of waiting for PPO to find them.

PPO learns discards slowly: a good throw's reward is buried under a whole deal of card luck.
Here every one of the 15 ways to split a 6-card hand gets an expected value, and the policy is
trained to prefer the best ones:

    value(discard) = E[kept hand over all 46 starters] +/- E[crib from the 2 thrown cards]
                     (+ when the AI deals and owns the crib, - when the opponent does)

The crib term comes from a table of average crib points for each kind of throw (rank pair,
suited or not), built once by simulation and cached in tools/crib_table.json.

The loss is the expected points the policy gives up: sum over discards of
P(discard) * (best value - its value). Cross-entropy towards the best discard was tried first
and made discards WORSE: the trained policy is very confident, so cross-entropy is dominated by
its confident mistakes and the network learns to hedge rather than to pick better. A loss in
points weighs every mistake by what it actually costs.

Pegging is protected while this trains: on pegging positions the model plays, the new policy is
held to the original's move probabilities (a KL penalty), so the shared layers can't drift.

    python -m tools.discard_teacher models/current models/taught
"""
import argparse
import itertools
import json
import os
import random
import time
from multiprocessing import Pool

import numpy as np
import torch

from cribbage import CribbageEnv
from cribbage.env import N_CARDS, WIN_SCORE

HERE = os.path.dirname(os.path.abspath(__file__))
TABLE_PATH = os.path.join(HERE, "crib_table.json")
TABLE_SAMPLES = 40_000          # simulated cribs per kind of throw


# ----------------------------------------------------------------------
# The crib table
# ----------------------------------------------------------------------
def throw_key(a, b):
    """ What matters about a 2-card throw for the crib: the two ranks and whether suited. """
    ra, rb = sorted((a % 13, b % 13))
    return f"{ra},{rb},{int(a // 13 == b // 13)}"


def _table_entry(args):
    ra, rb, suited, samples, seed = args
    env, rng = CribbageEnv(), random.Random(seed)
    a = ra                                       # rank ra, suit 0
    b = rb + 13 * (0 if suited else 1)           # rank rb, same suit or the next one
    rest = [c for c in range(52) if c not in (a, b)]
    total = 0
    for _ in range(samples):
        x, y, starter = rng.sample(rest, 3)
        total += env.score_hand([a, b, x, y], starter, is_crib=True)
    return f"{ra},{rb},{int(suited)}", total / samples


def crib_table(workers):
    if os.path.exists(TABLE_PATH):
        with open(TABLE_PATH) as f:
            return json.load(f)
    jobs = [(ra, rb, s, TABLE_SAMPLES, 1000 * ra + 10 * rb + s)
            for ra in range(13) for rb in range(ra, 13) for s in (0, 1) if not (s and ra == rb)]
    with Pool(workers) as pool:
        table = dict(pool.map(_table_entry, jobs))
    with open(TABLE_PATH, "w") as f:
        json.dump(table, f, indent=0, sort_keys=True)
    print(f"crib table: {len(table)} kinds of throw, {TABLE_SAMPLES:,} cribs each -> {TABLE_PATH}")
    return table


# ----------------------------------------------------------------------
# Teacher values for one hand
# ----------------------------------------------------------------------
def discard_values(env, six, ai_deals, table):
    """ {action: expected points} for all 15 discards of the AI's current six cards. """
    unseen = [c for c in range(52) if c not in six]
    values = {}
    for pair in itertools.combinations(six, 2):
        keep = [c for c in six if c not in pair]
        hand = sum(env.score_hand(keep, st) for st in unseen) / len(unseen)
        crib = table[throw_key(*pair)]
        values[env.discard_action(list(pair))] = hand + (crib if ai_deals else -crib)
    return values


def _make_discard_examples(args):
    """ Random discard positions with teacher values. Scores are randomised so the examples cover
    every stage of a game, not just the first deal. """
    start, count, table = args
    env, rng = CribbageEnv(), random.Random(start)
    obs, values = [], []
    for i in range(start, start + count):
        env.reset(seed=i)
        env.is_ai_dealer = rng.random() < 0.5
        env.p1_score, env.comp_score = rng.randrange(0, WIN_SCORE - 5), rng.randrange(0, WIN_SCORE - 5)
        vals = discard_values(env, list(env.hand), env.is_ai_dealer, table)
        v = np.full(15, np.nan, dtype=np.float32)
        for a, x in vals.items():
            v[a - N_CARDS] = x
        obs.append(env._get_obs())
        values.append(v)
    return np.array(obs), np.array(values)


def discard_dataset(n, workers, table, seed_base):
    block = -(-n // (workers * 4))
    jobs = [(seed_base + s, min(block, n - s), table) for s in range(0, n, block)]
    with Pool(workers) as pool:
        parts = pool.map(_make_discard_examples, jobs)
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


# ----------------------------------------------------------------------
# Pegging positions (to keep pegging unchanged)
# ----------------------------------------------------------------------
def pegging_positions(model, n, seed):
    env = CribbageEnv(opponent="smart")
    obs, masks, g = [], [], 0
    while len(obs) < n:
        o, _ = env.reset(seed=seed + g)
        g += 1
        done = trunc = False
        while not (done or trunc):
            m = env.action_masks()
            if env.phase == 1:
                obs.append(o)
                masks.append(m)
            a, _ = model.predict(o, action_masks=m, deterministic=True)
            o, _, done, trunc, _ = env.step(int(a))
    return np.array(obs[:n]), np.array(masks[:n])


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------
def discard_logits(policy, obs):
    """ Policy logits for the 15 discard actions only. """
    latent_pi, _ = policy.mlp_extractor(policy.extract_features(obs, policy.pi_features_extractor))
    return policy.action_net(latent_pi)[:, N_CARDS:]


def masked_log_probs(policy, obs, masks):
    latent_pi, _ = policy.mlp_extractor(policy.extract_features(obs, policy.pi_features_extractor))
    logits = policy.action_net(latent_pi).masked_fill(~masks, -1e8)
    return torch.log_softmax(logits, dim=1)


def points_lost(policy, obs, values):
    """ Average expected points given up by the policy's top discard vs the teacher's best. """
    with torch.no_grad():
        pick = discard_logits(policy, obs).argmax(1)
    chosen = values[torch.arange(len(values)), pick]
    best = values.max(1).values
    return float((best - chosen).mean()), float((pick == values.argmax(1)).float().mean())


def teach(model, d_obs, d_val, p_obs, p_masks, epochs, peg_weight, lr, batch):
    policy = model.policy
    orig = {k: v.clone() for k, v in policy.state_dict().items()}
    with torch.no_grad():                                  # the original pegging move probabilities
        p_target = masked_log_probs(policy, p_obs, p_masks).exp()

    n_val = len(d_obs) // 20                               # hold out 5% to measure honestly
    v_obs, v_val = d_obs[:n_val], d_val[:n_val]
    t_obs, t_val = d_obs[n_val:], d_val[n_val:]
    regret = t_val.max(1, keepdim=True).values - t_val     # points lost by each discard

    lost, agree = points_lost(policy, v_obs, v_val)
    print(f"before: {lost:.3f} pts lost per discard (held-out), picks the best {100 * agree:.0f}%")

    opt = torch.optim.Adam(policy.mlp_extractor.policy_net.parameters(), lr=lr)
    opt.add_param_group({"params": policy.action_net.parameters()})
    steps = len(t_obs) // batch
    for ep in range(epochs):
        perm = torch.randperm(len(t_obs))
        peg_perm = torch.randint(0, len(p_obs), (steps, batch))
        tot_d = tot_p = 0.0
        for s in range(steps):
            idx = perm[s * batch:(s + 1) * batch]
            probs = torch.softmax(discard_logits(policy, t_obs[idx]), dim=1)
            d_loss = (probs * regret[idx]).sum(1).mean()          # expected points given up

            j = peg_perm[s]
            new_logp = masked_log_probs(policy, p_obs[j], p_masks[j])
            p_loss = (p_target[j] * (torch.log(p_target[j] + 1e-12) - new_logp)).masked_fill(~p_masks[j], 0).sum(1).mean()

            loss = d_loss + peg_weight * p_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_d += d_loss.item()
            tot_p += p_loss.item()
        lost, agree = points_lost(policy, v_obs, v_val)
        print(f"epoch {ep + 1}: expected pts lost {tot_d / steps:.3f}, pegging drift (KL) {tot_p / steps:.5f} | "
              f"held-out: {lost:.3f} pts lost per discard, picks the best {100 * agree:.0f}%")

    changed = [k for k in orig if not torch.equal(orig[k], policy.state_dict()[k])]
    assert all(k.startswith(("mlp_extractor.policy_net", "action_net")) for k in changed), changed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("src")
    parser.add_argument("dst")
    parser.add_argument("--hands", type=int, default=500_000, help="discard positions to learn from")
    parser.add_argument("--peg-positions", type=int, default=100_000)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--peg-weight", type=float, default=5.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()

    from sb3_contrib import MaskablePPO
    torch.set_num_threads(args.threads)
    t0 = time.time()
    table = crib_table(args.workers)

    cache = os.path.join(os.path.dirname(os.path.abspath(args.dst)), f"teacher_data_{args.hands}.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        d_obs, d_val = z["obs"], z["values"]
    else:
        d_obs, d_val = discard_dataset(args.hands, args.workers, table, seed_base=5_000_000)
        np.savez_compressed(cache, obs=d_obs, values=d_val)
    print(f"{len(d_obs):,} discard positions ({time.time() - t0:.0f}s)")

    model = MaskablePPO.load(args.src, device="cpu")
    p_obs, p_masks = pegging_positions(model, args.peg_positions, seed=7_000_000)
    print(f"{len(p_obs):,} pegging positions ({time.time() - t0:.0f}s)")

    teach(model, torch.as_tensor(d_obs), torch.as_tensor(d_val), torch.as_tensor(p_obs),
          torch.as_tensor(p_masks), args.epochs, args.peg_weight, args.lr, args.batch)
    model.save(args.dst)
    print(f"saved {args.dst} ({time.time() - t0:.0f}s)")

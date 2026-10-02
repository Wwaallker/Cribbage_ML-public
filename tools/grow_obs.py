""" Grows a saved model to a bigger observation, keeping everything it has learned.

New observation blocks go at the END of OBS_BLOCKS, so the old inputs keep their positions.
This copies every weight across and gives the new inputs zero weights in the first layer of
both the policy and value networks, so the grown model plays exactly like the old one until
training teaches it to use them.

    python -m tools.grow_obs models/current models/current_grown
"""
import argparse

import numpy as np
import torch
from sb3_contrib import MaskablePPO

from cribbage import CribbageEnv, OBS_SIZE

FIRST_LAYERS = ("mlp_extractor.policy_net.0.weight", "mlp_extractor.value_net.0.weight")


def grow(src, dst, batch_size=None):
    old = MaskablePPO.load(src, device="cpu")
    n_old = old.observation_space.shape[0]
    if n_old > OBS_SIZE:
        raise SystemExit(f"{src} takes {n_old} inputs, more than the env's {OBS_SIZE}")

    new = MaskablePPO("MlpPolicy", CribbageEnv(), device="cpu", verbose=0,
                      n_steps=old.n_steps, batch_size=batch_size or old.batch_size, n_epochs=old.n_epochs,
                      gamma=old.gamma, gae_lambda=old.gae_lambda, learning_rate=old.learning_rate,
                      ent_coef=old.ent_coef, vf_coef=old.vf_coef, max_grad_norm=old.max_grad_norm,
                      clip_range=old.clip_range(1), policy_kwargs=old.policy_kwargs)

    state = old.policy.state_dict()
    for name in FIRST_LAYERS:
        w = state[name]
        state[name] = torch.cat([w, torch.zeros(w.shape[0], OBS_SIZE - n_old)], dim=1)
    new.policy.load_state_dict(state)
    new.num_timesteps = old.num_timesteps
    new.save(dst)
    print(f"{src}: {n_old} inputs -> {dst}: {OBS_SIZE} inputs ({old.num_timesteps:,} steps kept)")
    return old, new, n_old


def check_same_play(old, new, n_old, games=20):
    """ The grown model must choose the same moves and value positions the same as the old one. """
    env = CribbageEnv()
    worst_p = worst_v = 0.0
    for g in range(games):
        obs, _ = env.reset(seed=g)
        done = trunc = False
        while not (done or trunc):
            mask = env.action_masks()
            with torch.no_grad():
                o_new = torch.as_tensor(obs[None])
                o_old = o_new[:, :n_old]
                d_new = new.policy.get_distribution(o_new, action_masks=mask[None]).distribution.probs
                d_old = old.policy.get_distribution(o_old, action_masks=mask[None]).distribution.probs
                v_new, v_old = new.policy.predict_values(o_new), old.policy.predict_values(o_old)
            worst_p = max(worst_p, float((d_new - d_old).abs().max()))
            worst_v = max(worst_v, float((v_new - v_old).abs().max()))
            a, _ = new.predict(obs, action_masks=mask, deterministic=True)
            obs, _, done, trunc, _ = env.step(int(a))
    print(f"checked {games} games: largest move-probability difference {worst_p:.2e}, value difference {worst_v:.2e}")
    return worst_p < 1e-5 and worst_v < 1e-4


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("src")
    parser.add_argument("dst")
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args()
    old, new, n_old = grow(args.src, args.dst, args.batch_size)
    if not check_same_play(old, new, n_old):
        raise SystemExit("grown model plays differently -- not safe to use")

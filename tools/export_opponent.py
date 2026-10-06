""" Exports a saved model's policy network as an opponent for opponent="pool" (self-play).

The file holds the three layers the policy decides with (w0/b0, w1/b1: tanh layers; w2/b2:
action logits); the value network is left out. Both engines play it the same way
(cribbage.env.OpponentNet, and Core.add_net in rust/src/lib.rs).

    python -m tools.export_opponent models/current models/pool/s12.npz
"""
import argparse
import os

import numpy as np


def export(src, dst):
    """ src: a model path, or a loaded MaskablePPO. """
    if isinstance(src, str):
        from sb3_contrib import MaskablePPO
        src = MaskablePPO.load(src, device="cpu")
    state = src.policy.state_dict()
    layers = ("mlp_extractor.policy_net.0", "mlp_extractor.policy_net.2", "action_net")
    arrays = {}
    for k, name in enumerate(layers):
        arrays[f"w{k}"] = state[f"{name}.weight"].numpy()
        arrays[f"b{k}"] = state[f"{name}.bias"].numpy()
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    np.savez(dst, **arrays)
    return dst


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("src")
    parser.add_argument("dst")
    args = parser.parse_args()
    export(args.src, args.dst)
    print(f"{args.src} -> {args.dst}")

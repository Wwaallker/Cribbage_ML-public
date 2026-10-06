import os

from cribbage.env import DISCARD_PAIRS, N_ACTIONS, OBS_SIZE

# CRIBBAGE_ENGINE=rust plays the games in the Rust engine (rust/, see cribbage/rust_env.py).
# Python is the default and the reference.
ENGINE = os.environ.get("CRIBBAGE_ENGINE", "python").lower()
if ENGINE == "rust":
    from cribbage.rust_env import RustCribbageEnv as CribbageEnv
elif ENGINE == "python":
    from cribbage.env import CribbageEnv
else:
    raise ValueError(f"CRIBBAGE_ENGINE must be 'python' or 'rust', not {ENGINE!r}")

__all__ = ["CribbageEnv", "DISCARD_PAIRS", "N_ACTIONS", "OBS_SIZE", "ENGINE"]

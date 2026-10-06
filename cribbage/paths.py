import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "current")            # SB3 adds .zip
RELEASE_MODEL_PATH = os.path.join(MODELS_DIR, "release", "cribbage_ai")   # shipped in the public repo
STATE_PATH = os.path.join(MODELS_DIR, "training_state.json")
LOG_DIR = os.path.join(MODELS_DIR, "logs")
POOL_DIR = os.path.join(MODELS_DIR, "pool")                # opponent networks for self-play (.npz)
BASELINE_NET = os.path.join(POOL_DIR, "b54.npz")           # frozen 54% model: the self-play yardstick
CSV_PATH = os.path.join(LOG_DIR, "progress.csv")
REPLAY_DIR = os.path.join(ROOT, "replays")                 # one <SEED KEY>.json per game played
# The Obsidian vault training writes a note into after every session (tools/session_note.py).
VAULT_DIR = os.environ.get("CRIBBAGE_VAULT", os.path.join(os.path.dirname(ROOT), "Cribbage_ML"))

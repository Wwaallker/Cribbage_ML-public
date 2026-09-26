import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "current")            # SB3 adds .zip
RELEASE_MODEL_PATH = os.path.join(MODELS_DIR, "release", "cribbage_ai")   # shipped in the public repo
STATE_PATH = os.path.join(MODELS_DIR, "training_state.json")
LOG_DIR = os.path.join(MODELS_DIR, "logs")
CSV_PATH = os.path.join(LOG_DIR, "progress.csv")

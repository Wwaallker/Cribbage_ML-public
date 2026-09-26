# Cribbage_ML
Attempts at making a decent ML algorithm for the card game Cribbage.

A MaskablePPO agent (sb3-contrib) learns full two-player cribbage to 121 against a
heuristic opponent (smart discards + 2-ply pegging lookahead).

## Play against it
You need Python 3.11 or 3.12 and git. A trained model ships in `models/release/`.
```
git clone https://github.com/Wwaallker/Cribbage_ML-public.git
cd Cribbage_ML-public
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # optional: much smaller download
pip install -r requirements.txt
python hud.py                        # then [1] Play vs AI
```
- Type cards as rank + suit: `5H`, `TC` or `10D`, `QS`. Discard two at once: `5H TC`.
- `tab` shows or hides the AI's hand, `menu` goes back, `quit` exits.
- Make the terminal tall (about 40 lines) so the whole table fits.
- On Windows, use Windows Terminal (the old console can't draw the colors).

## Layout
```
cribbage/          game package
  env.py           CribbageEnv: rules, scoring, heuristic opponent, observation/actions
  ui.py            shared terminal drawing (card boxes, colors, bars)
  board.py         ASCII cribbage board + end-of-deal scoring replay
  paths.py         where models and logs live
train.py           train in timed sessions, resumes models/current.zip, evaluates at the end
evaluate.py        win % of any saved model over 2000 fixed, seeded games
metrics.py         win/skunk rates, points by source, discard quality -> models/metrics.json
play.py            play against the trained AI in the terminal
hud.py             menu + live training dashboard
monitor.py         plotext charts of the training log
tests/             env sanity, discard heuristic, pegging heuristic head-to-head
models/            current.zip, training_state.json, logs/ (ignored), archive/,
                   release/cribbage_ai.zip (the model play.py falls back to)
```

## Usage
Install with `pip install -r requirements.txt`, then run everything from the repo root:
```
python train.py --hours 2       # train (starts fresh if models/current.zip is missing)
python evaluate.py              # evaluate models/current (or pass a model path)
python metrics.py               # detailed metrics for models/current
python play.py                  # play vs the AI
python -m cribbage.board        # demo of the board replay animation
python hud.py                   # menu + live dashboard while training
python -m tests.test_env        # likewise tests.test_discard, tests.test_pegging
```

## Actions and observation
- Actions 0-51 play that card while pegging; 52-66 discard one of the 15 pairs of hand slots
  (hand is kept sorted by rank, then suit), so the crib discard is a single decision.
- The observation (456 values) is described block by block in `OBS_BLOCKS` in `cribbage/env.py`.

`models/archive/v1_680M/` holds the previous model (193-value observation, one-card discards),
which plateaued at ~39% vs the smart opponent. It is not loadable by the current env.

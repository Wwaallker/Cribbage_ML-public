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
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu   # Windows/Linux; skip on Mac
pip install -r requirements.txt
python hud.py                        # then [1] Play vs AI
```
Install the CPU-only PyTorch first, as above. A plain `pip install -r requirements.txt`
pulls the GPU build, which brings about 5 GB of NVIDIA libraries the game never uses
(about 7 GB installed in total, versus about 1 GB). On a Mac, PyTorch is CPU-only anyway,
so skip that line.
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
tools/             grow_obs.py (grow a model to a bigger observation), publish_public.sh
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
- The observation (560 values) is described block by block in `OBS_BLOCKS` in `cribbage/env.py`.
  The last two blocks help with pegging: for each card the AI could play, the points it scores
  now and the expected points of the opponent's best reply (the same 2-ply lookahead the
  heuristic opponent uses).
- New observation blocks go at the end. `python -m tools.grow_obs <old> <new>` then grows a saved
  model to fit, with zero weights on the new inputs, so it keeps everything it has learned.

## Models
| | Observation | Steps | vs smart | Notes |
|---|---|---|---|---|
| `archive/v1_680M/` | 193 | 680M | ~39% | one-card discards; not loadable by the current env |
| `archive/v2_168M/` | 456 | 168M | 34.7% | pair discards; grown into the current model |
| `current.zip` | 560 | 1.25B | 43.4% | v2 plus the pegging inputs, batch size 1024 |

## Roadmap
- **Discard teacher** (`tools/discard_teacher.py`, in progress). Rates all 15 discards of a hand
  (kept hand over every starter, plus or minus a crib table) and trains the policy to give up as
  few expected points as possible, while holding its pegging fixed. On the 377M model
  (500k hands, 40 epochs), same 1,000 games vs smart: discards 0.72 -> 0.51 pts lost/deal
  (bot 0.39), best discard 53% -> 60%, pegging unchanged, average margin -3.2 -> -1.5,
  win 43.0% -> 46.0% (±3.1). Cross-entropy towards the best discard made discards worse.
- **Self-play.** Train against a pool of earlier versions of itself mixed with the heuristic,
  so it can end up stronger than the heuristic rather than only learning to handle it.
- **Per-deal reward.** gamma 0.999 over a ~47-move game spreads credit for a discard thinly.

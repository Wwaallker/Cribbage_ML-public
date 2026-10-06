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
- Each game has a seed key (shown at the top, e.g. `K7QM2XPA`). Type a key at the start to get
  the same deals again. Every move is saved to `replays/<KEY>.json`, and
  `python -m tools.replay_game replays/<KEY>.json --upto <move>` rebuilds the position just
  before that move, with every score and its reason: handy for reporting a bug.

## Layout
```
cribbage/          game package
  env.py           CribbageEnv: rules, scoring, heuristic opponent, observation/actions
  rust_env.py      the same env with the game engine in Rust (optional, see below)
  ui.py            shared terminal drawing (card boxes, colors, bars)
  board.py         ASCII cribbage board + end-of-deal scoring replay
  paths.py         where models and logs live
  seeds.py         seed keys (K7QM2XPA) for repeatable games
train.py           train in timed sessions, resumes models/current.zip, evaluates at the end
evaluate.py        win % of any saved model over 2000 fixed, seeded games
metrics.py         win/skunk rates, points by source, discard quality -> models/metrics.json
rust/              the Rust game engine (PyO3 extension module cribbage_rs)
tools/             grow_obs.py (grow a model to a bigger observation), discard_teacher.py,
                   export_opponent.py (a model as a self-play opponent), replay_game.py,
                   bench_engine.py, publish_public.sh
play.py            play against the trained AI in the terminal
hud.py             menu + live training dashboard
monitor.py         plotext charts of the training log
tests/             env sanity, discard heuristic, pegging heuristic head-to-head, Rust vs Python engine
models/            current.zip, training_state.json, logs/ (ignored), archive/,
                   release/cribbage_ai.zip (the model play.py falls back to)
```

## Usage
Install with `pip install -r requirements.txt`, then run everything from the repo root:
```
python train.py --hours 2       # train (starts fresh if models/current.zip is missing)
python evaluate.py              # evaluate models/current (or pass a model path)
python evaluate.py --vs models/pool/b54.npz   # ... against the frozen 54% model instead
python train.py --hours 6 --opponent pool     # self-play, see Roadmap
python metrics.py               # detailed metrics for models/current
python play.py                  # play vs the AI
python -m cribbage.board        # demo of the board replay animation
python hud.py                   # menu + live dashboard while training
python -m tests.test_env        # likewise tests.test_discard, tests.test_pegging
```

## Rust engine (optional, faster training)
`rust/` holds the game engine of `cribbage/env.py` (rules, scoring, the heuristic opponent,
observation and masks) in Rust. It plays exactly the same games: it carries a copy of
Python's random number generator, so a seeded game goes move for move as on the Python engine,
and `tests/test_rust_engine.py` checks that. Python stays the default and the reference: change
the rules in `env.py` first, then in `rust/src/`.
```
pip install maturin
(cd rust && maturin develop --release)      # needs cargo / rustc 1.74+
CRIBBAGE_ENGINE=rust python train.py --hours 2  # likewise evaluate.py, metrics.py
python -m tests.test_rust_engine            # after any change to either engine
python -m tools.bench_engine                # speed of both engines
```
`play.py` always uses the Python engine, since it swaps the computer's moves for yours.

## Actions and observation
- Actions 0-51 play that card while pegging; 52-66 discard one of the 15 pairs of hand slots
  (hand is kept sorted by rank, then suit), so the crib discard is a single decision.
- The observation (575 values) is described block by block in `OBS_BLOCKS` in `cribbage/env.py`.
  `peg_now` and `peg_reply` help with pegging: for each card the AI could play, the points it
  scores now and the expected points of the opponent's best reply (the same 2-ply lookahead the
  heuristic opponent uses). `discard_value` helps with the crib discard: for each of the 15
  discards, how close it comes to the best one (kept hand over every starter, plus or minus a
  crib table, `cribbage/crib_table.json`).
- New observation blocks go at the end. `python -m tools.grow_obs <old> <new>` then grows a saved
  model to fit, with zero weights on the new inputs, so it keeps everything it has learned.

## Models
| | Observation | Steps | vs smart | Notes |
|---|---|---|---|---|
| `archive/v1_680M/` | 193 | 680M | ~39% | one-card discards; not loadable by the current env |
| `archive/v2_168M/` | 456 | 168M | 34.7% | pair discards; grown into the current model |
| `snapshots/d1005_obs560_taught.zip` | 560 | 1.25B | 46.6% | v2 plus the pegging inputs, batch size 1024; discard teacher |
| `current.zip` | 575 | 1.26B | 53.0% | the 560 model plus the discard inputs, discard teacher (--soft-ce) |

## Roadmap
- **Discard teacher** (`tools/discard_teacher.py`). Rates all 15 discards of a hand (kept hand
  over every starter, plus or minus a crib table) and trains the policy towards the best ones,
  while holding its pegging fixed. With the `discard_value` inputs and `--soft-ce 0.25`
  (40 epochs) on the 1.25B model, 2000 games vs smart: discards 0.53 -> 0.02 pts lost/deal
  (bot 0.38), win 46.6% -> 54.0%. PPO loosens the discards again (0.09 after 30 minutes), so
  re-run the teacher after long sessions.
- **Self-play** (`train.py --opponent pool`, in progress). Half the games against the heuristic
  (`--p-smart`), half against a network from `models/pool/`: the frozen 54% model (`b54.npz`)
  and the newest 8 snapshots of itself, one added every 20M steps (`--pool-every`). The computer
  seat sees the same 575 inputs from its own side (`_get_obs(seat=1)`); the networks run inside
  the Rust engine. A model plays its own copy to 49.9% over 10,000 games. Each session reports
  win % vs smart and vs `b54.npz`.
- **Per-deal reward.** gamma 0.999 over a ~47-move game spreads credit for a discard thinly.

import argparse
from sb3_contrib import MaskablePPO
from cribbage import CribbageEnv
from cribbage.paths import MODEL_PATH

EVAL_GAMES = 2000
EVAL_SEED = 1_000_000     # far from anything training uses; every model sees the same deals


def evaluate(model, opponent="smart", games=EVAL_GAMES, seed=EVAL_SEED):
    """ Win % of the model over a fixed, seeded set of games. ~±1.1% noise at 2000 games. """
    env = CribbageEnv(opponent=opponent)
    wins = 0
    for g in range(games):
        obs, _ = env.reset(seed=seed + g)
        done = trunc = False
        while not (done or trunc):
            a, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=True)
            obs, r, done, trunc, _ = env.step(int(a))
        wins += env.winner == 0
    return 100.0 * wins / games


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a saved model on a fixed set of games.")
    parser.add_argument("model", nargs="?", default=MODEL_PATH, help="path to a model (default: models/current)")
    parser.add_argument("--games", type=int, default=EVAL_GAMES)
    parser.add_argument("--opponent", default="smart", choices=["smart", "random", "mixed"])
    args = parser.parse_args()

    model = MaskablePPO.load(args.model, device="cpu")
    winrate = evaluate(model, args.opponent, args.games)
    print(f"{args.model} vs {args.opponent}, {args.games} seeded games: {winrate:.1f}%")

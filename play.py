import os
import sys
from sb3_contrib import MaskablePPO
from cribbage import CribbageEnv
from cribbage.board import replay, event_text
from cribbage.env import hand_sort_key
from cribbage.paths import MODEL_PATH, RELEASE_MODEL_PATH
from cribbage.ui import (clear_screen, BANNER, RED, GREEN, YELLOW, DIM, BOLD, RESET, card_str, parse_card_code,
                         score_bar, card_box_lines, hand_row_lines, print_hand_row, side_by_side,
                         section, rule)


class ReturnToMenu(Exception):
    pass


def print_board(env, state, header="", error=""):
    """ One screen, sized to fit a normal terminal: no big banner, the starter next to
    the table, and this deal's scores next to the AI's hand. """
    clear_screen()
    dealer = "YOU" if not env.is_ai_dealer else "AI"
    print(f" {RED}{BOLD}▓▒░ DEAL {env.deals}  //  {'DISCARD' if env.phase == 0 else 'PEGGING'} ░▒▓{RESET}"
          f"   {DIM}dealer: {dealer}{RESET}")
    rule()
    print(f" AI    {score_bar(env.p1_score, color=RED)}  {env.p1_score:>3}/121")
    print(f" YOU   {score_bar(env.comp_score)}  {env.comp_score:>3}/121")
    print()

    # the starter is cut after the discard; until then it sits face down on the deck
    cut = env.starter_card != -1
    starter = hand_row_lines("STARTER", [env.starter_card if cut else 0], hidden=not cut,
                             label_color=YELLOW + BOLD, border=YELLOW)
    if env.phase == 1:
        table = hand_row_lines(f"TABLE   COUNT: {BOLD}{env.current_peg_sum}{RESET}",
                               env.pegging_table, label_color=DIM)
        print("\n".join(side_by_side(starter, table)))
    else:
        print("\n".join(starter))
    print()

    ai = hand_row_lines("AI HAND", env.hand, hidden=not state["show_ai_hand"], label_color=RED)
    scores = [f" {YELLOW}SCORES THIS DEAL{RESET}"]
    scores += [event_text(*e) for e in state["events"][-5:]] or [f" {DIM}(none yet){RESET}"]
    print("\n".join(side_by_side(ai, scores)))
    print()
    print("\n".join(hand_row_lines("YOUR HAND", sorted(env.comp_hand, key=hand_sort_key),
                                   label_color=GREEN, codes=True)))
    print()
    if state.get("last_ai"):
        print(f" {RED}AI played: {state['last_ai']}{RESET}")
    if error:
        print(f" {RED}{BOLD}{error}{RESET}")
    if header:
        print(f" {YELLOW}{header}{RESET}")
    print(f"{DIM} ('tab' toggles AI hand visibility, 'menu' to return, 'quit' to exit){RESET}")


def read_command(prompt, state):
    """ Handles tab / menu / quit; returns the raw text otherwise, or None to redraw. """
    raw = input(f"{YELLOW}{prompt}{RESET}").strip()
    low = raw.lower()
    if low == "tab":
        state["show_ai_hand"] = not state["show_ai_hand"]
        return None
    if low == "quit":
        clear_screen(); sys.exit(0)
    if low == "menu":
        raise ReturnToMenu()
    return raw


def human_discard(env, state):
    """ Mistakes are shown on the redrawn screen, so the very next input is a real try. """
    error = ""
    while True:
        print_board(env, state, "YOUR DISCARD", error)
        raw = read_command("Discard 2 cards to crib (e.g. 5H TC): ", state)
        if raw is None:
            error = ""
            continue
        codes = raw.replace(",", " ").split()
        if len(codes) != 2:
            error = f"Enter exactly 2 cards (you entered {len(codes)})."
            continue
        idxs = [parse_card_code(c, env.comp_hand) for c in codes]
        if None in idxs or idxs[0] == idxs[1]:
            error = f"Invalid or duplicate card(s): {raw}"
            continue
        return idxs


def human_pick(env, legal, state):
    error = ""
    while True:
        print_board(env, state, f"YOUR PLAY  (legal: {' '.join(card_str(c) for c in legal)})", error)
        raw = read_command("Play a card: ", state)
        if raw is None:
            error = ""
            continue
        idx = parse_card_code(raw, legal)
        if idx is None:
            error = f"Can't play {raw!r}."
            continue
        state["last_ai"] = None
        return idx


def deal_counts(env):
    """ (label, cards, points, color, player, breakdown) for each count, in counting order:
    non-dealer's hand, dealer's hand, dealer's crib. player 0 = AI, 1 = you. """
    starter = env.starter_card

    def count(label, cards, color, player, is_crib=False):
        return (label, cards, env.score_hand(cards, starter, is_crib=is_crib), color, player,
                env.hand_breakdown(cards, starter, is_crib=is_crib))

    ai = count("AI HAND", env.ai_scoring_hand, RED, 0)
    you = count("YOUR HAND", env.comp_scoring_hand, GREEN, 1)
    if env.is_ai_dealer:
        return [you, ai, count("AI'S CRIB", env.crib, RED, 0, is_crib=True)]
    return [ai, you, count("YOUR CRIB", env.crib, GREEN, 1, is_crib=True)]


def counts_applied(counts, ai_score, your_score):
    """ How many counts get pegged, starting from these scores, before someone reaches 121. """
    scores = [ai_score, your_score]
    for n, (_, _, pts, _, player, _) in enumerate(counts):
        if max(scores) >= 121:
            return n
        scores[player] += pts
    return len(counts)


def show_deal_review(env, counts, applied):
    """ End-of-deal screen: starter, both hands and the crib, each with its count
    itemised (15s, pairs, runs, flush, nobs). """
    clear_screen()
    print(BANNER)
    dealer = "AI" if env.is_ai_dealer else "YOU"
    section(f"END OF DEAL {env.deals}")
    print(f" {DIM}dealer: {dealer}{RESET}")
    rule()

    print(f" {YELLOW}{BOLD}STARTER{RESET}")
    for line in card_box_lines(env.starter_card, border=YELLOW):
        print(f" {line}")
    print()

    for n, (label, cards, pts, color, _, items) in enumerate(counts):
        print_hand_row(f"{n + 1}. {label}   {BOLD}{pts} pts{RESET}",
                       sorted(cards, key=hand_sort_key), label_color=color)
        detail = " · ".join(f"{name} {p}" for name, p in items) or "nineteen (0)"
        if n >= applied:
            detail += "  (not counted -- game already over)"
        print(f" {DIM}{detail}{RESET}")
        print()

    print(f" AI    {score_bar(env.p1_score, color=RED)}  {min(env.p1_score, 121):>3}/121")
    print(f" YOU   {score_bar(env.comp_score)}  {min(env.comp_score, 121):>3}/121")
    print()
    input(f"{DIM}(press enter to continue){RESET}")


def result_lines(env):
    """ Win/loss line, with skunk (loser at 90 or less) or double skunk (60 or less). """
    ai, you = min(env.p1_score, 121), min(env.comp_score, 121)     # pegging stops at 121
    if env.winner == 0:
        line = f" {RED}{BOLD}AI WINS, {ai}-{you}{RESET}"
    else:
        line = f" {GREEN}{BOLD}YOU WIN, {you}-{ai}{RESET}"
    loser = min(ai, you)
    if loser <= 60:
        line += f"   {RED}{BOLD}DOUBLE SKUNK{RESET}"
    elif loser <= 90:
        line += f"   {YELLOW}{BOLD}SKUNK{RESET}"
    return ["", line]


def record_scores(env, state):
    """ Wraps env._award so every score is logged as (player, pts, reason) for the board replay. """
    award = env._award

    def award_and_record(player, pts):
        if state["count_labels"]:                       # hand / crib counting, in order
            reason = state["count_labels"].pop(0)
        elif env.phase == 0:                            # starter turned before pegging begins
            reason = "his heels"
        elif pts == 1:                                  # plays always score 0 or 2+
            reason = "last card" if not env.hand and not env.comp_hand else "go"
        else:
            reason = ", ".join(name for name, _ in env.peg_breakdown(env.pegging_table))
        if pts and not env.game_over:
            state["events"].append((player, pts, reason))
        return award(player, pts)
    env._award = award_and_record


def replay_deal(env, state, title):
    """ The board replay, with the game result under it if the game is over. """
    footer = result_lines(env) if env.game_over else ()
    replay(state["pegs"], state["events"], title=title, footer=footer)
    state["events"] = []
    state["replayed_end"] = env.game_over
    input(f"{DIM}(press enter to continue){RESET}")


def play_game():
    """ One game of you vs the trained AI. Raises ReturnToMenu if the player types 'menu'. """
    clear_screen()
    # your own training run's model if there is one, else the one shipped with the repo
    model_path = next((p for p in (MODEL_PATH, RELEASE_MODEL_PATH) if os.path.exists(p + ".zip")), None)
    if model_path is None:
        print(f"{RED}No trained model found at models/current.zip or models/release/cribbage_ai.zip{RESET}")
        print(f"{DIM}Train one first (python train.py), then come back.{RESET}\n")
        input(f"{DIM}(press enter to return){RESET}")
        return

    print(f"{DIM}Loading model...{RESET}")
    model = MaskablePPO.load(model_path, device="cpu")

    env = CribbageEnv(opponent="smart")   # computer brain is replaced by you, below
    obs, _ = env.reset()

    state = {"show_ai_hand": True, "last_ai": None,
             "pegs": {0: (0, 0), 1: (0, 0)},      # (front, back) peg per player
             "events": [], "count_labels": [], "replayed_end": False}
    env._comp_discard = lambda: human_discard(env, state)
    env._comp_pick = lambda legal: human_pick(env, legal, state)

    count_hands = env._count_hands

    def count_and_replay():
        """ Counts hands as usual, then shows the review and the replay before the next deal. """
        before = (env.p1_score, env.comp_score)
        counts = deal_counts(env)
        state["count_labels"] = ["crib" if "CRIB" in label else "hand" for label, *_ in counts]
        reward = count_hands()
        state["count_labels"] = []                      # counting may stop early at 121
        show_deal_review(env, counts, counts_applied(counts, *before))
        replay_deal(env, state, f"DEAL {env.deals} // REPLAY")
        return reward
    env._count_hands = count_and_replay
    record_scores(env, state)

    done = trunc = False
    while not (done or trunc):
        action, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=True)
        if env.phase == 1:
            state["last_ai"] = card_str(int(action))     # shown on the board at your next prompt
        obs, r, done, trunc, info = env.step(int(action))

    if not state["replayed_end"]:                       # game ended before the counts
        replay_deal(env, state, f"DEAL {env.deals} // FINAL")


if __name__ == "__main__":
    try:
        play_game()
    except ReturnToMenu:
        pass

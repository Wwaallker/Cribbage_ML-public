""" ASCII cribbage board with two-peg tracks, and a replay animation of a deal's scoring.

Layout, like a real board: street 1 runs out along holes 1-60, street 2 comes back
along 61-120 (right to left), and the far-left slot of street 2 is the 121 finish hole.

            5    10    15    20    25    30    35    40    45    50    55    60
 AI   ◘ ····· ····· ····· ····· ····· ····· ····· ····· ····· ····· ····· ·····║   95
 YOU  ◘ ····· ····· ····· ····· ····· ····· ····· ····· ····· ····○ ····· ··●··║   58
      ═════════════════════════════════════S═══════════════════════════════════SS
 AI   ◆ ····· ····· ····· ····· ····· ●····│·○··· ····· ····· ····· ····· ·····║
 YOU  ◆ ····· ····· ····· ····· ····· ·····│····· ····· ····· ····· ····· ·····║
        120   115   110   105   100   95    90    85    80    75    70    65

The skunk lines: finishing with 90 or less (behind the │ S line) is a skunk,
60 or less (still on the first street, behind the ║ SS line) a double skunk.
"""
import sys
import time

from cribbage.ui import BANNER, RED, GREEN, YELLOW, DIM, BOLD, RESET, BRIGHT_RED

BRIGHT_GREEN = "\033[92m"
HOME_CLEAR = "\033[H\033[J"     # redraw in place without spawning `clear` every frame

HOLE_DELAY = 0.035              # seconds per hole a peg travels
EVENT_PAUSE = 0.7               # pause after each score lands
LABEL_W = 6                     # "  AI  " / "  YOU "
PLAYERS = {0: ("AI", RED, BRIGHT_RED), 1: ("YOU", GREEN, BRIGHT_GREEN)}


def _street_holes(street):
    """ Holes of a street in left-to-right screen order, first slot being start (0) or finish (121). """
    if street == 0:
        return [0] + list(range(1, 61))
    return [121] + list(range(120, 60, -1))


def _hole_char(hole, front, back, color, bright):
    if hole == front:
        return f"{bright}{BOLD}●{RESET}"
    if hole == back:
        return f"{color}○{RESET}"
    if hole == 0:
        return f"{DIM}◘{RESET}"
    if hole == 121:
        return f"{YELLOW}◆{RESET}"
    return f"{DIM}·{RESET}"


def _separator(street, hole):
    """ What sits after a group of 5 holes: a gap, or a skunk line. """
    if hole in (60, 61):                        # end of either street: double skunk line
        return f"{BRIGHT_RED}{BOLD}║{RESET}"
    if street == 1 and hole == 91:              # between 91 and 90: skunk line
        return f"{YELLOW}│{RESET}"
    return " "


def _track_row(player, street, pegs):
    name, color, bright = PLAYERS[player]
    front, back = pegs[player]
    holes = _street_holes(street)
    cells = [_hole_char(holes[0], front, back, color, bright), " "]
    for i, h in enumerate(holes[1:]):
        cells.append(_hole_char(h, front, back, color, bright))
        if i % 5 == 4:
            cells.append(_separator(street, h))
    score = f"  {bright}{BOLD}{front:>3}{RESET}" if street == 0 else ""
    return f" {color}{name:<{LABEL_W - 1}}{RESET}" + "".join(cells) + score


def _scale(street):
    """ Hole numbers every 5 holes, aligned over/under the last hole of each group of 5. """
    width = LABEL_W + 2 + 12 * 6
    line = [" "] * width
    for g in range(12):
        num = str(5 * (g + 1)) if street == 0 else str(120 - 5 * g)
        end = LABEL_W + 2 + g * 6 + 4          # column of the 5th hole in group g
        start = end - len(num) + 1 if street == 0 else LABEL_W + 2 + g * 6
        for k, ch in enumerate(num):
            line[start + k] = ch
    return f"{DIM}{''.join(line).rstrip()}{RESET}"


def _divider():
    """ Line between the streets, labelling the skunk (S) and double skunk (SS) lines. """
    skunk = 2 + 5 * 6 + 5                       # column of the separator between 91 and 90
    double = 2 + 11 * 6 + 5                     # column of the ║ at the far right
    return (f" {' ' * (LABEL_W - 1)}{RED}{DIM}{'═' * skunk}{RESET}{YELLOW}{BOLD}S{RESET}"
            f"{RED}{DIM}{'═' * (double - skunk - 1)}{RESET}{BRIGHT_RED}{BOLD}SS{RESET}")


def render_board(pegs, banner="", log=(), title="THE BOARD", footer=()):
    """ Returns the full frame as one string. pegs: {player: (front, back)}; front = score.
    footer: lines under the log (e.g. the game result). """
    out = [BANNER]
    out.append(f" {RED}{BOLD}▓▒░ {title} ░▒▓{RESET}")
    out.append(f" {RED}{DIM}{'━' * 68}{RESET}")
    out.append(_scale(0))
    out.append(_track_row(0, 0, pegs))
    out.append(_track_row(1, 0, pegs))
    out.append(_divider())
    out.append(_track_row(0, 1, pegs))
    out.append(_track_row(1, 1, pegs))
    out.append(_scale(1))
    out.append("")
    out.append(banner if banner else "")
    out.append("")
    for line in log[-8:]:
        out.append(line)
    out.extend(footer)
    return "\n".join(out)


def event_text(player, pts, reason):
    name, color, bright = PLAYERS[player]
    return f" {bright}{BOLD}{name:<4} +{pts:<3}{RESET}{color}{reason}{RESET}"


def replay(pegs, events, title="THE BOARD", footer=()):
    """ Animates each (player, pts, reason) event: the back peg leapfrogs the front one,
    hole by hole, with a "+pts reason" banner. Mutates pegs to the end state.
    footer is added to the last frame only (e.g. the game result). """
    log = []
    sys.stdout.write(HOME_CLEAR + render_board(pegs, title=title) + "\n")
    sys.stdout.flush()
    time.sleep(EVENT_PAUSE)
    for player, pts, reason in events:
        front, _ = pegs[player]
        target = min(front + pts, 121)
        banner = event_text(player, pts, reason)
        for hole in range(front + 1, target + 1):
            pegs[player] = (hole, front)        # old front peg stays behind as the back peg
            sys.stdout.write(HOME_CLEAR + render_board(pegs, banner, log, title) + "\n")
            sys.stdout.flush()
            time.sleep(HOLE_DELAY)
        log.append(banner)
        sys.stdout.write(HOME_CLEAR + render_board(pegs, banner, log, title) + "\n")
        sys.stdout.flush()
        time.sleep(EVENT_PAUSE)
        if target == 121:
            break
    if footer:
        sys.stdout.write(HOME_CLEAR + render_board(pegs, "", log, title, footer) + "\n")
        sys.stdout.flush()
    return log


if __name__ == "__main__":
    # Demo: python -m cribbage.board  -- replays a made-up deal so the animation can be judged.
    pegs = {0: (52, 47), 1: (58, 50)}
    demo = [(1, 2, "fifteen"), (0, 2, "pair"), (1, 3, "run of 3"), (0, 1, "go"),
            (1, 2, "thirty-one"), (0, 1, "last card"), (1, 8, "hand"), (0, 12, "hand"), (0, 6, "crib")]
    replay(pegs, demo, title="DEMO // REPLAY")

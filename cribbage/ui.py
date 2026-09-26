""" Terminal drawing helpers shared by play.py and hud.py. """
import os
import re
import sys

if os.name == "nt":
    os.system("")                                   # turns on ANSI colors in the Windows console
    sys.stdout.reconfigure(encoding="utf-8")        # card borders and suits are not in cp1252


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


RED, GREEN, YELLOW, CYAN, DIM, BOLD, RESET = (
    "\033[31m", "\033[32m", "\033[33m", "\033[36m", "\033[2m", "\033[1m", "\033[0m"
)
BRIGHT_RED, BRIGHT_WHITE, BRIGHT_YELLOW = "\033[91m", "\033[97m", "\033[93m"

BANNER = f"""{RED}{BOLD}
 ██████╗██████╗ ██╗██████╗ ██████╗  █████╗  ██████╗ ███████╗
██╔════╝██╔══██╗██║██╔══██╗██╔══██╗██╔══██╗██╔════╝ ██╔════╝
██║     ██████╔╝██║██████╔╝██████╔╝███████║██║  ███╗█████╗
██║     ██╔══██╗██║██╔══██╗██╔══██╗██╔══██║██║   ██║██╔══╝
╚██████╗██║  ██║██║██████╔╝██████╔╝██║  ██║╚██████╔╝███████╗
 ╚═════╝╚═╝  ╚═╝╚═╝╚═════╝ ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝
{RESET}"""

RANKS = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
SUITS = ['S', 'H', 'D', 'C']
SUIT_GLYPH = {'S': '♠', 'H': '♥', 'D': '♦', 'C': '♣'}
SUIT_COLOR = {'S': BRIGHT_WHITE, 'C': BRIGHT_WHITE, 'H': BRIGHT_RED, 'D': BRIGHT_RED}
CARD_WIDTH = 7          # characters per card, border included
CARD_HEIGHT = 5


def card_str(idx):
    r, s = idx % 13, idx // 13
    return f"{RANKS[r]}{SUITS[s]}"


def parse_card_code(code, legal_indices):
    code = code.strip().upper()
    if code.startswith("10"):
        rank_str, suit_str = "10", code[2:]
    elif code.startswith("T"):
        rank_str, suit_str = "10", code[1:]
    else:
        rank_str, suit_str = code[:-1], code[-1:]
    if rank_str not in RANKS or suit_str not in SUITS:
        return None
    idx = SUITS.index(suit_str) * 13 + RANKS.index(rank_str)
    return idx if idx in legal_indices else None


def bar(value, lo, hi, width=30, color=GREEN):
    frac = 0.0 if hi == lo else max(0.0, min(1.0, (value - lo) / (hi - lo)))
    filled = int(frac * width)
    return f"{color}{'█' * filled}{DIM}{'░' * (width - filled)}{RESET}"


def score_bar(score, width=30, color=GREEN):
    return bar(score, 0, 121, width, color)


def rule(width=68):
    """ Heavy dark-red divider line. """
    print(f" {RED}{DIM}{'━' * width}{RESET}")


def card_box_lines(idx, hidden=False, border=DIM):
    """ Returns the 5 text lines of one heavy-bordered card, 7 characters wide:

        ┏━━━━━┓
        ┃10   ┃     rank in both corners, bold
        ┃  ♥  ┃     suit in the middle, bright red / bright white
        ┃   10┃
        ┗━━━━━┛

    Padding is done on plain text BEFORE color codes are added, so
    alignment can't be thrown off by invisible ANSI escape sequences. """
    top, bottom = f"{border}┏━━━━━┓{RESET}", f"{border}┗━━━━━┛{RESET}"
    side = f"{border}┃{RESET}"
    if hidden:
        back = f"{side}{RED}{DIM}▒▒▒▒▒{RESET}{side}"
        return [top, back, back, back, bottom]
    rank = RANKS[idx % 13]
    suit = SUITS[idx // 13]
    color = SUIT_COLOR[suit]
    return [
        top,
        f"{side}{BOLD}{color}{rank.ljust(5)}{RESET}{side}",
        f"{side}{color}{SUIT_GLYPH[suit].center(5)}{RESET}{side}",
        f"{side}{BOLD}{color}{rank.rjust(5)}{RESET}{side}",
        bottom,
    ]


def section(title, color=RED):
    """ DOOM-style section header:  ▓▒░ TITLE ░▒▓ """
    print(f" {color}{BOLD}▓▒░ {title} ░▒▓{RESET}")


def hand_row_lines(label, indices, hidden=False, label_color=RESET, codes=False, border=DIM):
    """ The lines of a labelled row of bordered cards side by side, DOS-style.
    codes=True adds each card's typing code (e.g. 10H) centered under it. """
    if not indices:
        return [f" {label_color}{label}{RESET}  {DIM}(empty){RESET}"]
    boxes = [card_box_lines(i, hidden=hidden, border=border) for i in indices]
    lines = [f" {label_color}{label}{RESET}"]
    for row in range(CARD_HEIGHT):
        lines.append(" " + " ".join(box[row] for box in boxes))
    if codes and not hidden:
        lines.append(" " + " ".join(f"{YELLOW}{card_str(i).center(CARD_WIDTH)}{RESET}" for i in indices))
    return lines


def print_hand_row(label, indices, hidden=False, label_color=RESET, codes=False):
    print("\n".join(hand_row_lines(label, indices, hidden, label_color, codes)))


_ANSI = re.compile(r"\033\[[0-9;]*m")


def visible_len(text):
    """ Length on screen, ignoring ANSI color codes. """
    return len(_ANSI.sub("", text))


def side_by_side(*blocks, gap=4):
    """ Joins blocks of lines horizontally, padding each to its widest visible line. """
    height = max(len(b) for b in blocks)
    widths = [max(visible_len(line) for line in b) for b in blocks]
    out = []
    for row in range(height):
        parts = []
        for b, w in zip(blocks, widths):
            line = b[row] if row < len(b) else ""
            parts.append(line + " " * (w - visible_len(line)))
        out.append((" " * gap).join(parts).rstrip())
    return out

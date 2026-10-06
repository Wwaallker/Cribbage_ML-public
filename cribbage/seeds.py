""" Seed keys: short codes like "K7QM2XPA" that name a game's deals, as in Balatro.

A key is a number written in base 32 with symbols that are hard to mistake for one another
(no 0/O or 1/I). env.reset(seed=key_to_seed(key)) then deals the same cards every time, on
the Python and the Rust engine alike. The cards dealt depend only on the key; what happens
in the game also depends on the moves, which play.py records in replays/<KEY>.json.
"""
import secrets

ALPHABET = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ"      # 32 symbols
KEY_LEN = 8                  # 32**8 = about 10**12 different games


def new_key():
    return "".join(secrets.choice(ALPHABET) for _ in range(KEY_LEN))


def normalise(key):
    """ Upper case, spaces and dashes removed. Raises ValueError for any other character. """
    key = key.strip().upper().replace(" ", "").replace("-", "")
    bad = sorted(set(key) - set(ALPHABET))
    if not key or bad:
        raise ValueError(f"a seed key uses only {ALPHABET}" + (f" (not {''.join(bad)})" if bad else ""))
    return key


def key_to_seed(key):
    n = 0
    for ch in normalise(key):
        n = n * len(ALPHABET) + ALPHABET.index(ch)
    return n

""" The Rust game engine (rust/, built with maturin) behind the same interface as CribbageEnv.

Rules, scoring, the opponent, observations and masks all run in Rust; env.py stays the
reference. The Rust engine carries its own copy of CPython's random generator, so a game
seeded with the same number plays out move for move as it would on the Python engine.

Not available here: replacing _award, _comp_discard, _comp_pick or _count_hands at runtime
(play.py does that, so it always uses the Python engine). metrics.py reads points_by_source()
instead of wrapping _award.
"""
import os

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import cribbage_rs
from cribbage.env import CribbageEnv as PyCribbageEnv, DISCARD_PAIRS, N_ACTIONS, N_CARDS, OBS, OBS_SIZE

# Game state held in Rust, readable and writable as plain attributes like on CribbageEnv.
STATE_FIELDS = ("opp_smart", "phase", "p1_score", "comp_score", "current_peg_sum", "hand",
                "comp_hand", "pegging_table", "peg_history", "crib", "deck", "starter_card",
                "is_ai_dealer", "turn", "last_player", "steps", "deals", "game_over", "winner",
                "ai_scoring_hand", "comp_scoring_hand")


def _seed_key(seed):
    """ random.seed(n) feeds abs(n) to Mersenne Twister as 32-bit words, low word first. """
    n = abs(int(seed))
    key = []
    while n:
        key.append(n & 0xFFFFFFFF)
        n >>= 32
    return key or [0]


class RustCribbageEnv(gym.Env):
    """ CribbageEnv with the game engine in Rust. Same spaces, observations, masks and rewards. """

    def __init__(self, opponent="smart", pool=(), p_smart=0.5):
        super().__init__()
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.obs_size = OBS_SIZE
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.obs_size,), dtype=np.float32)
        self.opponent = opponent
        self._core = cribbage_rs.Core(opponent)
        # like the module-level random generator, start from OS entropy until a seed is given
        self._core.seed(list(np.frombuffer(os.urandom(16), dtype=np.uint32)))
        self._core.p_smart = p_smart
        self.pool = []
        self.set_pool(pool)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._core.seed(_seed_key(seed))
        obs = np.empty(self.obs_size, dtype=np.float32)
        self._core.reset(obs)
        return obs, {}

    def step(self, action):
        obs = np.empty(self.obs_size, dtype=np.float32)
        reward, done, trunc, error = self._core.step(int(action), obs)
        return obs, reward, done, trunc, ({"error": error} if error else {})

    def action_masks(self):
        mask = np.empty(N_ACTIONS, dtype=np.uint8)
        self._core.action_masks(mask)
        return mask.view(bool)

    def _get_obs(self, seat=0):
        obs = np.empty(self.obs_size, dtype=np.float32)
        self._core.get_obs_seat(seat, obs)
        return obs

    def set_pool(self, paths):
        """ See CribbageEnv.set_pool. """
        self._core.clear_nets()
        for path in paths:
            z = np.load(path)
            self._core.add_net(*(z[f"{x}{k}"].astype(np.float64).ravel() for k in range(3) for x in "wb"))
        self.pool = list(paths)

    @property
    def p_smart(self):
        return self._core.p_smart

    @p_smart.setter
    def p_smart(self, value):
        self._core.p_smart = value

    @property
    def opp_net(self):
        """ Index into the pool of the network playing this game, or None. """
        return self._core.opp_net

    def points_by_source(self):
        """ Points awarded this game, {"pegging" | "hand" | "crib" | "his heels": [ai, opp]}. """
        return self._core.points_by_source()

    def discard_action(self, pair):
        slots = tuple(sorted(self.hand.index(c) for c in pair))
        return N_CARDS + DISCARD_PAIRS.index(slots)

    # --- scoring, in Rust ---
    def score_hand(self, hand, starter, is_crib=False):
        return cribbage_rs.score_hand(list(hand), starter, is_crib)

    def _peg_points(self, seq):
        return cribbage_rs.peg_points(list(seq))

    def _expected_best_reply(self, card, own_hand, own_discards, n_opp):
        return self._core.expected_best_reply(card, list(own_hand), list(own_discards), n_opp)

    def discard_values(self, six, owns_crib):
        return self._core.discard_values(list(six), owns_crib)

    # --- the opponent's choices, callable for tests (they cannot be replaced, see above) ---
    def _comp_discard(self):
        return self._core.comp_discard()

    def _comp_pick(self, legal):
        return self._core.comp_pick(list(legal))

    # --- display helpers: not on the training path, so shared with the Python engine ---
    _card_value = PyCribbageEnv._card_value
    hand_breakdown = PyCribbageEnv.hand_breakdown
    peg_breakdown = PyCribbageEnv.peg_breakdown


def _state_property(name):
    return property(lambda self: getattr(self._core, name),
                    lambda self, value: setattr(self._core, name, value))


for _name in STATE_FIELDS:
    setattr(RustCribbageEnv, _name, _state_property(_name))

_rust_offsets = dict(cribbage_rs.Core("smart").obs_offsets())
assert _rust_offsets.pop("size") == OBS_SIZE and _rust_offsets == OBS, \
    "OBS_BLOCKS in env.py and the offsets in rust/src/lib.rs disagree"

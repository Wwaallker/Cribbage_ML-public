import random
import itertools
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

WIN_SCORE = 121
POINT_SCALE = 0.05     # reward per point of differential
WIN_BONUS = 1.0         # reward for winning / losing the whole game
MAX_STEPS = 400          # safety net so no game can run forever

N_CARDS = 52
DISCARD_PAIRS = list(itertools.combinations(range(6), 2))    # 15 ways to pick 2 of 6 hand slots
N_ACTIONS = N_CARDS + len(DISCARD_PAIRS)                      # 0-51 play a card, 52-66 discard a pair

# Observation layout: (name, size). Offsets are derived below so adding a block can't misalign the rest.
OBS_BLOCKS = [
    ("hand", 52),           # cards currently in the AI's hand
    ("table", 52),          # cards in the current count (cleared on go / 31)
    ("starter", 52),
    ("count", 32),          # one-hot running count 0-31
    ("scores", 2),          # AI, computer, scaled to 121
    ("phase", 2),           # discard / pegging
    ("dealer", 1),          # 1 = AI deals (AI owns the crib)
    ("opp_played", 52),     # computer's cards pegged this deal, survives count resets
    ("own_played", 52),     # AI's cards pegged this deal
    ("own_discards", 52),   # the 2 cards the AI put in the crib
    ("opp_cards_left", 5),  # one-hot 0-4 cards left in the computer's hand (pegging only)
    ("hand_slots", 6 * 17), # sorted hand, slot by slot: 13 rank + 4 suit one-hot. Lets the
                            # network tie each DISCARD_PAIRS action to concrete cards.
    # Pegging help, per card the AI could legally play now (scaled by PEG_SCALE, capped at 1).
    # Added after the rest so older models can be grown to fit (tools/grow_obs.py).
    ("peg_now", 52),        # points the card scores as it's played
    ("peg_reply", 52),      # expected points of the computer's best reply to it
]
PEG_SCALE = 1 / 12.0
OBS = {}
_off = 0
for _name, _size in OBS_BLOCKS:
    OBS[_name] = _off
    _off += _size
OBS_SIZE = _off


def hand_sort_key(card):
    """ Rank first, then suit -- the order hand slots (and so discard-pair actions) refer to. """
    return (card % 13, card // 13)


class CribbageEnv(gym.Env):
    """
    Full cribbage, played to 121, dealer alternates each deal.
    action_space: Discrete(67). 0-51 play that card (pegging); 52-66 discard the hand-slot
        pair DISCARD_PAIRS[a - 52] to the crib (discard phase, one decision per deal).
    observation_space: Box(OBS_SIZE,) -- see OBS_BLOCKS.
    opponent: "random", "smart", or "mixed" (50/50 chosen at each reset).
    """

    def __init__(self, opponent="smart"):
        super().__init__()
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.obs_size = OBS_SIZE
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.obs_size,), dtype=np.float32)

        self.opponent = opponent
        self.opp_smart = True

        self.phase = 0                  # 0 = discard, 1 = pegging
        self.p1_score = 0
        self.comp_score = 0
        self.current_peg_sum = 0
        self.hand = []
        self.comp_hand = []
        self.pegging_table = []
        self.peg_history = []           # every card pegged this deal, survives count resets
        self.crib = []
        self.deck = []
        self.starter_card = -1
        self.is_ai_dealer = False
        self.turn = 0                   # 0 = AI, 1 = computer (pegging only)
        self.last_player = None
        self.steps = 0
        self.deals = 1
        self.game_over = False
        self.winner = None
        self.ai_scoring_hand = []
        self.comp_scoring_hand = []

    # ------------------------------------------------------------------
    # OBSERVATION / MASKING
    # ------------------------------------------------------------------
    def _get_obs(self):
        obs = np.zeros(self.obs_size, dtype=np.float32)

        def cards(block, cs):
            for c in cs:
                obs[OBS[block] + c] = 1.0

        cards("hand", self.hand)
        cards("table", self.pegging_table)
        if self.starter_card != -1:
            obs[OBS["starter"] + self.starter_card] = 1.0
        if 0 <= self.current_peg_sum <= 31:
            obs[OBS["count"] + self.current_peg_sum] = 1.0
        obs[OBS["scores"]] = min(self.p1_score / float(WIN_SCORE), 1.0)
        obs[OBS["scores"] + 1] = min(self.comp_score / float(WIN_SCORE), 1.0)
        obs[OBS["phase"] + self.phase] = 1.0
        obs[OBS["dealer"]] = 1.0 if self.is_ai_dealer else 0.0

        if self.phase == 1:
            cards("opp_played", [c for c in self.comp_scoring_hand if c not in self.comp_hand])
            cards("own_played", [c for c in self.ai_scoring_hand if c not in self.hand])
            obs[OBS["opp_cards_left"] + len(self.comp_hand)] = 1.0
        cards("own_discards", self.crib[:2])        # the AI always discards first

        for slot, c in enumerate(self.hand):        # self.hand is kept in hand_sort_key order
            base = OBS["hand_slots"] + slot * 17
            obs[base + c % 13] = 1.0
            obs[base + 13 + c // 13] = 1.0

        if self.phase == 1:
            for c in self._legal_cards(self.hand):
                now = self._peg_points(self.pegging_table + [c])
                reply = self._expected_best_reply(c, self.hand, self.crib[:2], len(self.comp_hand))
                obs[OBS["peg_now"] + c] = min(now * PEG_SCALE, 1.0)
                obs[OBS["peg_reply"] + c] = min(reply * PEG_SCALE, 1.0)
        return obs

    def action_masks(self):
        """ True = legal action. Used by MaskablePPO. """
        mask = np.zeros(N_ACTIONS, dtype=bool)
        if self.phase == 0:
            mask[N_CARDS:] = len(self.hand) == 6
        else:
            for c in self._legal_cards(self.hand):
                mask[c] = True
        if not mask.any():
            mask[:] = True
        return mask

    def discard_action(self, pair):
        """ The action that discards these 2 cards from the AI's hand (for tests, tools, UIs). """
        slots = tuple(sorted(self.hand.index(c) for c in pair))
        return N_CARDS + DISCARD_PAIRS.index(slots)

    # ------------------------------------------------------------------
    # RESET / DEALING
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)

        self.p1_score = 0
        self.comp_score = 0
        self.game_over = False
        self.winner = None
        self.deals = 1
        self.steps = 0

        if self.opponent == "mixed":
            self.opp_smart = random.random() < 0.5
        else:
            self.opp_smart = (self.opponent == "smart")

        self.is_ai_dealer = random.choice([True, False])
        self._new_deal()
        return self._get_obs(), {}

    def _new_deal(self):
        self.phase = 0
        self.current_peg_sum = 0
        self.pegging_table = []
        self.peg_history = []
        self.crib = []
        self.starter_card = -1
        self.last_player = None
        self.turn = 0
        self.ai_scoring_hand = []
        self.comp_scoring_hand = []
        self.deck = list(range(52))
        random.shuffle(self.deck)
        self.hand = sorted((self.deck.pop(0) for _ in range(6)), key=hand_sort_key)
        self.comp_hand = [self.deck.pop(0) for _ in range(6)]

    # ------------------------------------------------------------------
    # CARD HELPERS
    # ------------------------------------------------------------------
    def _card_value(self, card_index):
        rank = card_index % 13
        return 10 if rank >= 9 else rank + 1

    def _can_play(self, cards):
        return any(self.current_peg_sum + self._card_value(c) <= 31 for c in cards)

    def _legal_cards(self, cards):
        return [c for c in cards if self.current_peg_sum + self._card_value(c) <= 31]

    def _reset_count(self):
        self.current_peg_sum = 0
        self.pegging_table = []

    def _award(self, player, pts):
        """ Adds points, ends the game at 121. Returns scaled reward for the AI. """
        if pts == 0 or self.game_over:
            return 0.0
        if player == 0:
            self.p1_score += pts
            r = float(pts)
        else:
            self.comp_score += pts
            r = -float(pts)
        if self.p1_score >= WIN_SCORE:
            self.game_over, self.winner = True, 0
        elif self.comp_score >= WIN_SCORE:
            self.game_over, self.winner = True, 1
        return r * POINT_SCALE


    # ------------------------------------------------------------------
    # OPPONENT BRAINS
    # ------------------------------------------------------------------
    def _comp_discard(self):
        """ Returns the 2 cards the computer sends to the crib.
        Considers not just the kept hand's value, but whether the crib
        being fed is the computer's own (favor pairs/15s in the discard)
        or the opponent's (favor throwing dead cards). """
        if not self.opp_smart:
            return random.sample(self.comp_hand, 2)

        comp_is_dealer = not self.is_ai_dealer
        best_pair, best_val = None, -999.0
        for pair in itertools.combinations(self.comp_hand, 2):
            keep = [c for c in self.comp_hand if c not in pair]
            hand_val = self.score_hand(keep, None)              # cheap: 1 call, no starter loop

            crib_proxy = self._discard_pair_value(pair)      # fixed     # pairs/15 from the 2 discards alone
            crib_term = crib_proxy if comp_is_dealer else -crib_proxy

            val = hand_val + crib_term + random.random() * 0.01
            if val > best_val:
                best_pair, best_val = pair, val
        return list(best_pair)
      
    def _comp_pick(self, legal):
        """ Returns the card the computer plays during pegging.
        2-ply: points scored now, minus the expected points of the AI's best
        reply. The AI's hand is never peeked at -- replies are drawn from the
        cards the computer hasn't seen, weighted by how likely the AI holds them. """
        if not self.opp_smart:
            return random.choice(legal)
        best_card, best_val = None, -99.0
        for c in legal:
            val = self._peg_points(self.pegging_table + [c]) - self._expected_reply(c)
            val += random.random() * 0.5
            if val > best_val:
                best_card, best_val = c, val
        return best_card

    def _expected_reply(self, card):
        """ Expected points of the AI's best pegging reply if the computer plays `card`,
        assuming the AI's remaining cards are a random draw from the unseen cards. """
        return self._expected_best_reply(card, self.comp_hand, self.crib[2:], len(self.hand))

    def _expected_best_reply(self, card, own_hand, own_discards, n_opp):
        """ Expected points of the other player's best pegging reply if `card` is played now,
        seen from the side holding own_hand (who threw own_discards to the crib). The other
        player's n_opp cards (a public count) are treated as a random draw from the cards
        that side hasn't seen. """
        if n_opp == 0:
            return 0.0
        seen = set(own_hand) | set(self.peg_history) | set(own_discards)
        if self.starter_card != -1:
            seen.add(self.starter_card)
        unseen = [u for u in range(52) if u not in seen]

        table = self.pegging_table + [card]
        new_sum = self.current_peg_sum + self._card_value(card)
        if new_sum == 31:                       # count resets; the reply leads fresh, can't score off this card
            return 0.0

        # points each rank would score as a reply, and how many unseen cards have that rank
        by_rank = {}
        for u in unseen:
            if new_sum + self._card_value(u) <= 31:
                by_rank.setdefault(u % 13, [0, 0])[1] += 1
        for r, entry in by_rank.items():
            entry[0] = self._peg_points(table + [r])     # rank r, suit 0 -- suit doesn't matter in pegging

        # E[best reply] = sum over ranks, best first, of pts * P(this is the best rank they hold)
        total = len(unseen)
        denom = math.comb(total, n_opp)
        exp, excluded, p_none_prev = 0.0, 0, 1.0
        for pts, cnt in sorted(by_rank.values(), reverse=True):
            excluded += cnt
            p_none = math.comb(total - excluded, n_opp) / denom
            exp += pts * (p_none_prev - p_none)
            p_none_prev = p_none
        return exp

    def _discard_pair_value(self, pair):
        """ Cheap, correct proxy for what a 2-card discard is worth to a crib:
        pairs and 15s only. (Flush/runs need more cards and don't apply here.) """
        a, b = pair
        val = 0
        if a % 13 == b % 13:
            val += 2
        if self._card_value(a) + self._card_value(b) == 15:
            val += 2
        return val
    # ------------------------------------------------------------------
    # SCORING (public: score_hand is used by external tools too)
    # ------------------------------------------------------------------
    def _peg_points(self, seq):
        pts = 0
        total = sum(self._card_value(c) for c in seq)
        if total == 15:
            pts += 2
        if total == 31:
            pts += 2

        ranks = [c % 13 for c in seq]
        n = 0
        for r in reversed(ranks):
            if r == ranks[-1]:
                n += 1
            else:
                break
        pts += {1: 0, 2: 2, 3: 6, 4: 12}[n]

        for length in range(len(ranks), 2, -1):
            tail = ranks[-length:]
            if len(set(tail)) == length and max(tail) - min(tail) == length - 1:
                pts += length
                break
        return pts

    def peg_breakdown(self, seq):
        """ Same scoring as _peg_points, itemised for display: [("fifteen", 2), ("pair", 2)].
        Kept separate so the hot training path stays a plain sum. """
        items = []
        total = sum(self._card_value(c) for c in seq)
        if total == 15:
            items.append(("fifteen", 2))
        if total == 31:
            items.append(("thirty-one", 2))

        ranks = [c % 13 for c in seq]
        n = 0
        for r in reversed(ranks):
            if r == ranks[-1]:
                n += 1
            else:
                break
        if n >= 2:
            items.append(({2: "pair", 3: "pair royal", 4: "double pair royal"}[n], {2: 2, 3: 6, 4: 12}[n]))

        for length in range(len(ranks), 2, -1):
            tail = ranks[-length:]
            if len(set(tail)) == length and max(tail) - min(tail) == length - 1:
                items.append((f"run of {length}", length))
                break
        return items

    def score_hand(self, hand, starter, is_crib=False):
        """ 15s, pairs, runs, flush, nobs. starter=None scores the 4 cards alone
        (used for comparing discards before the starter is known). """
        cards = hand + ([starter] if starter is not None else [])
        score = 0
        ranks = sorted([c % 13 for c in cards])
        values = [self._card_value(c) for c in cards]

        for r in range(2, len(cards) + 1):
            for combo in itertools.combinations(values, r):
                if sum(combo) == 15:
                    score += 2

        for r1, r2 in itertools.combinations(ranks, 2):
            if r1 == r2:
                score += 2

        counts = {r: ranks.count(r) for r in set(ranks)}
        unique_ranks = sorted(counts.keys())
        best_run, best_mult = 0, 1
        current_run, current_mult = 1, counts[unique_ranks[0]]
        for i in range(1, len(unique_ranks)):
            if unique_ranks[i] == unique_ranks[i - 1] + 1:
                current_run += 1
                current_mult *= counts[unique_ranks[i]]
            else:
                if current_run >= 3 and current_run > best_run:
                    best_run, best_mult = current_run, current_mult
                current_run, current_mult = 1, counts[unique_ranks[i]]
        if current_run >= 3 and current_run > best_run:
            best_run, best_mult = current_run, current_mult
        if best_run >= 3:
            score += best_run * best_mult

        suits = [c // 13 for c in hand]
        if len(set(suits)) == 1:
            if starter is None:
                if not is_crib:
                    score += 4
            elif starter // 13 == suits[0]:
                score += 5
            elif not is_crib:
                score += 4

        if starter is not None:
            for c in hand:
                if c % 13 == 10 and c // 13 == starter // 13:
                    score += 1
        return score

    def hand_breakdown(self, hand, starter, is_crib=False):
        """ Same scoring as score_hand, itemised for display: [("15s", 4), ("run", 3), ("nobs", 1)].
        Kept separate so the hot training path stays a plain sum. """
        items = []
        cards = hand + [starter]
        values = [self._card_value(c) for c in cards]
        fifteens = sum(1 for r in range(2, 6) for combo in itertools.combinations(values, r)
                       if sum(combo) == 15)
        if fifteens:
            items.append(("15s", 2 * fifteens))

        ranks = sorted(c % 13 for c in cards)
        pairs = sum(1 for r1, r2 in itertools.combinations(ranks, 2) if r1 == r2)
        if pairs:
            items.append(("pairs" if pairs > 1 else "pair", 2 * pairs))

        counts = {r: ranks.count(r) for r in set(ranks)}
        unique = sorted(counts)
        best_run, best_mult, run, mult = 0, 1, 1, counts[unique[0]]
        for i in range(1, len(unique)):
            if unique[i] == unique[i - 1] + 1:
                run += 1
                mult *= counts[unique[i]]
            else:
                if run >= 3 and run > best_run:
                    best_run, best_mult = run, mult
                run, mult = 1, counts[unique[i]]
        if run >= 3 and run > best_run:
            best_run, best_mult = run, mult
        if best_run >= 3:
            items.append((f"runs of {best_run}" if best_mult > 1 else f"run of {best_run}",
                          best_run * best_mult))

        suits = [c // 13 for c in hand]
        if len(set(suits)) == 1:
            if starter // 13 == suits[0]:
                items.append(("flush", 5))
            elif not is_crib:
                items.append(("flush", 4))

        if any(c % 13 == 10 and c // 13 == starter // 13 for c in hand):
            items.append(("nobs", 1))
        return items

    # ------------------------------------------------------------------
    # PEGGING ENGINE
    # ------------------------------------------------------------------
    def _advance(self):
        """ Runs the computer's plays, "go"s, and resets until the AI must
        choose a card or the deal's pegging is over. Returns (reward, finished). """
        reward = 0.0
        while True:
            if self.game_over:
                return reward, True

            ai_can = self._can_play(self.hand)
            comp_can = self._can_play(self.comp_hand)

            if not ai_can and not comp_can:
                if self.pegging_table and self.last_player is not None:
                    reward += self._award(self.last_player, 1)
                    if self.game_over:
                        return reward, True
                self._reset_count()
                if not self.hand and not self.comp_hand:
                    return reward, True
                self.turn = 1 - self.last_player
                continue

            if self.turn == 0:
                if ai_can:
                    return reward, False
                self.turn = 1
            else:
                if comp_can:
                    card = self._comp_pick(self._legal_cards(self.comp_hand))
                    self.comp_hand.remove(card)
                    self.pegging_table.append(card)
                    self.peg_history.append(card)
                    self.current_peg_sum += self._card_value(card)
                    self.last_player = 1
                    reward += self._award(1, self._peg_points(self.pegging_table))
                    if self.game_over:
                        return reward, True
                    if self.current_peg_sum == 31:
                        self._reset_count()
                self.turn = 0

    # ------------------------------------------------------------------
    # END OF A DEAL: count hands, then either end the game or deal again
    # ------------------------------------------------------------------
    def _count_hands(self):
        ai_hand = self.score_hand(self.ai_scoring_hand, self.starter_card)
        comp_hand = self.score_hand(self.comp_scoring_hand, self.starter_card)
        crib = self.score_hand(self.crib, self.starter_card, is_crib=True)

        if self.is_ai_dealer:
            order = [(1, comp_hand), (0, ai_hand), (0, crib)]
        else:
            order = [(0, ai_hand), (1, comp_hand), (1, crib)]

        reward = 0.0
        for player, pts in order:
            reward += self._award(player, pts)
            if self.game_over:
                break
        return reward

    def _end_of_deal(self):
        """ Returns (reward, game_done). """
        reward = self._count_hands() if not self.game_over else 0.0
        if self.game_over:
            reward += WIN_BONUS if self.winner == 0 else -WIN_BONUS
            return reward, True

        self.is_ai_dealer = not self.is_ai_dealer
        self.deals += 1
        self._new_deal()
        return reward, False

    # ------------------------------------------------------------------
    # STEP
    # ------------------------------------------------------------------
    def step(self, action):
        action = int(action)
        reward = 0.0

        self.steps += 1
        if self.steps > MAX_STEPS:
            return self._get_obs(), -1.0, False, True, {"error": "timeout"}

        if not (0 <= action < N_ACTIONS) or not self.action_masks()[action] \
                or (self.phase == 1 and action not in self.hand):
            return self._get_obs(), -1.0, False, False, {"error": "illegal_move"}

        finished = False

        if self.phase == 0:
            # --- DISCARD (both cards in one decision) ---
            i, j = DISCARD_PAIRS[action - N_CARDS]
            pair = [self.hand[i], self.hand[j]]
            for c in pair:
                self.hand.remove(c)
                self.crib.append(c)

            for c in self._comp_discard():
                self.comp_hand.remove(c)
                self.crib.append(c)

            self.starter_card = self.deck.pop(0)
            if self.starter_card % 13 == 10:            # his heels
                reward += self._award(0 if self.is_ai_dealer else 1, 2)

            self.ai_scoring_hand = list(self.hand)
            self.comp_scoring_hand = list(self.comp_hand)
            self.phase = 1
            self._reset_count()
            self.last_player = None
            self.turn = 1 if self.is_ai_dealer else 0   # non-dealer leads

            if self.game_over:
                finished = True
            else:
                adv, finished = self._advance()
                reward += adv
        else:
            # --- PEGGING ---
            self.hand.remove(action)
            self.pegging_table.append(action)
            self.peg_history.append(action)
            self.current_peg_sum += self._card_value(action)
            self.last_player = 0
            reward += self._award(0, self._peg_points(self.pegging_table))

            if self.game_over:
                finished = True
            else:
                if self.current_peg_sum == 31:
                    self._reset_count()
                self.turn = 1
                adv, finished = self._advance()
                reward += adv

        done = False
        if finished:
            r, done = self._end_of_deal()
            reward += r
        return self._get_obs(), reward, done, False, {}
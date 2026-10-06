//! The cribbage game engine of cribbage/env.py, in Rust. `Core` holds one game and mirrors
//! CribbageEnv method for method; cribbage/rust_env.py wraps it as a gymnasium Env.
//! Python's env.py remains the reference: any rule change goes there first, then here.

use pyo3::buffer::PyBuffer;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::OnceLock;

mod rng;
mod rules;

use rng::Mt;
use rules::{card_value, comb, discard_pair_value, peg_points, rank, score_hand, suit};

const WIN_SCORE: i32 = 121;
const POINT_SCALE: f64 = 0.05;
const WIN_BONUS: f64 = 1.0;
const MAX_STEPS: u32 = 400;
const N_CARDS: usize = 52;
const N_ACTIONS: usize = N_CARDS + 15;
const PEG_SCALE: f64 = 1.0 / 12.0;
const DISCARD_HALF: f64 = 0.5;

/// Offsets of OBS_BLOCKS in env.py; rust_env.py checks them against the Python layout.
const OBS_HAND: usize = 0;
const OBS_TABLE: usize = 52;
const OBS_STARTER: usize = 104;
const OBS_COUNT: usize = 156;
const OBS_SCORES: usize = 188;
const OBS_PHASE: usize = 190;
const OBS_DEALER: usize = 192;
const OBS_OPP_PLAYED: usize = 193;
const OBS_OWN_PLAYED: usize = 245;
const OBS_OWN_DISCARDS: usize = 297;
const OBS_OPP_CARDS_LEFT: usize = 349;
const OBS_HAND_SLOTS: usize = 354;
const OBS_PEG_NOW: usize = 456;
const OBS_PEG_REPLY: usize = 508;
const OBS_DISCARD_VALUE: usize = 560;
const OBS_SIZE: usize = 575;

/// itertools.combinations(range(6), 2)
const DISCARD_PAIRS: [(usize, usize); 15] = [
    (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 2), (1, 3), (1, 4), (1, 5),
    (2, 3), (2, 4), (2, 5), (3, 4), (3, 5), (4, 5),
];

// sources for the points tally that metrics.py reads
const SRC_PEGGING: usize = 0;
const SRC_HAND: usize = 1;
const SRC_CRIB: usize = 2;
const SRC_HEELS: usize = 3;

/// cribbage/crib_table.json, read at compile time: average crib points of a 2-card throw,
/// by [low rank][high rank][suited]. The file is json.dump(indent=0): one "ra,rb,s": value per line.
fn crib_table() -> &'static [[[f64; 2]; 13]; 13] {
    static TABLE: OnceLock<[[[f64; 2]; 13]; 13]> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut t = [[[f64::NAN; 2]; 13]; 13];
        for line in include_str!("../../cribbage/crib_table.json").lines() {
            let line = line.trim().trim_end_matches(',');
            let Some((key, value)) = line.split_once(": ") else { continue };
            let k: Vec<usize> = key.trim_matches('"').split(',').map(|x| x.parse().unwrap()).collect();
            t[k[0]][k[1]][k[2]] = value.parse().unwrap();
        }
        t
    })
}

/// CribbageEnv.discard_values: expected points of each of the 15 discards of a sorted 6-card
/// hand, in DISCARD_PAIRS order (kept 4 over all 46 starters, plus or minus the crib table).
fn discard_values(six: &[u8], owns_crib: bool) -> [f64; 15] {
    let unseen: Vec<u8> = (0..52u8).filter(|c| !six.contains(c)).collect();
    let mut values = [0f64; 15];
    for (k, &(i, j)) in DISCARD_PAIRS.iter().enumerate() {
        let keep: Vec<u8> = six.iter().enumerate().filter(|&(s, _)| s != i && s != j).map(|(_, &c)| c).collect();
        let total: i32 = unseen.iter().map(|&st| score_hand(&keep, Some(st), false)).sum();
        let hand = total as f64 / unseen.len() as f64;
        let (a, b) = (six[i], six[j]);
        let crib = crib_table()[rank(a).min(rank(b)) as usize][rank(a).max(rank(b)) as usize]
            [(suit(a) == suit(b)) as usize];
        values[k] = hand + if owns_crib { crib } else { -crib };
    }
    values
}

fn hand_sort_key(c: &u8) -> (u8, u8) {
    (rank(*c), suit(*c))
}

fn remove(v: &mut Vec<u8>, c: u8) {
    let i = v.iter().position(|&x| x == c).expect("card not in list");
    v.remove(i);
}

fn with_card(seq: &[u8], c: u8) -> ([u8; 16], usize) {
    let mut buf = [0u8; 16];
    buf[..seq.len()].copy_from_slice(seq);
    buf[seq.len()] = c;
    (buf, seq.len() + 1)
}

#[derive(Clone, Copy, PartialEq)]
enum Opponent {
    Random,
    Smart,
    Mixed,
    Pool,
}

/// OpponentNet in env.py: a trained policy network playing the computer's seat (two tanh layers,
/// then the action logits), in f64. Weights are row-major, as tools/export_opponent.py writes them.
struct Net {
    w0: Vec<f64>,
    b0: Vec<f64>,
    w1: Vec<f64>,
    b1: Vec<f64>,
    w2: Vec<f64>,
    b2: Vec<f64>,
}

/// One layer, w row-major. Zero inputs are skipped (the observation is mostly zeros); adding
/// their zero products would not change the sums.
fn dense(w: &[f64], b: &[f64], x: &[f64], tanh: bool) -> Vec<f64> {
    let n = x.len();
    let nz: Vec<usize> = (0..n).filter(|&j| x[j] != 0.0).collect();
    b.iter()
        .enumerate()
        .map(|(i, &bi)| {
            let row = &w[i * n..(i + 1) * n];
            let dot: f64 = nz.iter().map(|&j| row[j] * x[j]).sum();
            if tanh { (dot + bi).tanh() } else { dot + bi }
        })
        .collect()
}

impl Net {
    fn logits(&self, obs: &[f32; OBS_SIZE]) -> Vec<f64> {
        let x: Vec<f64> = obs.iter().map(|&v| v as f64).collect();
        let h = dense(&self.w0, &self.b0, &x, true);
        let h = dense(&self.w1, &self.b1, &h, true);
        dense(&self.w2, &self.b2, &h, false)
    }

    /// The legal action with the highest logit; the lowest action on a tie.
    fn act(&self, obs: &[f32; OBS_SIZE], actions: &[usize]) -> usize {
        let logits = self.logits(obs);
        let mut sorted = actions.to_vec();
        sorted.sort();
        let mut best = sorted[0];
        for &a in &sorted[1..] {
            if logits[a] > logits[best] {
                best = a;
            }
        }
        best
    }
}

#[pyclass(module = "cribbage_rs")]
struct Core {
    rng: Mt,
    opponent: Opponent,
    #[pyo3(get, set)]
    opp_smart: bool,
    #[pyo3(get, set)]
    phase: u8,
    #[pyo3(get, set)]
    p1_score: i32,
    #[pyo3(get, set)]
    comp_score: i32,
    #[pyo3(get, set)]
    current_peg_sum: i32,
    #[pyo3(get, set)]
    hand: Vec<u8>,
    #[pyo3(get, set)]
    comp_hand: Vec<u8>,
    #[pyo3(get, set)]
    pegging_table: Vec<u8>,
    #[pyo3(get, set)]
    peg_history: Vec<u8>,
    #[pyo3(get, set)]
    crib: Vec<u8>,
    #[pyo3(get, set)]
    deck: Vec<u8>,
    #[pyo3(get, set)]
    starter_card: i32,
    #[pyo3(get, set)]
    is_ai_dealer: bool,
    #[pyo3(get, set)]
    turn: u8,
    #[pyo3(get, set)]
    last_player: Option<u8>,
    #[pyo3(get, set)]
    steps: u32,
    #[pyo3(get, set)]
    deals: u32,
    #[pyo3(get, set)]
    game_over: bool,
    #[pyo3(get, set)]
    winner: Option<u8>,
    #[pyo3(get, set)]
    ai_scoring_hand: Vec<u8>,
    #[pyo3(get, set)]
    comp_scoring_hand: Vec<u8>,
    points: [[i32; 2]; 4],
    nets: Vec<Net>,
    #[pyo3(get, set)]
    p_smart: f64,
    #[pyo3(get)]
    opp_net: Option<usize>,
}

// ----------------------------------------------------------------------
// Game logic (plain Rust)
// ----------------------------------------------------------------------
impl Core {
    fn starter(&self) -> Option<u8> {
        if self.starter_card >= 0 {
            Some(self.starter_card as u8)
        } else {
            None
        }
    }

    fn can_play(&self, cards: &[u8]) -> bool {
        cards.iter().any(|&c| self.current_peg_sum + card_value(c) <= 31)
    }

    fn legal_cards(&self, cards: &[u8]) -> Vec<u8> {
        cards.iter().copied().filter(|&c| self.current_peg_sum + card_value(c) <= 31).collect()
    }

    fn reset_count(&mut self) {
        self.current_peg_sum = 0;
        self.pegging_table.clear();
    }

    fn award(&mut self, player: u8, pts: i32, src: usize) -> f64 {
        if pts == 0 || self.game_over {
            return 0.0;
        }
        self.points[src][player as usize] += pts;
        let r = if player == 0 {
            self.p1_score += pts;
            pts as f64
        } else {
            self.comp_score += pts;
            -(pts as f64)
        };
        if self.p1_score >= WIN_SCORE {
            self.game_over = true;
            self.winner = Some(0);
        } else if self.comp_score >= WIN_SCORE {
            self.game_over = true;
            self.winner = Some(1);
        }
        r * POINT_SCALE
    }

    fn do_reset(&mut self) {
        self.p1_score = 0;
        self.comp_score = 0;
        self.game_over = false;
        self.winner = None;
        self.deals = 1;
        self.steps = 0;
        self.points = [[0; 2]; 4];
        self.opp_net = None;
        self.opp_smart = match self.opponent {
            Opponent::Mixed => self.rng.random() < 0.5,
            Opponent::Pool => {
                if self.rng.random() >= self.p_smart && !self.nets.is_empty() {
                    self.opp_net = Some(self.rng.randbelow(self.nets.len()));
                }
                true
            }
            o => o == Opponent::Smart,
        };
        self.is_ai_dealer = self.rng.randbelow(2) == 0; // random.choice([True, False])
        self.new_deal();
    }

    fn new_deal(&mut self) {
        self.phase = 0;
        self.reset_count();
        self.peg_history.clear();
        self.crib.clear();
        self.starter_card = -1;
        self.last_player = None;
        self.turn = 0;
        self.ai_scoring_hand.clear();
        self.comp_scoring_hand.clear();
        let mut deck: Vec<u8> = (0..52).collect();
        self.rng.shuffle(&mut deck);
        self.hand = deck[..6].to_vec();
        self.hand.sort_by_key(hand_sort_key);
        self.comp_hand = deck[6..12].to_vec();
        self.deck = deck[12..].to_vec();
    }

    // --- opponent -------------------------------------------------------

    fn comp_discard(&mut self) -> [u8; 2] {
        if let Some(k) = self.opp_net {
            let mut six = self.comp_hand.clone();
            six.sort_by_key(hand_sort_key);
            let mut obs = [0f32; OBS_SIZE];
            self.fill_obs(1, &mut obs);
            let actions: Vec<usize> = (N_CARDS..N_ACTIONS).collect();
            let (i, j) = DISCARD_PAIRS[self.nets[k].act(&obs, &actions) - N_CARDS];
            return [six[i], six[j]];
        }
        if !self.opp_smart {
            return self.rng.sample2(&self.comp_hand);
        }
        let comp_is_dealer = !self.is_ai_dealer;
        let h = self.comp_hand.clone();
        let (mut best, mut best_val) = ([h[0], h[1]], -999.0);
        for i in 0..h.len() {
            for j in i + 1..h.len() {
                let keep: Vec<u8> =
                    h.iter().enumerate().filter(|&(k, _)| k != i && k != j).map(|(_, &c)| c).collect();
                let hand_val = score_hand(&keep, None, false);
                let proxy = discard_pair_value(h[i], h[j]);
                let crib_term = if comp_is_dealer { proxy } else { -proxy };
                let val = (hand_val + crib_term) as f64 + self.rng.random() * 0.01;
                if val > best_val {
                    best = [h[i], h[j]];
                    best_val = val;
                }
            }
        }
        best
    }

    fn comp_pick(&mut self, legal: &[u8]) -> u8 {
        if let Some(k) = self.opp_net {
            let mut obs = [0f32; OBS_SIZE];
            self.fill_obs(1, &mut obs);
            let actions: Vec<usize> = legal.iter().map(|&c| c as usize).collect();
            return self.nets[k].act(&obs, &actions) as u8;
        }
        if !self.opp_smart {
            return legal[self.rng.randbelow(legal.len())];
        }
        let (mut best, mut best_val) = (legal[0], -99.0);
        for &c in legal {
            let (buf, n) = with_card(&self.pegging_table, c);
            let mut val = peg_points(&buf[..n]) as f64 - self.expected_reply(c);
            val += self.rng.random() * 0.5;
            if val > best_val {
                best = c;
                best_val = val;
            }
        }
        best
    }

    fn expected_reply(&self, card: u8) -> f64 {
        let discards = if self.crib.len() > 2 { &self.crib[2..] } else { &[][..] };
        self.expected_best_reply(card, &self.comp_hand, discards, self.hand.len())
    }

    /// Expected points of the other side's best pegging reply if `card` is played now, seen
    /// from the side holding own_hand. See env.py for the reasoning.
    fn expected_best_reply(&self, card: u8, own_hand: &[u8], own_discards: &[u8], n_opp: usize) -> f64 {
        if n_opp == 0 {
            return 0.0;
        }
        let mut seen: u64 = 0;
        for &c in own_hand.iter().chain(&self.peg_history).chain(own_discards) {
            seen |= 1 << c;
        }
        if let Some(s) = self.starter() {
            seen |= 1 << s;
        }
        let total = 52 - (seen & ((1u64 << 52) - 1)).count_ones() as usize;

        let new_sum = self.current_peg_sum + card_value(card);
        if new_sum == 31 {
            return 0.0;
        }
        let mut counts = [0usize; 13];
        for u in 0..52u8 {
            if seen >> u & 1 == 0 && new_sum + card_value(u) <= 31 {
                counts[rank(u) as usize] += 1;
            }
        }
        let (table, tn) = with_card(&self.pegging_table, card);
        let mut by_rank: Vec<(i32, usize)> = Vec::with_capacity(13);
        for r in 0..13u8 {
            if counts[r as usize] > 0 {
                let (seq, n) = with_card(&table[..tn], r);
                by_rank.push((peg_points(&seq[..n]), counts[r as usize]));
            }
        }
        by_rank.sort_unstable_by(|a, b| b.cmp(a)); // as Python's sorted(..., reverse=True)

        let denom = comb(total, n_opp) as f64;
        let (mut exp, mut excluded, mut p_none_prev) = (0.0, 0usize, 1.0);
        for (pts, cnt) in by_rank {
            excluded += cnt;
            let p_none = comb(total - excluded, n_opp) as f64 / denom;
            exp += pts as f64 * (p_none_prev - p_none);
            p_none_prev = p_none;
        }
        exp
    }

    // --- observation / masks ------------------------------------------

    /// What one seat sees, from its own point of view: 0 = the AI, 1 = the computer.
    fn fill_obs(&self, seat: u8, obs: &mut [f32; OBS_SIZE]) {
        let sorted_comp;
        let (hand, other, mine, theirs, i_deal, my_start, their_start, my_discards): (
            &[u8], &[u8], i32, i32, bool, &[u8], &[u8], &[u8]);
        if seat == 0 {
            (hand, other, mine, theirs, i_deal) =
                (&self.hand, &self.comp_hand, self.p1_score, self.comp_score, self.is_ai_dealer);
            (my_start, their_start) = (&self.ai_scoring_hand, &self.comp_scoring_hand);
            my_discards = &self.crib[..self.crib.len().min(2)]; // the AI always discards first
        } else {
            let mut h = self.comp_hand.clone();
            h.sort_by_key(hand_sort_key);
            sorted_comp = h;
            (hand, other, mine, theirs, i_deal) =
                (&sorted_comp, &self.hand, self.comp_score, self.p1_score, !self.is_ai_dealer);
            (my_start, their_start) = (&self.comp_scoring_hand, &self.ai_scoring_hand);
            my_discards = &self.crib[self.crib.len().min(2)..self.crib.len().min(4)];
        }

        obs.fill(0.0);
        for &c in hand {
            obs[OBS_HAND + c as usize] = 1.0;
        }
        for &c in &self.pegging_table {
            obs[OBS_TABLE + c as usize] = 1.0;
        }
        if let Some(s) = self.starter() {
            obs[OBS_STARTER + s as usize] = 1.0;
        }
        if (0..=31).contains(&self.current_peg_sum) {
            obs[OBS_COUNT + self.current_peg_sum as usize] = 1.0;
        }
        obs[OBS_SCORES] = (mine as f64 / WIN_SCORE as f64).min(1.0) as f32;
        obs[OBS_SCORES + 1] = (theirs as f64 / WIN_SCORE as f64).min(1.0) as f32;
        obs[OBS_PHASE + self.phase as usize] = 1.0;
        obs[OBS_DEALER] = if i_deal { 1.0 } else { 0.0 };

        if self.phase == 1 {
            for &c in their_start {
                if !other.contains(&c) {
                    obs[OBS_OPP_PLAYED + c as usize] = 1.0;
                }
            }
            for &c in my_start {
                if !hand.contains(&c) {
                    obs[OBS_OWN_PLAYED + c as usize] = 1.0;
                }
            }
            obs[OBS_OPP_CARDS_LEFT + other.len()] = 1.0;
        }
        for &c in my_discards {
            obs[OBS_OWN_DISCARDS + c as usize] = 1.0;
        }

        for (slot, &c) in hand.iter().enumerate() {
            let base = OBS_HAND_SLOTS + slot * 17;
            obs[base + rank(c) as usize] = 1.0;
            obs[base + 13 + suit(c) as usize] = 1.0;
        }

        if self.phase == 1 {
            for c in self.legal_cards(hand) {
                let (buf, n) = with_card(&self.pegging_table, c);
                let now = peg_points(&buf[..n]);
                let reply = self.expected_best_reply(c, hand, my_discards, other.len());
                obs[OBS_PEG_NOW + c as usize] = (now as f64 * PEG_SCALE).min(1.0) as f32;
                obs[OBS_PEG_REPLY + c as usize] = (reply * PEG_SCALE).min(1.0) as f32;
            }
        }

        if self.phase == 0 && hand.len() == 6 {
            let values = discard_values(hand, i_deal);
            let best = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            for (i, v) in values.iter().enumerate() {
                obs[OBS_DISCARD_VALUE + i] = (1.0 / (1.0 + (best - v) / DISCARD_HALF)) as f32;
            }
        }
    }

    fn masks(&self) -> [bool; N_ACTIONS] {
        let mut mask = [false; N_ACTIONS];
        if self.phase == 0 {
            if self.hand.len() == 6 {
                mask[N_CARDS..].fill(true);
            }
        } else {
            for c in self.legal_cards(&self.hand) {
                mask[c as usize] = true;
            }
        }
        if !mask.iter().any(|&m| m) {
            mask.fill(true);
        }
        mask
    }

    // --- pegging engine -------------------------------------------------

    /// Runs the computer's plays, "go"s and resets until the AI must choose a card or the
    /// deal's pegging is over. Returns (reward, finished).
    fn advance(&mut self) -> (f64, bool) {
        let mut reward = 0.0;
        loop {
            if self.game_over {
                return (reward, true);
            }
            let ai_can = self.can_play(&self.hand);
            let comp_can = self.can_play(&self.comp_hand);

            if !ai_can && !comp_can {
                if !self.pegging_table.is_empty() {
                    if let Some(lp) = self.last_player {
                        reward += self.award(lp, 1, SRC_PEGGING);
                        if self.game_over {
                            return (reward, true);
                        }
                    }
                }
                self.reset_count();
                if self.hand.is_empty() && self.comp_hand.is_empty() {
                    return (reward, true);
                }
                self.turn = 1 - self.last_player.unwrap_or(1);
                continue;
            }

            if self.turn == 0 {
                if ai_can {
                    return (reward, false);
                }
                self.turn = 1;
            } else {
                if comp_can {
                    let legal = self.legal_cards(&self.comp_hand);
                    let card = self.comp_pick(&legal);
                    remove(&mut self.comp_hand, card);
                    self.pegging_table.push(card);
                    self.peg_history.push(card);
                    self.current_peg_sum += card_value(card);
                    self.last_player = Some(1);
                    let pts = peg_points(&self.pegging_table);
                    reward += self.award(1, pts, SRC_PEGGING);
                    if self.game_over {
                        return (reward, true);
                    }
                    if self.current_peg_sum == 31 {
                        self.reset_count();
                    }
                }
                self.turn = 0;
            }
        }
    }

    // --- end of a deal --------------------------------------------------

    fn count_hands(&mut self) -> f64 {
        let st = self.starter();
        let ai_hand = score_hand(&self.ai_scoring_hand, st, false);
        let comp_hand = score_hand(&self.comp_scoring_hand, st, false);
        let crib = score_hand(&self.crib, st, true);
        let order = if self.is_ai_dealer {
            [(1, comp_hand, SRC_HAND), (0, ai_hand, SRC_HAND), (0, crib, SRC_CRIB)]
        } else {
            [(0, ai_hand, SRC_HAND), (1, comp_hand, SRC_HAND), (1, crib, SRC_CRIB)]
        };
        let mut reward = 0.0;
        for (player, pts, src) in order {
            reward += self.award(player, pts, src);
            if self.game_over {
                break;
            }
        }
        reward
    }

    fn end_of_deal(&mut self) -> (f64, bool) {
        let mut reward = if !self.game_over { self.count_hands() } else { 0.0 };
        if self.game_over {
            reward += if self.winner == Some(0) { WIN_BONUS } else { -WIN_BONUS };
            return (reward, true);
        }
        self.is_ai_dealer = !self.is_ai_dealer;
        self.deals += 1;
        self.new_deal();
        (reward, false)
    }

    /// Returns (reward, terminated, truncated, error).
    fn do_step(&mut self, action: i64) -> (f64, bool, bool, Option<&'static str>) {
        let mut reward = 0.0;
        self.steps += 1;
        if self.steps > MAX_STEPS {
            return (-1.0, false, true, Some("timeout"));
        }
        if !(0..N_ACTIONS as i64).contains(&action)
            || !self.masks()[action as usize]
            || (self.phase == 1 && !self.hand.contains(&(action as u8)))
        {
            return (-1.0, false, false, Some("illegal_move"));
        }
        let finished;

        if self.phase == 0 {
            let (i, j) = DISCARD_PAIRS[action as usize - N_CARDS];
            let pair = [self.hand[i], self.hand[j]];
            for c in pair {
                remove(&mut self.hand, c);
                self.crib.push(c);
            }
            for c in self.comp_discard() {
                remove(&mut self.comp_hand, c);
                self.crib.push(c);
            }
            let starter = self.deck.remove(0);
            self.starter_card = starter as i32;
            if rank(starter) == 10 {
                // his heels
                let dealer = if self.is_ai_dealer { 0 } else { 1 };
                reward += self.award(dealer, 2, SRC_HEELS);
            }
            self.ai_scoring_hand = self.hand.clone();
            self.comp_scoring_hand = self.comp_hand.clone();
            self.phase = 1;
            self.reset_count();
            self.last_player = None;
            self.turn = if self.is_ai_dealer { 1 } else { 0 };

            if self.game_over {
                finished = true;
            } else {
                let (adv, f) = self.advance();
                reward += adv;
                finished = f;
            }
        } else {
            let card = action as u8;
            remove(&mut self.hand, card);
            self.pegging_table.push(card);
            self.peg_history.push(card);
            self.current_peg_sum += card_value(card);
            self.last_player = Some(0);
            let pts = peg_points(&self.pegging_table);
            reward += self.award(0, pts, SRC_PEGGING);

            if self.game_over {
                finished = true;
            } else {
                if self.current_peg_sum == 31 {
                    self.reset_count();
                }
                self.turn = 1;
                let (adv, f) = self.advance();
                reward += adv;
                finished = f;
            }
        }

        let mut done = false;
        if finished {
            let (r, d) = self.end_of_deal();
            reward += r;
            done = d;
        }
        (reward, done, false, None)
    }
}

// ----------------------------------------------------------------------
// Python interface
// ----------------------------------------------------------------------
fn write_obs(core: &Core, py: Python<'_>, buf: &PyBuffer<f32>) -> PyResult<()> {
    let mut obs = [0f32; OBS_SIZE];
    core.fill_obs(0, &mut obs);
    buf.copy_from_slice(py, &obs)
}

#[pymethods]
impl Core {
    #[new]
    fn new(opponent: &str) -> PyResult<Self> {
        let opponent = match opponent {
            "random" => Opponent::Random,
            "smart" => Opponent::Smart,
            "mixed" => Opponent::Mixed,
            "pool" => Opponent::Pool,
            o => return Err(PyValueError::new_err(format!("unknown opponent {o:?}"))),
        };
        Ok(Core {
            rng: Mt::new(),
            opponent,
            opp_smart: true,
            phase: 0,
            p1_score: 0,
            comp_score: 0,
            current_peg_sum: 0,
            hand: vec![],
            comp_hand: vec![],
            pegging_table: vec![],
            peg_history: vec![],
            crib: vec![],
            deck: vec![],
            starter_card: -1,
            is_ai_dealer: false,
            turn: 0,
            last_player: None,
            steps: 0,
            deals: 1,
            game_over: false,
            winner: None,
            ai_scoring_hand: vec![],
            comp_scoring_hand: vec![],
            points: [[0; 2]; 4],
            nets: vec![],
            p_smart: 0.5,
            opp_net: None,
        })
    }

    /// Seeds the generator as random.seed(n) would: key = abs(n) in 32-bit words, low first.
    fn seed(&mut self, key: Vec<u32>) {
        self.rng.seed(&key);
    }

    fn reset(&mut self, py: Python<'_>, obs: PyBuffer<f32>) -> PyResult<()> {
        self.do_reset();
        write_obs(self, py, &obs)
    }

    fn step(&mut self, py: Python<'_>, action: i64, obs: PyBuffer<f32>)
            -> PyResult<(f64, bool, bool, Option<&'static str>)> {
        let out = self.do_step(action);
        write_obs(self, py, &obs)?;
        Ok(out)
    }

    fn get_obs(&self, py: Python<'_>, obs: PyBuffer<f32>) -> PyResult<()> {
        write_obs(self, py, &obs)
    }

    /// The observation of either seat (1 = the computer), for tests.
    fn get_obs_seat(&self, py: Python<'_>, seat: u8, buf: PyBuffer<f32>) -> PyResult<()> {
        let mut obs = [0f32; OBS_SIZE];
        self.fill_obs(seat, &mut obs);
        buf.copy_from_slice(py, &obs)
    }

    /// Adds a network to the opponent="pool" draw (layers as tools/export_opponent.py writes them).
    fn add_net(&mut self, w0: Vec<f64>, b0: Vec<f64>, w1: Vec<f64>, b1: Vec<f64>,
               w2: Vec<f64>, b2: Vec<f64>) -> PyResult<()> {
        if w0.len() != b0.len() * OBS_SIZE || w1.len() != b1.len() * b0.len()
            || w2.len() != b2.len() * b1.len() || b2.len() != N_ACTIONS {
            return Err(PyValueError::new_err("network shapes do not fit the observation and actions"));
        }
        self.nets.push(Net { w0, b0, w1, b1, w2, b2 });
        Ok(())
    }

    fn clear_nets(&mut self) {
        self.nets.clear();
    }

    /// An opponent network's logits for an observation, for tests.
    fn net_logits(&self, k: usize, obs: Vec<f32>) -> Vec<f64> {
        let mut o = [0f32; OBS_SIZE];
        o.copy_from_slice(&obs);
        self.nets[k].logits(&o)
    }

    /// Writes the action mask (1 = legal) into a uint8 buffer of N_ACTIONS.
    fn action_masks(&self, py: Python<'_>, mask: PyBuffer<u8>) -> PyResult<()> {
        let m = self.masks().map(|b| b as u8);
        mask.copy_from_slice(py, &m)
    }

    /// Points awarded this game by source, as metrics.py tallies them: {src: [ai, opp]}.
    fn points_by_source<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        for (name, src) in [("pegging", SRC_PEGGING), ("hand", SRC_HAND), ("crib", SRC_CRIB),
                            ("his heels", SRC_HEELS)] {
            d.set_item(name, self.points[src].to_vec())?;
        }
        Ok(d)
    }

    fn obs_offsets(&self) -> Vec<(&'static str, usize)> {
        vec![("hand", OBS_HAND), ("table", OBS_TABLE), ("starter", OBS_STARTER),
             ("count", OBS_COUNT), ("scores", OBS_SCORES), ("phase", OBS_PHASE),
             ("dealer", OBS_DEALER), ("opp_played", OBS_OPP_PLAYED), ("own_played", OBS_OWN_PLAYED),
             ("own_discards", OBS_OWN_DISCARDS), ("opp_cards_left", OBS_OPP_CARDS_LEFT),
             ("hand_slots", OBS_HAND_SLOTS), ("peg_now", OBS_PEG_NOW), ("peg_reply", OBS_PEG_REPLY),
             ("discard_value", OBS_DISCARD_VALUE), ("size", OBS_SIZE)]
    }

    // exposed for tests and tools
    #[pyo3(name = "expected_best_reply")]
    fn py_expected_best_reply(&self, card: u8, own_hand: Vec<u8>, own_discards: Vec<u8>, n_opp: usize) -> f64 {
        self.expected_best_reply(card, &own_hand, &own_discards, n_opp)
    }

    #[pyo3(name = "discard_values")]
    fn py_discard_values(&self, six: Vec<u8>, owns_crib: bool) -> Vec<f64> {
        discard_values(&six, owns_crib).to_vec()
    }

    #[pyo3(name = "comp_discard")]
    fn py_comp_discard(&mut self) -> Vec<u8> {
        self.comp_discard().to_vec()
    }

    #[pyo3(name = "comp_pick")]
    fn py_comp_pick(&mut self, legal: Vec<u8>) -> u8 {
        self.comp_pick(&legal)
    }

    fn random(&mut self) -> f64 {
        self.rng.random()
    }
}

#[pyfunction]
#[pyo3(name = "score_hand", signature = (hand, starter=None, is_crib=false))]
fn py_score_hand(hand: Vec<u8>, starter: Option<u8>, is_crib: bool) -> i32 {
    score_hand(&hand, starter, is_crib)
}

#[pyfunction]
#[pyo3(name = "peg_points")]
fn py_peg_points(seq: Vec<u8>) -> i32 {
    peg_points(&seq)
}

#[pymodule]
fn cribbage_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Core>()?;
    m.add_function(wrap_pyfunction!(py_score_hand, m)?)?;
    m.add_function(wrap_pyfunction!(py_peg_points, m)?)?;
    Ok(())
}

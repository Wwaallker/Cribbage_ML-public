//! Scoring, card for card as cribbage/env.py computes it. Cards are 0-51: rank = c % 13
//! (0 = ace, 10 = jack), suit = c / 13.

#[inline]
pub fn rank(c: u8) -> u8 {
    c % 13
}

#[inline]
pub fn suit(c: u8) -> u8 {
    c / 13
}

#[inline]
pub fn card_value(c: u8) -> i32 {
    let r = rank(c) as i32;
    if r >= 9 {
        10
    } else {
        r + 1
    }
}

/// Points for the last card of a pegging sequence: 15, 31, pairs, runs.
pub fn peg_points(seq: &[u8]) -> i32 {
    if seq.is_empty() {
        return 0;
    }
    let mut pts = 0;
    let total: i32 = seq.iter().map(|&c| card_value(c)).sum();
    if total == 15 {
        pts += 2;
    }
    if total == 31 {
        pts += 2;
    }

    let last = rank(seq[seq.len() - 1]);
    let n = seq.iter().rev().take_while(|&&c| rank(c) == last).count();
    pts += match n {
        0 | 1 => 0,
        2 => 2,
        3 => 6,
        _ => 12,
    };

    for length in (3..=seq.len()).rev() {
        let tail = &seq[seq.len() - length..];
        let mut seen: u16 = 0;
        let (mut lo, mut hi) = (u8::MAX, 0u8);
        for &c in tail {
            let r = rank(c);
            seen |= 1 << r;
            lo = lo.min(r);
            hi = hi.max(r);
        }
        if seen.count_ones() as usize == length && (hi - lo) as usize == length - 1 {
            pts += length as i32;
            break;
        }
    }
    pts
}

/// 15s, pairs, runs, flush, nobs. `starter` None scores the hand alone (discard comparisons).
pub fn score_hand(hand: &[u8], starter: Option<u8>, is_crib: bool) -> i32 {
    let mut cards = [0u8; 16];
    let n = hand.len() + starter.is_some() as usize;
    cards[..hand.len()].copy_from_slice(hand);
    if let Some(s) = starter {
        cards[hand.len()] = s;
    }
    let cards = &cards[..n];
    if cards.is_empty() {
        return 0;
    }
    let mut score = 0;

    // fifteens: every subset of 2+ cards
    for mask in 1u32..(1 << n) {
        if mask.count_ones() < 2 {
            continue;
        }
        let s: i32 = (0..n).filter(|i| mask >> i & 1 == 1).map(|i| card_value(cards[i])).sum();
        if s == 15 {
            score += 2;
        }
    }

    // pairs, and rank counts for runs
    let mut counts = [0i32; 13];
    for &c in cards {
        counts[rank(c) as usize] += 1;
    }
    for &k in &counts {
        score += k * (k - 1); // k choose 2 pairs, 2 points each
    }

    // runs: longest run of consecutive ranks (3+), times the multiplicity of its ranks
    let (mut best_run, mut best_mult) = (0, 1);
    let (mut run, mut mult) = (0, 1);
    let mut prev: i32 = -2;
    for r in 0..13i32 {
        let k = counts[r as usize];
        if k == 0 {
            continue;
        }
        if run > 0 && r == prev + 1 {
            run += 1;
            mult *= k;
        } else {
            if run >= 3 && run > best_run {
                best_run = run;
                best_mult = mult;
            }
            run = 1;
            mult = k;
        }
        prev = r;
    }
    if run >= 3 && run > best_run {
        best_run = run;
        best_mult = mult;
    }
    if best_run >= 3 {
        score += best_run * best_mult;
    }

    // flush: the hand's own cards all one suit (the crib needs the starter too)
    if !hand.is_empty() && hand.iter().all(|&c| suit(c) == suit(hand[0])) {
        match starter {
            None => {
                if !is_crib {
                    score += 4
                }
            }
            Some(s) if suit(s) == suit(hand[0]) => score += 5,
            Some(_) => {
                if !is_crib {
                    score += 4
                }
            }
        }
    }

    // nobs: jack of the starter's suit
    if let Some(s) = starter {
        for &c in hand {
            if rank(c) == 10 && suit(c) == suit(s) {
                score += 1;
            }
        }
    }
    score
}

/// Pairs and 15s in a 2-card discard: the smart opponent's crib proxy.
pub fn discard_pair_value(a: u8, b: u8) -> i32 {
    let mut v = 0;
    if rank(a) == rank(b) {
        v += 2;
    }
    if card_value(a) + card_value(b) == 15 {
        v += 2;
    }
    v
}

/// math.comb(n, k): 0 when k > n. Exact for everything a 52-card deck needs.
pub fn comb(n: usize, k: usize) -> u64 {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    let mut r: u64 = 1;
    for i in 0..k {
        r = r * (n - i) as u64 / (i + 1) as u64;
    }
    r
}

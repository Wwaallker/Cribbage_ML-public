//! CPython's `random` module, reproduced exactly: MT19937 seeded the way `random.seed(int)`
//! seeds it, plus `random()`, `_randbelow`, `shuffle`, `choice` and `sample` as
//! Lib/random.py (3.12) builds them. Seeded games therefore consume the same random numbers
//! as the Python engine and can be compared move for move.

const N: usize = 624;
const M: usize = 397;

pub struct Mt {
    mt: [u32; N],
    idx: usize,
}

impl Mt {
    pub fn new() -> Self {
        let mut r = Mt { mt: [0; N], idx: N };
        r.seed(&[0]);
        r
    }

    fn init_genrand(&mut self, s: u32) {
        self.mt[0] = s;
        for i in 1..N {
            let prev = self.mt[i - 1];
            self.mt[i] = 1812433253u32.wrapping_mul(prev ^ (prev >> 30)).wrapping_add(i as u32);
        }
        self.idx = N;
    }

    /// `random.seed(n)`: `key` is abs(n) split into 32-bit words, least significant first
    /// ([0] for n == 0).
    pub fn seed(&mut self, key: &[u32]) {
        let key: &[u32] = if key.is_empty() { &[0] } else { key };
        self.init_genrand(19650218);
        let (mut i, mut j) = (1usize, 0usize);
        for _ in 0..N.max(key.len()) {
            let prev = self.mt[i - 1];
            self.mt[i] = (self.mt[i] ^ (prev ^ (prev >> 30)).wrapping_mul(1664525))
                .wrapping_add(key[j])
                .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= N {
                self.mt[0] = self.mt[N - 1];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
        }
        for _ in 0..N - 1 {
            let prev = self.mt[i - 1];
            self.mt[i] = (self.mt[i] ^ (prev ^ (prev >> 30)).wrapping_mul(1566083941))
                .wrapping_sub(i as u32);
            i += 1;
            if i >= N {
                self.mt[0] = self.mt[N - 1];
                i = 1;
            }
        }
        self.mt[0] = 0x8000_0000;
        self.idx = N;
    }

    fn next_u32(&mut self) -> u32 {
        if self.idx >= N {
            for k in 0..N {
                let y = (self.mt[k] & 0x8000_0000) | (self.mt[(k + 1) % N] & 0x7fff_ffff);
                let mag = if y & 1 == 1 { 0x9908_b0df } else { 0 };
                self.mt[k] = self.mt[(k + M) % N] ^ (y >> 1) ^ mag;
            }
            self.idx = 0;
        }
        let mut y = self.mt[self.idx];
        self.idx += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    /// `random.random()`: 53-bit float in [0, 1).
    pub fn random(&mut self) -> f64 {
        let a = (self.next_u32() >> 5) as f64;
        let b = (self.next_u32() >> 6) as f64;
        (a * 67108864.0 + b) * (1.0 / 9007199254740992.0)
    }

    /// `random._randbelow(n)`, n > 0: rejection sampling on n.bit_length() bits.
    pub fn randbelow(&mut self, n: usize) -> usize {
        let n = n as u32;
        let k = 32 - n.leading_zeros();
        loop {
            let r = self.next_u32() >> (32 - k);
            if r < n {
                return r as usize;
            }
        }
    }

    pub fn shuffle<T>(&mut self, x: &mut [T]) {
        for i in (1..x.len()).rev() {
            let j = self.randbelow(i + 1);
            x.swap(i, j);
        }
    }

    /// `random.sample(pop, 2)` for a small population (the pool branch, n <= 21).
    pub fn sample2<T: Copy>(&mut self, pop: &[T]) -> [T; 2] {
        let n = pop.len();
        debug_assert!(n >= 2 && n <= 21);
        let mut pool = pop.to_vec();
        let j = self.randbelow(n);
        let a = pool[j];
        pool[j] = pool[n - 1];
        let b = pool[self.randbelow(n - 1)];
        [a, b]
    }
}

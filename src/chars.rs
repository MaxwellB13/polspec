//! Characters drawn in bulk from an alphabet, each equally likely.
//!
//! A plain string's characters, and a template's runs of characters, come
//! from one continuous stream per seed chunk: a draw of 64 bits is cut into
//! lanes of `bits` bits, the smallest width that indexes the alphabet, and a
//! lane past the end of the alphabet is skipped. Skipping, not folding back,
//! is what keeps every character equally likely.
//!
//! No branch decides a lane's fate. Every lane is written at the next
//! position, and the position moves on only past a kept one, so a skipped
//! lane is overwritten by the next. That needs a draw's worth of scratch past
//! the end of the output (`SLACK`), and it is why a fill runs at about a third
//! of a nanosecond a character where one `Uniform` draw a character ran at one
//! and a half.

use rand::Rng;
use rand_xoshiro::Xoshiro256PlusPlus;

/// Bytes past the end of an output a fill may write: the most lanes one draw
/// is cut into. A draw holds 64 one-bit lanes, but no more than 32 are taken,
/// so the slack stays small; a two-character alphabet loses nothing by it.
pub const SLACK: usize = 32;

/// An alphabet, prepared for bulk draws: padded to a power of two, with the
/// lane width that indexes it.
#[derive(Clone, Debug)]
pub struct Alphabet {
    table: [u8; 128],
    size: usize,
    bits: u32,
}

impl Alphabet {
    /// `chars` must be non-empty ASCII, at most 128 characters.
    pub fn new(chars: &[u8]) -> Self {
        assert!(
            !chars.is_empty() && chars.len() <= 128 && chars.is_ascii(),
            "an alphabet is 1 to 128 ASCII characters"
        );
        let mut table = [chars[0]; 128];
        table[..chars.len()].copy_from_slice(chars);
        let bits = usize::BITS - (chars.len() - 1).leading_zeros();
        Alphabet {
            table,
            size: chars.len(),
            bits,
        }
    }

    /// Fills `out[..len]` from `rng`. `out` must hold `len + SLACK` bytes;
    /// past `len` it is scratch. A draw's lanes left over at `len` are
    /// dropped, so a shorter fill is a prefix of a longer one.
    #[inline]
    pub fn fill(&self, rng: &mut Xoshiro256PlusPlus, out: &mut [u8], len: usize) {
        assert!(out.len() >= len + SLACK, "a fill needs its slack");
        if self.size == 1 {
            out[..len].fill(self.table[0]);
            return;
        }
        match self.bits {
            1 => self.fill_lanes::<1>(rng, out, len),
            2 => self.fill_lanes::<2>(rng, out, len),
            3 => self.fill_lanes::<3>(rng, out, len),
            4 => self.fill_lanes::<4>(rng, out, len),
            5 => self.fill_lanes::<5>(rng, out, len),
            6 => self.fill_lanes::<6>(rng, out, len),
            _ => self.fill_lanes::<7>(rng, out, len),
        }
    }

    #[inline(always)]
    fn fill_lanes<const BITS: u32>(
        &self,
        rng: &mut Xoshiro256PlusPlus,
        out: &mut [u8],
        len: usize,
    ) {
        let lanes = (64 / BITS as usize).min(SLACK);
        let mask = (1u64 << BITS) - 1;
        let mut at = 0;
        while at < len {
            let mut x = rng.next_u64();
            let slots = &mut out[at..at + lanes];
            let mut kept = 0;
            for _ in 0..lanes {
                let lane = (x & mask) as usize;
                x >>= BITS;
                // `kept` is at most the lanes seen so far, so in range.
                slots[kept.min(lanes - 1)] = self.table[lane];
                kept += (lane < self.size) as usize;
            }
            at += kept;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn every_size_of_alphabet_fills_from_itself_evenly_and_as_a_prefix() {
        let source: Vec<u8> = (0u8..128).collect();
        for size in [1, 2, 3, 5, 26, 33, 62, 64, 65, 100, 128] {
            let chars = &source[..size];
            let alphabet = Alphabet::new(chars);
            let n = 2_000 * size; // about 2,000 of each, so 15% is many deviations
            let mut out = vec![0u8; n + SLACK];
            alphabet.fill(&mut Xoshiro256PlusPlus::seed_from_u64(1), &mut out, n);
            let mut counts = vec![0usize; size];
            for &b in &out[..n] {
                assert!(
                    (b as usize) < size,
                    "size {size}: {b} is not in the alphabet"
                );
                counts[b as usize] += 1;
            }
            let expected = n as f64 / size as f64;
            for (c, &count) in counts.iter().enumerate() {
                let ratio = count as f64 / expected;
                assert!(
                    (0.85..1.15).contains(&ratio),
                    "size {size}: character {c} appeared {count} times, about {expected:.0} expected"
                );
            }
            for len in [0, 1, 7, 63, 64, 65, 1000] {
                let mut short = vec![0u8; len + SLACK];
                alphabet.fill(&mut Xoshiro256PlusPlus::seed_from_u64(1), &mut short, len);
                assert_eq!(short[..len], out[..len], "size {size}, length {len}");
            }
        }
    }
}

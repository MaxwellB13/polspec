//! A keyed pseudo-random permutation of `[0, domain)`.
//!
//! A unique column's value at row `r` is `decode(permutation.apply(r))`:
//! distinct rows get distinct indices, so distinct values, and a row's value
//! depends on nothing but the row and the seed. So a column is unique over the
//! whole frame however it is batched, any window of rows is the whole
//! column's slice, chunks fill in parallel, and nothing is held but the output
//! -- no set of values already drawn, and no memory in proportion to anything.
//!
//! The permutation is a balanced Feistel network over the smallest even
//! number of bits that holds the domain. A Feistel network is a bijection on
//! `[0, 2^bits)` whatever its round function, and walking the cycle -- applying
//! it again to an index that lands past `domain` -- stays inside the cycle the
//! start is on, so the first index back in range is a bijection on
//! `[0, domain)` too. `bits` is rounded up to even, so `2^bits < 4 * domain`
//! and the walk takes fewer than four steps on average.

/// Rounds of the network. Four make a Feistel network a pseudo-random
/// permutation in theory; eight leave the output well spread for a round
/// function as cheap as the one below.
const ROUNDS: usize = 8;

pub struct Permutation {
    domain: u128,
    half: u32,
    mask: u128,
    keys: [u64; ROUNDS],
}

impl Permutation {
    /// A permutation of `[0, domain)`, keyed by `seed`. `domain` must be
    /// positive.
    pub fn new(domain: u128, seed: u64) -> Self {
        assert!(domain > 0, "a permutation needs a non-empty domain");
        let needed = 128 - (domain - 1).leading_zeros();
        let bits = needed.max(2).next_multiple_of(2);
        let half = bits / 2;
        let mask = if half >= 64 {
            u64::MAX as u128
        } else {
            (1u128 << half) - 1
        };
        let mut state = seed;
        let keys = std::array::from_fn(|_| {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            mix(state)
        });
        Permutation {
            domain,
            half,
            mask,
            keys,
        }
    }

    /// Where `index` goes. `index` must be below the domain.
    #[inline]
    pub fn apply(&self, index: u128) -> u128 {
        debug_assert!(
            index < self.domain,
            "{index} is outside [0, {})",
            self.domain
        );
        let mut x = index;
        loop {
            x = self.network(x);
            if x < self.domain {
                return x;
            }
        }
    }

    #[inline]
    fn network(&self, x: u128) -> u128 {
        let mut left = (x >> self.half) & self.mask;
        let mut right = x & self.mask;
        for key in &self.keys {
            // A half is at most 64 bits, so it fits the round function whole.
            let round = mix(right as u64 ^ key) as u128 & self.mask;
            (left, right) = (right, left ^ round);
        }
        (left << self.half) | right
    }
}

/// Positions `start..start + n` of the permutation of `[0, domain)` that
/// `seed` keys: the indices a window of rows takes, so any window is the
/// whole range's slice. Refuses a window that runs past the domain -- there
/// are not that many distinct indices -- naming both.
pub fn window(domain: u64, seed: u64, start: u64, n: u64) -> Result<Vec<u64>, String> {
    let end = start
        .checked_add(n)
        .filter(|end| *end <= domain)
        .ok_or_else(|| {
            format!(
                "rows {start}..{} need {} distinct indices, but the domain holds {domain}",
                start.saturating_add(n),
                start.saturating_add(n)
            )
        })?;
    if n == 0 {
        return Ok(Vec::new());
    }
    let permutation = Permutation::new(domain as u128, seed);
    Ok((start..end)
        .map(|i| permutation.apply(i as u128) as u64)
        .collect())
}

/// The splitmix64 finaliser: every input bit reaches every output bit.
#[inline]
fn mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn image(domain: u128, seed: u64) -> Vec<u128> {
        let p = Permutation::new(domain, seed);
        (0..domain).map(|i| p.apply(i)).collect()
    }

    #[test]
    fn it_is_a_bijection_on_every_small_domain() {
        for domain in (1..=300).chain([1_000, 4_095, 4_096, 4_097, 65_537]) {
            for seed in [0, 1, 42, u64::MAX] {
                let mut out = image(domain, seed);
                out.sort_unstable();
                assert!(
                    out.iter().copied().eq(0..domain),
                    "domain {domain}, seed {seed} is not a permutation"
                );
            }
        }
    }

    #[test]
    fn a_seed_is_a_different_permutation() {
        assert_ne!(image(1_000, 1), image(1_000, 2));
        assert_eq!(image(1_000, 7), image(1_000, 7));
    }

    #[test]
    fn it_is_not_close_to_the_identity() {
        let out = image(10_000, 3);
        let fixed = out
            .iter()
            .enumerate()
            .filter(|(i, v)| *i as u128 == **v)
            .count();
        assert!(fixed < 10, "{fixed} fixed points");
        let rising = out.windows(2).filter(|w| w[1] == w[0] + 1).count();
        assert!(rising < 10, "{rising} runs of consecutive outputs");
    }

    #[test]
    fn a_prefix_of_the_rows_spreads_over_the_domain() {
        // The first tenth of the rows, bucketed into tenths of the domain: a
        // chi-squared statistic well inside what 9 degrees of freedom allow.
        let domain = 100_000u128;
        let p = Permutation::new(domain, 11);
        let mut buckets = [0f64; 10];
        for r in 0..domain / 10 {
            buckets[(p.apply(r) * 10 / domain) as usize] += 1.0;
        }
        let expected = (domain / 100) as f64;
        let chi2: f64 = buckets
            .iter()
            .map(|b| (b - expected).powi(2) / expected)
            .sum();
        assert!(chi2 < 30.0, "chi-squared {chi2} over {buckets:?}");
    }

    #[test]
    fn a_window_is_the_whole_ranges_slice() {
        let whole = window(1_000, 9, 0, 1_000).unwrap();
        let mut sorted = whole.clone();
        sorted.sort_unstable();
        assert!(sorted.into_iter().eq(0..1_000));
        for (start, n) in [(0, 10), (17, 100), (990, 10), (500, 0)] {
            let part = window(1_000, 9, start, n).unwrap();
            assert_eq!(
                part,
                whole[start as usize..(start + n) as usize],
                "{start}+{n}"
            );
        }
    }

    #[test]
    fn a_window_past_the_domain_is_refused() {
        let err = window(100, 1, 90, 20).unwrap_err();
        assert!(
            err.contains("110 distinct") && err.contains("holds 100"),
            "{err}"
        );
        assert!(window(0, 1, 0, 0).unwrap().is_empty());
    }

    #[test]
    fn the_widest_domains_stay_in_range_and_distinct() {
        for domain in [1u128 << 64, u128::MAX, (1u128 << 100) + 3] {
            let p = Permutation::new(domain, 5);
            let out: HashSet<u128> = (0..10_000u128).map(|i| p.apply(i)).collect();
            assert_eq!(out.len(), 10_000);
            assert!(out.iter().all(|v| *v < domain));
        }
    }
}

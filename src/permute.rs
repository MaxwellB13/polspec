//! A keyed pseudo-random permutation of `[0, domain)`.
//!
//! A unique column's value at row `r` is `decode(permutation.apply(r))`:
//! distinct rows get distinct indices, so distinct values, and a row's value
//! depends on nothing but the row and the seed. So a column is unique over the
//! whole frame however it is batched, any window of rows is the whole
//! column's slice, chunks fill in parallel, and nothing is held but the output
//! -- no set of values already drawn, and no memory in proportion to anything.
//!
//! The permutation is a Feistel network over the fewest bits that hold the
//! domain, split into a high half of `ceil(bits / 2)` bits and a low half of
//! `floor(bits / 2)`. A Feistel network is a bijection on `[0, 2^bits)`
//! whatever its round function -- an unbalanced one too, as long as each
//! round's output is masked to the half it is mixed into -- and walking the
//! cycle -- applying it again to an index that lands past `domain` -- stays
//! inside the cycle the start is on, so the first index back in range is a
//! bijection on `[0, domain)` too. `2^bits < 2 * domain`, so the walk takes
//! fewer than two steps on average.

/// Rounds of the network: the four that make a Feistel network a
/// pseudo-random permutation in theory (Luby-Rackoff), with a round
/// function -- the splitmix64 finaliser -- that reaches every output bit
/// from every input bit. 1.x took eight.
const ROUNDS: usize = 4;

/// The indices `apply_run` takes through the network together.
const RUN: usize = 256;

pub struct Permutation {
    domain: u128,
    /// Width of the low half; the high half is `bits - low`, one more when
    /// `bits` is odd.
    low: u32,
    high: u32,
    keys: [u64; ROUNDS],
}

impl Permutation {
    /// A permutation of `[0, domain)`, keyed by `seed`. `domain` must be
    /// positive.
    pub fn new(domain: u128, seed: u64) -> Self {
        assert!(domain > 0, "a permutation needs a non-empty domain");
        let bits = (128 - (domain - 1).leading_zeros()).max(2);
        let low = bits / 2;
        let mut state = seed;
        let keys = std::array::from_fn(|_| {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            mix(state)
        });
        Permutation {
            domain,
            low,
            high: bits - low,
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
        if self.high <= 32 {
            return self.apply_narrow(index as u64) as u128;
        }
        self.apply_wide(index)
    }

    /// `apply` for a domain of at most 2^64, whose halves are at most 32
    /// bits: the same network in 64-bit arithmetic, so the same outputs at
    /// a fraction of the cost. `last` rather than the domain, which 2^64
    /// itself would overflow.
    #[inline]
    fn apply_narrow(&self, index: u64) -> u64 {
        let last = (self.domain - 1) as u64;
        let mut x = index;
        loop {
            x = self.network_narrow(x);
            if x <= last {
                return x;
            }
        }
    }

    /// `apply` of `first`, `first + 1`, ... into each slot of `out` in turn:
    /// the same outputs, computed a block at a time. Every index takes one
    /// pass of the network with no branch between them, so the rows overlap
    /// in the processor; the few that landed past the domain walk again
    /// together. `first + out.len()` must not pass the domain.
    pub fn apply_run(&self, first: u128, out: &mut [u128]) {
        if self.high > 32 {
            for (i, slot) in out.iter_mut().enumerate() {
                *slot = self.apply_wide(first + i as u128);
            }
            return;
        }
        let last = (self.domain - 1) as u64;
        let mut xs = [0u64; RUN];
        let mut strays = [0u16; RUN];
        for (b, block) in out.chunks_mut(RUN).enumerate() {
            let base = first as u64 + (b * RUN) as u64;
            let xs = &mut xs[..block.len()];
            for (i, x) in xs.iter_mut().enumerate() {
                *x = self.network_narrow(base + i as u64);
            }
            let mut k = 0;
            for (i, x) in xs.iter().enumerate() {
                strays[k] = i as u16;
                k += usize::from(*x > last);
            }
            while k > 0 {
                for &i in &strays[..k] {
                    xs[i as usize] = self.network_narrow(xs[i as usize]);
                }
                let mut kept = 0;
                for j in 0..k {
                    let i = strays[j];
                    strays[kept] = i;
                    kept += usize::from(xs[i as usize] > last);
                }
                k = kept;
            }
            for (slot, x) in block.iter_mut().zip(xs.iter()) {
                *slot = *x as u128;
            }
        }
    }

    /// One pass of the network in 64-bit arithmetic, for `apply_narrow` and
    /// `apply_run`.
    #[inline(always)]
    fn network_narrow(&self, x: u64) -> u64 {
        let low = self.low;
        let (low_mask, high_mask) = (mask64(low), mask64(self.high));
        // Rounds in pairs: the first mixes the low half into the high one,
        // the second the high into the low, so after each pair the halves
        // are back where they started.
        let mut left = x >> low;
        let mut right = x & low_mask;
        for [first, second] in self.keys.as_chunks::<2>().0 {
            left ^= mix(right ^ first) & high_mask;
            right ^= mix(left ^ second) & low_mask;
        }
        (left << low) | right
    }

    #[inline]
    fn apply_wide(&self, index: u128) -> u128 {
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
        let (low_mask, high_mask) = (mask128(self.low), mask128(self.high));
        let mut left = x >> self.low;
        let mut right = x & low_mask;
        for [first, second] in self.keys.as_chunks::<2>().0 {
            // A half is at most 64 bits, so it fits the round function whole.
            left ^= mix(right as u64 ^ first) as u128 & high_mask;
            right ^= mix(left as u64 ^ second) as u128 & low_mask;
        }
        (left << self.low) | right
    }
}

#[inline]
fn mask64(bits: u32) -> u64 {
    if bits >= 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    }
}

#[inline]
fn mask128(bits: u32) -> u128 {
    if bits >= 128 {
        u128::MAX
    } else {
        (1u128 << bits) - 1
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
    let mut out = vec![0u128; (end - start) as usize];
    permutation.apply_run(start as u128, &mut out);
    Ok(out.into_iter().map(|i| i as u64).collect())
}

/// Positions `start..start + n` of the permutation of a product space --
/// every combination of members with `sizes` values each -- that `seed`
/// keys, decoded into one index per member, the first member varying
/// slowest. Distinct positions give distinct combinations, so a composite
/// key drawn this way is distinct over the whole frame, and a window takes
/// the combinations the whole frame takes there.
pub fn composite_window(
    sizes: &[u64],
    seed: u64,
    start: u64,
    n: u64,
) -> Result<Vec<Vec<u64>>, String> {
    let domain = sizes
        .iter()
        .try_fold(1u128, |total, &size| total.checked_mul(size as u128))
        .ok_or("the members' combinations number more than 2^128")?;
    let end = (start as u128) + (n as u128);
    if end > domain {
        return Err(format!(
            "rows {start}..{end} need {end} distinct combinations, but the members take {domain}"
        ));
    }
    let mut members: Vec<Vec<u64>> = sizes
        .iter()
        .map(|_| Vec::with_capacity(n as usize))
        .collect();
    if n == 0 {
        return Ok(members);
    }
    let permutation = Permutation::new(domain, seed);
    let mut positions = vec![0u128; n as usize];
    permutation.apply_run(start as u128, &mut positions);
    for mut position in positions {
        for (member, &size) in members.iter_mut().zip(sizes).rev() {
            member.push((position % size as u128) as u64);
            position /= size as u128;
        }
    }
    Ok(members)
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

    #[test]
    fn a_composite_window_is_distinct_and_the_whole_spaces_slice() {
        let sizes = [3u64, 1, 7, 5];
        let all = composite_window(&sizes, 9, 0, 105).unwrap();
        let mut seen: Vec<(u64, u64, u64, u64)> = (0..105)
            .map(|i| (all[0][i], all[1][i], all[2][i], all[3][i]))
            .collect();
        for (k, member) in all.iter().enumerate() {
            assert!(member.iter().all(|&v| v < sizes[k]));
        }
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), 105, "every combination, once");
        let part = composite_window(&sizes, 9, 40, 30).unwrap();
        for k in 0..sizes.len() {
            assert_eq!(part[k], all[k][40..70]);
        }
        assert!(composite_window(&sizes, 9, 100, 6).is_err());
        // Two members of 2^64 values: a space past 2^64.
        let wide = composite_window(&[u64::MAX, u64::MAX], 3, 0, 1000).unwrap();
        assert_eq!(wide[0].len(), 1000);
    }
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
    fn neighbouring_rows_are_not_neighbouring_values() {
        // At four rounds, over an odd and an even bit width and a domain just
        // past a power of two (the longest walks): consecutive rows'
        // values are uncorrelated, and as rarely close as chance says.
        for domain in [1u128 << 20, (1 << 21) - 7, (1 << 17) + 1, 10_000_019] {
            for seed in [1, 99] {
                let p = Permutation::new(domain, seed);
                let rows = 200_000.min(domain) as usize;
                let v: Vec<f64> = (0..rows as u128)
                    .map(|r| p.apply(r) as f64 / domain as f64)
                    .collect();
                let mean = v.iter().sum::<f64>() / rows as f64;
                let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f64>();
                let cov: f64 = v.windows(2).map(|w| (w[0] - mean) * (w[1] - mean)).sum();
                let r = cov / var;
                assert!(
                    r.abs() < 5.0 / (rows as f64).sqrt(),
                    "domain {domain}, seed {seed}: serial correlation {r}"
                );
                // |a - b| < 1% of the domain: about 2% of independent pairs.
                let close = v.windows(2).filter(|w| (w[0] - w[1]).abs() < 0.01).count();
                let expected = 0.0199 * (rows - 1) as f64;
                assert!(
                    (close as f64 - expected).abs() < 6.0 * expected.sqrt(),
                    "domain {domain}, seed {seed}: {close} close pairs, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn a_walk_takes_under_two_steps_on_average() {
        // The worst case, a domain one past a power of two: 2^bits < 2 * domain.
        let domain = (1u128 << 20) + 1;
        let p = Permutation::new(domain, 3);
        let mut steps = 0u64;
        for r in 0..100_000u64 {
            let mut x = r;
            loop {
                steps += 1;
                let mut left = x >> p.low;
                let mut right = x & mask64(p.low);
                for [first, second] in p.keys.as_chunks::<2>().0 {
                    left ^= mix(right ^ first) & mask64(p.high);
                    right ^= mix(left ^ second) & mask64(p.low);
                }
                x = (left << p.low) | right;
                if (x as u128) < domain {
                    break;
                }
            }
            assert_eq!(x as u128, p.apply(r as u128));
        }
        assert!(
            steps < 200_000 * 101 / 100,
            "{steps} steps for 100,000 rows"
        );
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
    fn the_narrow_path_is_the_wide_one_bit_for_bit() {
        // Every domain width up to 2^64, each edge of it, several seeds: the
        // 64-bit network must give exactly what the 128-bit one gives, or a
        // unique column's seeded values would move.
        let mut domains: Vec<u128> = (1..=64)
            .flat_map(|b| {
                let p = 1u128 << b;
                [p - 1, p, p + 1]
            })
            .filter(|d| *d <= 1u128 << 64)
            .collect();
        domains.extend([1, 2, 3, 1_000, 65_537, 100_000_000]);
        for domain in domains {
            for seed in [0, 7, u64::MAX] {
                let p = Permutation::new(domain, seed);
                assert!(p.high <= 32, "domain {domain} takes the narrow path");
                let step = (domain / 997).max(1);
                let mut i = 0u128;
                while i < domain.min(5_000 * step) {
                    assert_eq!(
                        p.apply_narrow(i as u64) as u128,
                        p.apply_wide(i),
                        "domain {domain}, seed {seed}, index {i}"
                    );
                    i += step;
                }
                let last = domain - 1;
                assert_eq!(p.apply_narrow(last as u64) as u128, p.apply_wide(last));
            }
        }
    }

    #[test]
    fn a_run_is_apply_one_index_at_a_time() {
        for domain in [
            1u128,
            2,
            3,
            1_000,
            (1 << 20) + 1,
            10_000_000_000,
            1 << 64,
            (1 << 70) + 9,
        ] {
            let p = Permutation::new(domain, 17);
            let n = (RUN * 3 + 5).min(domain as usize);
            let first = (domain - n as u128) / 2;
            let mut out = vec![0u128; n];
            p.apply_run(first, &mut out);
            for (i, v) in out.iter().enumerate() {
                assert_eq!(*v, p.apply(first + i as u128), "domain {domain}, index {i}");
            }
        }
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

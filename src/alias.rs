//! Weighted categories by the alias method: one draw, one multiply and one
//! comparison a row, whatever the number of categories.
//!
//! Each of `n` buckets holds a capacity of `2^32`, so the table spreads a
//! total mass of `n * 2^32`. A category's mass is its weight's share of that
//! total, an integer; Vose's construction then fills every bucket to capacity
//! with at most two categories -- its own, below `cut`, and its `alias` from
//! there. Every step of the construction is integer arithmetic, and the masses
//! come from correctly rounded float operations in a fixed order, so every
//! platform builds the same table and a seed draws the same rows on each.
//!
//! A draw is one 64-bit value `x`. `x * n` as 128 bits splits it in two: the
//! high word is a bucket, uniform over `[0, n)`, and the low word is where in
//! that bucket `x` fell, uniform too; its top 32 bits pick the bucket's own
//! category or its alias. The bias of either part is below `n / 2^64`.
//!
//! A category's probability is held to within `1 / (n * 2^32)` of its
//! weight's share (the heaviest's to within `1 / 2^32`, as it takes what
//! rounding left over), so a positive weight below that share is never drawn.

use rand::Rng;

/// A bucket's capacity: `cut` is out of this.
const CAPACITY: u64 = 1 << 32;

/// One bucket: below `cut` it gives its own category, `pick[1]`, and from
/// there its alias, `pick[0]`. A bucket its own category fills has `cut` 0
/// and itself as its alias, so a cut always fits 32 bits.
///
/// The draw indexes `pick` by the coin rather than choosing between the two
/// in code: written as an `if` or as masks, LLVM makes a branch of it, and
/// a coin in a split bucket mispredicts about as often as it is right --
/// 3.9 ns a draw against 1.05 for the indexed load, at three categories.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Bucket {
    cut: u32,
    pick: [u32; 2],
}

#[cfg(test)]
impl Bucket {
    fn alias(&self) -> u32 {
        self.pick[0]
    }
}

#[derive(Clone, Debug)]
pub struct Alias {
    buckets: Vec<Bucket>,
}

impl Alias {
    /// The table for `weights`, which must be finite, non-negative and not
    /// all zero (`ColumnPlan::build` checks), and fewer than `2^32`.
    pub fn new(weights: &[f64]) -> Result<Self, String> {
        let n = weights.len();
        if n == 0 || n > u32::MAX as usize {
            return Err(format!("{n} categories cannot be weighted"));
        }
        // A plain left-to-right sum: the same bits on every platform.
        let total = weights.iter().fold(0.0f64, |acc, w| acc + w);
        if !total.is_finite() || total <= 0.0 {
            return Err("weights must sum to a positive finite total".to_string());
        }
        let mass_total = n as u64 * CAPACITY;
        // Each category's mass, rounded down; what rounding left over goes to
        // the heaviest category (the first, on a tie), so the masses sum to
        // the table's total exactly and a zero weight keeps a zero mass.
        // `n * 2^32` is exact as a float, and `/` and `*` round correctly.
        let scale = mass_total as f64;
        let mut mass: Vec<u64> = weights
            .iter()
            .map(|w| ((w / total) * scale).min(scale) as u64)
            .collect();
        let assigned: u64 = mass.iter().sum();
        let heaviest = (0..n)
            .reduce(|best, i| if weights[i] > weights[best] { i } else { best })
            .expect("n > 0");
        if assigned <= mass_total {
            mass[heaviest] += mass_total - assigned;
        } else {
            // Rounding can only push the sum over by a unit or so a category:
            // the heaviest holds at least a bucket's capacity to give.
            mass[heaviest] -= assigned - mass_total;
        }

        let mut buckets: Vec<Bucket> = (0..n as u32)
            .map(|i| Bucket {
                cut: 0,
                pick: [i, i],
            })
            .collect();
        let (mut small, mut large): (Vec<usize>, Vec<usize>) =
            (0..n).partition(|i| mass[*i] < CAPACITY);
        while let (Some(&s), Some(&l)) = (small.last(), large.last()) {
            small.pop();
            buckets[s] = Bucket {
                cut: mass[s] as u32,
                pick: [l as u32, s as u32],
            };
            // `l` lends `s` what it lacks, and may fall below capacity itself.
            mass[l] -= CAPACITY - mass[s];
            if mass[l] < CAPACITY {
                large.pop();
                small.push(l);
            }
        }
        // The unplaced masses always sum to capacity times their count, so
        // `small` empties first and every bucket left in `large` is exactly
        // full: its own category, as it already reads.
        debug_assert!(small.is_empty());
        debug_assert!(large.iter().all(|l| mass[*l] == CAPACITY));
        Ok(Alias { buckets })
    }

    #[inline(always)]
    pub fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> u32 {
        let product = rng.next_u64() as u128 * self.buckets.len() as u128;
        let bucket = (product >> 64) as usize;
        let within = ((product as u64) >> 32) as u32;
        let b = &self.buckets[bucket];
        b.pick[(within < b.cut) as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    /// Each category's probability as the table holds it: its own buckets'
    /// cuts and the rest of every bucket aliasing it, over the total mass.
    fn probabilities(table: &Alias) -> Vec<f64> {
        let n = table.buckets.len();
        let mut mass = vec![0u128; n];
        for (b, bucket) in table.buckets.iter().enumerate() {
            mass[b] += bucket.cut as u128;
            assert_eq!(bucket.pick[1] as usize, b, "a bucket's own category");
            mass[bucket.alias() as usize] += (CAPACITY - bucket.cut as u64) as u128;
        }
        let total = (n as u128 * CAPACITY as u128) as f64;
        mass.iter().map(|m| *m as f64 / total).collect()
    }

    /// The shares of `3 * 2^32` floor to 2576980377, 6442450944 and
    /// 3865470566; 0.5 takes the one unit left over, lends to 0.3's bucket
    /// and then 0.2's, and ends exactly full. A change here moves every
    /// weighted seed.
    #[test]
    fn the_table_for_three_weights_is_pinned() {
        let table = Alias::new(&[0.2, 0.5, 0.3]).unwrap();
        let pinned: Vec<(u32, u32)> = table.buckets.iter().map(|b| (b.cut, b.alias())).collect();
        assert_eq!(pinned, PINNED_THREE);
    }

    const PINNED_THREE: [(u32, u32); 3] = [(2_576_980_377, 1), (0, 1), (3_865_470_566, 1)];

    #[test]
    fn the_table_holds_every_weight_to_within_its_resolution() {
        let cases: Vec<Vec<f64>> = vec![
            vec![1.0],
            vec![0.2, 0.5, 0.3],
            vec![0.0, 0.0, 1.0],
            vec![1e-12, 1.0, 1e12],
            vec![1e300, 1e300, 1.0],
            (0..200).map(|i| 1.0 / (i + 1) as f64).collect(),
            (0..1_000).map(|i| (i % 7) as f64).collect(),
        ];
        for weights in cases {
            let table = Alias::new(&weights).unwrap();
            let total: f64 = weights.iter().sum();
            for (i, (p, w)) in probabilities(&table).iter().zip(&weights).enumerate() {
                let want = w / total;
                assert!(
                    (p - want).abs() <= 2.0 / CAPACITY as f64 + want * 1e-12,
                    "category {i}: {p} against {want}"
                );
                if *w == 0.0 {
                    assert_eq!(*p, 0.0, "a zero weight is never drawn");
                }
            }
        }
    }

    #[test]
    fn a_total_that_overflows_is_refused() {
        assert!(Alias::new(&[f64::MAX, f64::MAX]).is_err());
        assert!(Alias::new(&[]).is_err());
    }

    #[test]
    fn draws_follow_the_weights() {
        let weights: Vec<f64> = (0..50).map(|i| (i + 1) as f64).collect();
        let total: f64 = weights.iter().sum();
        let table = Alias::new(&weights).unwrap();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(9);
        let draws = 1_000_000;
        let mut counts = vec![0f64; weights.len()];
        for _ in 0..draws {
            counts[table.sample(&mut rng) as usize] += 1.0;
        }
        let chi2: f64 = counts
            .iter()
            .zip(&weights)
            .map(|(c, w)| {
                let e = draws as f64 * w / total;
                (c - e).powi(2) / e
            })
            .sum();
        // 49 degrees of freedom: well past the 99.9th percentile (85).
        assert!(chi2 < 100.0, "chi-squared {chi2}");
    }

    #[test]
    fn a_zero_weight_is_never_drawn() {
        let table = Alias::new(&[0.0, 3.0, 0.0, 1.0, 0.0]).unwrap();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(2);
        for _ in 0..100_000 {
            assert!(matches!(table.sample(&mut rng), 1 | 3));
        }
    }
}

//! A drawn order: each row's value of an ordered column from its position.
//!
//! A sorted column is a sample's order statistics, drawn here one row at a
//! time. Position `j` of `valid` takes `q = (j + u_j) / valid`, with `u_j`
//! uniform on `[0, 1)` from the seed and `j` alone. The strata are disjoint
//! and increasing, so `q` never decreases with `j`, and a window of
//! positions is the whole frame's there. `q` is then read off a sorted
//! table of the column's own draws (`floats`, `integers`, `indices`), or
//! placed in its own slot of a unique column's range (`int_slots`,
//! `float_slots`), which keeps the values distinct as well as in order.
//!
//! The table is drawn and sorted in Python, through the engine's ordinary
//! path, so every distribution, bound and clamp is the column's own.

use rayon::prelude::*;

use crate::permute::mix;

/// Positions a parallel map takes per task: below it a task costs more
/// than the work it would share.
const PAR_MIN: usize = 4_096;

#[inline]
fn order_key(seed: u64) -> u64 {
    mix(seed ^ 0x2D35_8DCC_AA6C_78A5)
}

/// `u_j`: uniform on `[0, 1)` for one position, a function of the key and
/// the position alone -- the top 53 bits of a mixed hash.
#[inline]
fn unit(key: u64, position: u64) -> f64 {
    let h = mix(key ^ mix(position.wrapping_add(0x9E37_79B9_7F4A_7C15)));
    (h >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
}

/// Where position `j` of `valid` sits along a table of `size` values: the
/// index at or below it, and how far it is past that index, in `[0, 1)`.
#[inline]
fn along(key: u64, j: u64, valid: u64, size: usize) -> (usize, f64) {
    let last = (size - 1) as f64;
    let at = ((j as f64 + unit(key, j)) / valid as f64 * last).clamp(0.0, last);
    let below = (at.floor() as usize).min(size - 1);
    (below, at - below as f64)
}

fn check(valid: u64, first: u64, count: u64) -> Result<(), String> {
    match first.checked_add(count) {
        Some(end) if end <= valid => Ok(()),
        _ => Err(format!(
            "positions {first}..{first}+{count} run past the {valid} drawn in order"
        )),
    }
}

fn positions<T: Send>(first: u64, count: u64, f: impl Fn(u64) -> T + Sync + Send) -> Vec<T> {
    (0..count as usize)
        .into_par_iter()
        .with_min_len(PAR_MIN)
        .map(|i| f(first + i as u64))
        .collect()
}

/// For a finite domain: the index of the table value at or below each
/// position `first..first + count` of `valid`.
pub fn indices(
    seed: u64,
    valid: u64,
    first: u64,
    count: u64,
    size: usize,
) -> Result<Vec<u64>, String> {
    check(valid, first, count)?;
    if size == 0 {
        return Err("a drawn order needs a table of at least one value".into());
    }
    // Each table value takes an equal share of the positions: the one whose
    // `1 / size` of `[0, 1)` holds `q`.
    let key = order_key(seed);
    Ok(positions(first, count, |j| {
        let q = (j as f64 + unit(key, j)) / valid as f64;
        ((q * size as f64) as u64).min(size as u64 - 1)
    }))
}

/// A float column: each position interpolated between its two neighbours
/// in the sorted `table`, and kept between them, so the values never
/// decrease.
pub fn floats(
    table: &[f64],
    seed: u64,
    valid: u64,
    first: u64,
    count: u64,
) -> Result<Vec<f64>, String> {
    check(valid, first, count)?;
    if table.is_empty() {
        return Err("a drawn order needs a table of at least one value".into());
    }
    let key = order_key(seed);
    let last = table.len() - 1;
    Ok(positions(first, count, |j| {
        let (below, frac) = along(key, j, valid, table.len());
        let (lo, hi) = (table[below], table[(below + 1).min(last)]);
        (lo + frac * (hi - lo)).clamp(lo, hi)
    }))
}

/// An integer, temporal or decimal column, in physical units: each
/// position stepped from the neighbour below towards the one above, by the
/// whole units its place between them reaches, so it stays an integer and
/// the values never decrease. Generic over the physical width -- `i64` for
/// most columns, `i128` for the rest -- and exact in either: a value lies
/// between two of the table's, so it fits wherever they do.
pub fn integers<T>(
    table: &[T],
    seed: u64,
    valid: u64,
    first: u64,
    count: u64,
) -> Result<Vec<T>, String>
where
    T: Copy + Send + Sync + Into<i128> + TryFrom<i128>,
{
    check(valid, first, count)?;
    if table.is_empty() {
        return Err("a drawn order needs a table of at least one value".into());
    }
    let key = order_key(seed);
    let last = table.len() - 1;
    Ok(positions(first, count, |j| {
        let (below, frac) = along(key, j, valid, table.len());
        let (lo, hi): (i128, i128) = (table[below].into(), table[(below + 1).min(last)].into());
        let gap = hi.abs_diff(lo);
        // A 128-bit integer and a float convert in software; nearly every
        // gap fits 64 bits, which convert in one instruction.
        let step = match u64::try_from(gap) {
            Ok(narrow) => ((frac * narrow as f64) as u64).min(narrow) as u128,
            Err(_) => ((frac * gap as f64) as u128).min(gap),
        };
        // Between `lo` and `hi`, both of which are `T`s.
        T::try_from(lo.wrapping_add_unsigned(step)).unwrap_or(table[below])
    }))
}

/// A unique integer, temporal or decimal column over `lo..=hi`, physical:
/// position `j` takes a uniform value in its own slot,
/// `[lo + j*S/valid, lo + (j+1)*S/valid)` with `S` the range's size, so
/// the values are distinct and increasing.
pub fn int_slots(
    lo: i128,
    hi: i128,
    seed: u64,
    valid: u64,
    first: u64,
    count: u64,
) -> Result<Vec<i128>, String> {
    check(valid, first, count)?;
    if hi < lo {
        return Err(format!("an empty range, {lo}..={hi}"));
    }
    // The range's size less one, which an i128's full width still fits.
    let span = hi.abs_diff(lo);
    if span < valid.saturating_sub(1) as u128 {
        return Err(format!(
            "a unique column over {lo}..={hi} holds {} values, fewer than the \
             {valid} asked for",
            span + 1
        ));
    }
    // `j * S / valid` without forming `j * S`, which can pass 2^128:
    // `S = q * valid + r`, so it is `j * q + j * r / valid`, and `j * r`
    // is below `valid^2`, which fits.
    let valid128 = valid as u128;
    let (q, r) = if span == u128::MAX {
        // `S` is 2^128 itself: compute from `span = S - 1`.
        let q = span / valid128;
        let r = span % valid128 + 1;
        if r == valid128 { (q + 1, 0) } else { (q, r) }
    } else {
        ((span + 1) / valid128, (span + 1) % valid128)
    };
    // Modulo 2^128: the last slot's end is `S`, which is 2^128 itself for
    // a full-width range, and every width is still exact.
    let offset = |j: u128| -> u128 { j.wrapping_mul(q).wrapping_add(j * r / valid128) };
    let key = order_key(seed);
    Ok(positions(first, count, |j| {
        let start = offset(j as u128);
        let width = offset(j as u128 + 1).wrapping_sub(start);
        let within = ((unit(key, j) * width as f64) as u128).min(width - 1);
        lo.wrapping_add_unsigned(start + within)
    }))
}

/// A unique float column over `[lo, hi]`: position `j` at its stratum's
/// own point of the range, `lo + (j + u_j) / valid * (hi - lo)`.
pub fn float_slots(
    lo: f64,
    hi: f64,
    seed: u64,
    valid: u64,
    first: u64,
    count: u64,
) -> Result<Vec<f64>, String> {
    check(valid, first, count)?;
    if lo.is_nan() || hi.is_nan() || lo > hi {
        return Err(format!("an empty range, {lo}..={hi}"));
    }
    let key = order_key(seed);
    Ok(positions(first, count, |j| {
        (lo + (j as f64 + unit(key, j)) / valid as f64 * (hi - lo)).clamp(lo, hi)
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn non_decreasing<T: PartialOrd>(values: &[T]) -> bool {
        values.windows(2).all(|w| w[0] <= w[1])
    }

    fn sorted_table(n: usize) -> Vec<f64> {
        let key = order_key(99);
        let mut t: Vec<f64> = (0..n as u64).map(|i| unit(key, i) * 10.0 - 3.0).collect();
        t.sort_by(f64::total_cmp);
        t
    }

    #[test]
    fn units_are_in_the_half_open_unit_interval() {
        let key = order_key(1);
        assert!((0..100_000).all(|j| (0.0..1.0).contains(&unit(key, j))));
    }

    #[test]
    fn floats_never_decrease_and_stay_in_the_table() {
        let table = sorted_table(1_000);
        let values = floats(&table, 5, 50_000, 0, 50_000).unwrap();
        assert!(non_decreasing(&values));
        assert!(values.iter().all(|v| *v >= table[0] && *v <= table[999]));
    }

    #[test]
    fn a_window_is_the_whole_frames_positions_there() {
        let table = sorted_table(500);
        let whole = floats(&table, 5, 20_000, 0, 20_000).unwrap();
        let window = floats(&table, 5, 20_000, 7_321, 999).unwrap();
        assert_eq!(window, whole[7_321..8_320]);
        let ints: Vec<i128> = table.iter().map(|v| (*v * 1e6) as i128).collect();
        let whole = integers(&ints, 5, 20_000, 0, 20_000).unwrap();
        assert_eq!(
            integers(&ints, 5, 20_000, 19_000, 1_000).unwrap(),
            whole[19_000..]
        );
        let narrow: Vec<i64> = ints.iter().map(|v| *v as i64).collect();
        let as_i64 = integers(&narrow, 5, 20_000, 0, 20_000).unwrap();
        assert!(as_i64.iter().zip(&whole).all(|(a, b)| *a as i128 == *b));
        let whole = indices(5, 20_000, 0, 20_000, 3).unwrap();
        assert_eq!(
            indices(5, 20_000, 4_000, 10, 3).unwrap(),
            whole[4_000..4_010]
        );
    }

    #[test]
    fn integers_never_decrease_and_stay_between_their_neighbours() {
        let table: Vec<i128> = vec![0, 0, 3, 10, 10, 11, i64::MAX as i128];
        let values = integers(&table, 2, 100_000, 0, 100_000).unwrap();
        assert!(non_decreasing(&values));
        assert!(values.iter().all(|v| (0..=i64::MAX as i128).contains(v)));
    }

    #[test]
    fn indices_cover_a_finite_domain_in_proportion() {
        let values = indices(3, 90_000, 0, 90_000, 9).unwrap();
        assert!(non_decreasing(&values));
        for k in 0..9 {
            let share = values.iter().filter(|v| **v == k).count();
            assert!((share as i64 - 10_000).abs() <= 1, "{k}: {share}");
        }
    }

    #[test]
    fn int_slots_are_distinct_increasing_and_in_range() {
        for (lo, hi, n) in [
            (0i128, 99, 100u64),
            (-5, 1_000_000, 70_000),
            (i64::MIN as i128, i64::MAX as i128, 50_000),
            (0, u64::MAX as i128, 50_000),
            (i128::MIN, i128::MAX, 50_000),
        ] {
            let values = int_slots(lo, hi, 9, n, 0, n).unwrap();
            assert!(values.windows(2).all(|w| w[0] < w[1]), "{lo}..={hi}");
            assert!(values[0] >= lo && values[n as usize - 1] <= hi);
            let window = int_slots(lo, hi, 9, n, 31, 17).unwrap();
            assert_eq!(window, values[31..48]);
        }
    }

    #[test]
    fn int_slots_refuse_a_range_too_small() {
        assert!(int_slots(0, 98, 1, 100, 0, 100).is_err());
        assert!(int_slots(0, 99, 1, 100, 0, 100).is_ok());
    }

    #[test]
    fn float_slots_are_increasing_and_in_range() {
        let values = float_slots(-1.0, 1.0, 4, 100_000, 0, 100_000).unwrap();
        assert!(values.windows(2).all(|w| w[0] < w[1]));
        assert!(values.iter().all(|v| (-1.0..=1.0).contains(v)));
    }

    #[test]
    fn positions_past_the_frame_are_refused() {
        assert!(floats(&[1.0], 1, 10, 5, 6).is_err());
        assert!(indices(1, 10, 0, 10, 0).is_err());
    }
}

//! Filling a column whose values must all differ.
//!
//! The ordinary generators draw each value independently, which is why a
//! `unique` column used to emit duplicates its own spec rejected. These draw
//! *without replacement* instead.
//!
//! An enumerable domain -- an integer range, a set of categories, the two
//! booleans -- is drawn one of two ways, and which one depends on how much
//! room the domain has over the `k` values wanted.
//!
//! With room to spare, values are drawn and rejected against a set. Nearly
//! every draw is new, so this costs about one draw per value, from a range the
//! sampler prepares once. `CROWDED` is the line: at `D = 8k` rejection expects
//! about 1.07 draws per value.
//!
//! Below that line rejection degrades -- it spends its time rediscovering
//! values it already holds -- so the draw switches to Floyd's algorithm, which
//! takes exactly `k` steps whatever the ratio, and holds only the values it
//! has chosen. That last part is the point. The obvious way to serve a crowded
//! domain is to materialise and shuffle it, and that allocates in proportion
//! to `D` rather than to `k`: ten million distinct values from a range of
//! eighty million used to reserve well over a gigabyte before writing anything.
//! Floyd's needs no more room than its own output.
//!
//! A float range and the strings of a given length range are not enumerable --
//! there are no offsets to draw -- so those two always reject, sharing one
//! helper with one budget between them.
//!
//! Nulls are exempt, as they are everywhere else in polspec: a null means "no
//! value", and repeating it is not repeating a value. So the null mask is
//! decided first and only the non-null rows draw from the domain.
//!
//! Unlike the ordinary generators, these fill a column in one pass rather than
//! in independently-seeded chunks: distinctness is a property of the whole
//! column, so it cannot be established chunk by chunk.

use std::collections::HashSet;
use std::hash::Hash;

use polars::prelude::*;
use polars_core::chunked_array::builder::{
    BooleanChunkedBuilder, PrimitiveChunkedBuilder, StringChunkedBuilder,
};
use rand::distr::{Bernoulli, Distribution as _, Uniform};
use rand::{RngExt, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::plan::{ColumnPlan, Kind};
use crate::sample::{CHARSET, random_ascii, seed_for_chunk};

/// Above this ratio of domain size to values wanted, draw and reject; at or
/// below it, use Floyd's. At `D = 8k` rejection expects about 1.07 draws per
/// value, which is where paying for a set lookup beats paying for Floyd's
/// varying-range draw and its shuffle.
const CROWDED: u128 = 8;

/// How many draws a rejection loop may take per value before giving up. A
/// domain wide enough to be rejected against needs barely more than one; this
/// only stops a domain that is secretly too small -- three-character strings,
/// a hair's breadth of float range -- from looping forever.
const MAX_DRAWS_PER_VALUE: usize = 64;

/// Which rows are null, and how many are not.
fn null_mask(
    plan: &ColumnPlan,
    n: usize,
    rng: &mut Xoshiro256PlusPlus,
) -> Result<(Vec<bool>, usize), String> {
    if !plan.nullable || plan.null_probability <= 0.0 {
        return Ok((vec![false; n], n));
    }
    // Built once, not once per row: `random_bool` constructs one of these on
    // every call.
    let bernoulli = Bernoulli::new(plan.null_probability).map_err(|e| {
        format!(
            "Invalid null_probability {} for column '{}': {e}",
            plan.null_probability, plan.name
        )
    })?;
    let mask: Vec<bool> = (0..n).map(|_| bernoulli.sample(rng)).collect();
    let wanted = mask.iter().filter(|is_null| !**is_null).count();
    Ok((mask, wanted))
}

fn too_small(plan: &ColumnPlan, wanted: usize, domain: u128) -> String {
    format!(
        "Column '{}' is unique, but its domain holds only {domain} distinct value(s) \
         and {wanted} are needed. Widen its bounds or choices, or generate fewer rows.",
        plan.name
    )
}

/// `wanted` distinct offsets into a domain of `domain` values, in random order.
///
/// Rejection while the domain has room, Floyd's algorithm once it does not;
/// see the module documentation for why the line falls where it does. Neither
/// branch allocates more than the output it returns.
fn distinct_offsets(
    plan: &ColumnPlan,
    wanted: usize,
    domain: u128,
    rng: &mut Xoshiro256PlusPlus,
) -> Result<Vec<u128>, String> {
    if wanted as u128 > domain {
        return Err(too_small(plan, wanted, domain));
    }
    if wanted == 0 {
        return Ok(Vec::new());
    }

    let mut seen: HashSet<u128> = HashSet::with_capacity(wanted);
    let mut out: Vec<u128> = Vec::with_capacity(wanted);

    if domain > CROWDED * wanted as u128 {
        // Roomy: one draw per value, over a range that does not change.
        for _ in 0..wanted.saturating_mul(MAX_DRAWS_PER_VALUE) {
            let candidate = rng.random_range(0..domain);
            if seen.insert(candidate) {
                out.push(candidate);
                if out.len() == wanted {
                    return Ok(out);
                }
            }
        }
        return Err(too_small(plan, wanted, domain));
    }

    // Crowded: for each `j` in the last `wanted` positions of the domain, draw
    // a candidate in `[0, j]` and take it if it is new, or take `j` itself if
    // it is not -- `j` cannot already be held, since every value taken so far
    // came from a strictly smaller range. That yields every `wanted`-sized
    // subset with equal probability, in exactly `wanted` steps.
    for j in (domain - wanted as u128)..domain {
        let candidate = rng.random_range(0..=j);
        out.push(if seen.insert(candidate) {
            candidate
        } else {
            seen.insert(j);
            j
        });
    }

    // Floyd's builds its subset biased toward increasing order, so it is
    // shuffled before being returned -- `wanted` swaps, not `domain`.
    for i in (1..out.len()).rev() {
        out.swap(i, rng.random_range(0..=i));
    }
    Ok(out)
}

/// `wanted` distinct values drawn by rejection, for a domain with no offsets.
///
/// `key` is what "distinct" means for the value type: a float's bit pattern,
/// a string's own text. Returns None when the budget runs out, which is the
/// caller's cue to explain what its domain could not supply.
fn distinct_by_rejection<T, K>(
    wanted: usize,
    mut draw: impl FnMut() -> T,
    key: impl Fn(&T) -> K,
) -> Option<Vec<T>>
where
    K: Eq + Hash,
{
    let mut seen: HashSet<K> = HashSet::with_capacity(wanted);
    let mut out: Vec<T> = Vec::with_capacity(wanted);
    for _ in 0..wanted.saturating_mul(MAX_DRAWS_PER_VALUE) {
        if out.len() == wanted {
            break;
        }
        let candidate = draw();
        if seen.insert(key(&candidate)) {
            out.push(candidate);
        }
    }
    (out.len() == wanted).then_some(out)
}

/// Spreads `values` over the non-null rows of `mask`.
macro_rules! place {
    ($builder:expr, $mask:expr, $values:expr) => {{
        let mut values = $values.into_iter();
        for is_null in $mask {
            if is_null {
                $builder.append_null();
            } else {
                $builder.append_value(values.next().expect("one value per non-null row"));
            }
        }
        $builder.finish()
    }};
}

macro_rules! impl_unique_int_column {
    ($fn_name:ident, $polars_type:ident, $native_type:ty, $default_min:expr, $default_max:expr) => {
        fn $fn_name(
            plan: &ColumnPlan,
            n: usize,
            seed: u64,
        ) -> Result<ChunkedArray<$polars_type>, String> {
            let name = PlSmallStr::from(plan.name.as_str());
            let mut builder = PrimitiveChunkedBuilder::<$polars_type>::new(name, n);
            if n == 0 {
                return Ok(builder.finish());
            }
            let clamp = |v: i128| -> $native_type {
                v.clamp(<$native_type>::MIN as i128, <$native_type>::MAX as i128) as $native_type
            };
            let lo = plan.min.map(|l| clamp(l.as_i128())).unwrap_or($default_min);
            let hi = plan.max.map(|l| clamp(l.as_i128())).unwrap_or($default_max);
            let (lo, hi) = if lo <= hi { (lo, hi) } else { (hi, lo) };
            let domain = (hi as i128 - lo as i128 + 1) as u128;

            let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed_for_chunk(seed, 0));
            let (mask, wanted) = null_mask(plan, n, &mut rng)?;
            let offsets = distinct_offsets(plan, wanted, domain, &mut rng)?;
            let values: Vec<$native_type> = offsets
                .into_iter()
                .map(|o| (lo as i128 + o as i128) as $native_type)
                .collect();
            Ok(place!(builder, mask, values))
        }
    };
}

impl_unique_int_column!(unique_int8, Int8Type, i8, i8::MIN, i8::MAX);
impl_unique_int_column!(unique_int16, Int16Type, i16, i16::MIN, i16::MAX);
impl_unique_int_column!(unique_int32, Int32Type, i32, i32::MIN, i32::MAX);
impl_unique_int_column!(unique_int64, Int64Type, i64, i64::MIN, i64::MAX);
impl_unique_int_column!(unique_uint8, UInt8Type, u8, u8::MIN, u8::MAX);
impl_unique_int_column!(unique_uint16, UInt16Type, u16, u16::MIN, u16::MAX);
impl_unique_int_column!(unique_uint32, UInt32Type, u32, u32::MIN, u32::MAX);
impl_unique_int_column!(unique_uint64, UInt64Type, u64, u64::MIN, u64::MAX);

macro_rules! impl_unique_float_column {
    ($fn_name:ident, $polars_type:ident, $native_type:ty, $default_bound:expr) => {
        fn $fn_name(
            plan: &ColumnPlan,
            n: usize,
            seed: u64,
        ) -> Result<ChunkedArray<$polars_type>, String> {
            let name = PlSmallStr::from(plan.name.as_str());
            let mut builder = PrimitiveChunkedBuilder::<$polars_type>::new(name, n);
            if n == 0 {
                return Ok(builder.finish());
            }
            let lo = plan
                .min
                .map(|l| l.as_f64() as $native_type)
                .unwrap_or(-$default_bound);
            let hi = plan
                .max
                .map(|l| l.as_f64() as $native_type)
                .unwrap_or($default_bound);
            let (lo, hi) = if lo <= hi { (lo, hi) } else { (hi, lo) };

            let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed_for_chunk(seed, 0));
            let (mask, wanted) = null_mask(plan, n, &mut rng)?;

            // A float range is not enumerable, so there is no domain to draw
            // offsets from: draw and reject on the bit pattern.
            let range = Uniform::new_inclusive(lo, hi).map_err(|e| {
                format!(
                    "Cannot sample column '{}' over the range [{lo}, {hi}]: {e}",
                    plan.name
                )
            })?;
            let values = distinct_by_rejection(
                wanted,
                || range.sample(&mut rng),
                |v: &$native_type| (*v as f64).to_bits(),
            )
            .ok_or_else(|| {
                format!(
                    "Column '{}' is unique, but {wanted} distinct value(s) could not be \
                     drawn from [{lo}, {hi}]. Widen its bounds, or generate fewer rows.",
                    plan.name
                )
            })?;
            Ok(place!(builder, mask, values))
        }
    };
}

impl_unique_float_column!(unique_float32, Float32Type, f32, 1_000_000.0);
impl_unique_float_column!(unique_float64, Float64Type, f64, 1_000_000.0);

fn unique_bool(plan: &ColumnPlan, n: usize, seed: u64) -> Result<BooleanChunked, String> {
    let name = PlSmallStr::from(plan.name.as_str());
    let mut builder = BooleanChunkedBuilder::new(name, n);
    if n == 0 {
        return Ok(builder.finish());
    }
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed_for_chunk(seed, 0));
    let (mask, wanted) = null_mask(plan, n, &mut rng)?;
    let offsets = distinct_offsets(plan, wanted, 2, &mut rng)?;
    let values: Vec<bool> = offsets.into_iter().map(|o| o == 1).collect();
    Ok(place!(builder, mask, values))
}

/// Distinct indices into a finite domain; Python gathers the typed values.
fn unique_index(plan: &ColumnPlan, n: usize, seed: u64) -> Result<UInt32Chunked, String> {
    let name = PlSmallStr::from(plan.name.as_str());
    let mut builder = PrimitiveChunkedBuilder::<UInt32Type>::new(name, n);
    if n == 0 {
        return Ok(builder.finish());
    }
    // Weights cannot bias a draw without replacement into anything meaningful
    // once the domain is barely larger than the sample, so they are ignored
    // here. Python refuses the combination before it reaches this point.
    let domain = plan.n_categories.unwrap_or(0) as u128;
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed_for_chunk(seed, 0));
    let (mask, wanted) = null_mask(plan, n, &mut rng)?;
    let offsets = distinct_offsets(plan, wanted, domain, &mut rng)?;
    let values: Vec<u32> = offsets.into_iter().map(|o| o as u32).collect();
    Ok(place!(builder, mask, values))
}

fn unique_string(plan: &ColumnPlan, n: usize, seed: u64) -> Result<StringChunked, String> {
    let name = PlSmallStr::from(plan.name.as_str());
    let mut builder = StringChunkedBuilder::new(name, n);
    if n == 0 {
        return Ok(builder.finish());
    }
    let min_len = plan.str_min_len;
    let max_len = plan.str_max_len.max(min_len);
    let len_dist = match max_len > min_len {
        true => Some(Uniform::new_inclusive(min_len, max_len).map_err(|e| {
            format!(
                "Cannot sample string lengths for column '{}' over [{min_len}, {max_len}]: {e}",
                plan.name
            )
        })?),
        false => None,
    };

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed_for_chunk(seed, 0));
    let (mask, wanted) = null_mask(plan, n, &mut rng)?;

    let mut scratch = vec![0u8; max_len];
    let values = distinct_by_rejection(
        wanted,
        || {
            let len = len_dist.as_ref().map_or(min_len, |d| d.sample(&mut rng));
            random_ascii(&mut rng, &mut scratch, len);
            // SAFETY: CHARSET holds only ASCII bytes.
            unsafe { std::str::from_utf8_unchecked(&scratch[..len]) }.to_owned()
        },
        |s: &String| s.clone(),
    )
    .ok_or_else(|| {
        format!(
            "Column '{}' is unique, but {wanted} distinct string(s) of length {min_len}..\
             {max_len} could not be drawn from a {}-character alphabet. Allow longer \
             strings, or generate fewer rows.",
            plan.name,
            CHARSET.len()
        )
    })?;
    Ok(place!(builder, mask, values))
}

/// Fills one column whose values must all differ.
pub fn generate_unique_series(plan: &ColumnPlan, n: usize, seed: u64) -> Result<Series, String> {
    Ok(match plan.kind {
        Kind::Int64 => unique_int64(plan, n, seed)?.into_series(),
        Kind::Int32 => unique_int32(plan, n, seed)?.into_series(),
        Kind::Int16 => unique_int16(plan, n, seed)?.into_series(),
        Kind::Int8 => unique_int8(plan, n, seed)?.into_series(),
        Kind::UInt64 => unique_uint64(plan, n, seed)?.into_series(),
        Kind::UInt32 => unique_uint32(plan, n, seed)?.into_series(),
        Kind::UInt16 => unique_uint16(plan, n, seed)?.into_series(),
        Kind::UInt8 => unique_uint8(plan, n, seed)?.into_series(),
        Kind::Float64 => unique_float64(plan, n, seed)?.into_series(),
        Kind::Float32 => unique_float32(plan, n, seed)?.into_series(),
        Kind::Bool => unique_bool(plan, n, seed)?.into_series(),
        Kind::String => unique_string(plan, n, seed)?.into_series(),
        Kind::Index => unique_index(plan, n, seed)?.into_series(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::{Limit, PlanArgs};

    fn plan(
        kind: &str,
        min: Option<Limit>,
        max: Option<Limit>,
        n_categories: Option<usize>,
    ) -> ColumnPlan {
        ColumnPlan::build(PlanArgs {
            name: "c".into(),
            kind,
            min,
            max,
            n_categories,
            unique: true,
            ..Default::default()
        })
        .expect("valid plan")
    }

    fn nullable(kind: &str, min: Option<Limit>, max: Option<Limit>, null_p: f64) -> ColumnPlan {
        ColumnPlan::build(PlanArgs {
            name: "c".into(),
            kind,
            nullable: true,
            null_probability: null_p,
            min,
            max,
            unique: true,
            ..Default::default()
        })
        .expect("valid plan")
    }

    fn distinct_count(s: &Series) -> usize {
        s.n_unique().expect("countable")
    }

    #[test]
    fn a_crowded_domain_is_drawn_without_stalling() {
        // 100 values from a domain of exactly 100: rejection sampling would
        // never finish, so this only terminates because Floyd's does not
        // rediscover values it already holds.
        let p = plan("int64", Some(Limit::Int(1)), Some(Limit::Int(100)), None);
        let s = generate_unique_series(&p, 100, 7).unwrap();
        assert_eq!(s.len(), 100);
        assert_eq!(distinct_count(&s), 100);
        let ca = s.i64().unwrap();
        assert_eq!(ca.min(), Some(1));
        assert_eq!(ca.max(), Some(100));
    }

    #[test]
    fn a_roomy_domain_still_gives_distinct_values() {
        let p = plan("int64", None, None, None);
        let s = generate_unique_series(&p, 10_000, 3).unwrap();
        assert_eq!(distinct_count(&s), 10_000);
    }

    #[test]
    fn a_domain_smaller_than_the_frame_names_the_column() {
        let p = plan("int64", Some(Limit::Int(1)), Some(Limit::Int(10)), None);
        let err = generate_unique_series(&p, 50, 1).unwrap_err();
        assert!(err.contains("'c' is unique"), "{err}");
        assert!(err.contains("10 distinct value(s)"), "{err}");
        assert!(err.contains("50 are needed"), "{err}");
    }

    #[test]
    fn every_kind_draws_without_replacement() {
        for (kind, min, max, cats) in [
            ("int8", Some(Limit::Int(-100)), Some(Limit::Int(100)), None),
            ("uint8", Some(Limit::UInt(0)), Some(Limit::UInt(200)), None),
            ("int16", None, None, None),
            ("uint16", None, None, None),
            ("int32", None, None, None),
            ("uint32", None, None, None),
            ("uint64", None, None, None),
            ("float64", None, None, None),
            ("float32", None, None, None),
            ("string", None, None, None),
            ("index", None, None, Some(500)),
        ] {
            let p = plan(kind, min, max, cats);
            let s = generate_unique_series(&p, 150, 11).unwrap_or_else(|e| panic!("{kind}: {e}"));
            assert_eq!(distinct_count(&s), 150, "{kind} repeated a value");
        }
    }

    #[test]
    fn a_whole_domain_is_covered_exactly_once() {
        // Asking for every value a domain holds is the tightest case Floyd's
        // has to get right: the result is a permutation, not a sample.
        let p = plan("index", None, None, Some(64));
        let s = generate_unique_series(&p, 64, 5).unwrap();
        let mut values: Vec<u32> = s.u32().unwrap().into_no_null_iter().collect();
        values.sort_unstable();
        assert_eq!(values, (0..64).collect::<Vec<u32>>());
    }

    #[test]
    fn offsets_are_not_returned_in_ascending_order() {
        // Floyd's builds the subset biased toward increasing order; without
        // the shuffle a unique column would arrive sorted, which is a
        // surprising thing for "random" data to be.
        let p = plan("int64", Some(Limit::Int(0)), Some(Limit::Int(9_999)), None);
        let s = generate_unique_series(&p, 1_000, 13).unwrap();
        let values: Vec<i64> = s.i64().unwrap().into_no_null_iter().collect();
        let mut sorted = values.clone();
        sorted.sort_unstable();
        assert_ne!(values, sorted, "the draw came back in ascending order");
    }

    #[test]
    fn a_bool_column_holds_at_most_its_two_values() {
        let p = plan("bool", None, None, None);
        let s = generate_unique_series(&p, 2, 5).unwrap();
        assert_eq!(distinct_count(&s), 2);
        assert!(generate_unique_series(&p, 3, 5).is_err());
    }

    #[test]
    fn nulls_repeat_but_values_do_not() {
        let p = nullable("int64", Some(Limit::Int(1)), Some(Limit::Int(60)), 0.5);
        let s = generate_unique_series(&p, 60, 9).unwrap();
        assert_eq!(s.len(), 60);
        assert!(s.null_count() > 0, "the null probability did nothing");
        let present = s.drop_nulls();
        assert_eq!(distinct_count(&present), present.len());
        // Nulls are exempt, so a domain of 60 covers 60 rows even though
        // fewer than 60 values are drawn.
        assert!(present.len() < 60);
    }

    #[test]
    fn the_same_seed_gives_the_same_column() {
        let p = plan("int64", Some(Limit::Int(1)), Some(Limit::Int(10_000)), None);
        let a = generate_unique_series(&p, 500, 42).unwrap();
        let b = generate_unique_series(&p, 500, 42).unwrap();
        assert!(a.equals(&b));
        let c = generate_unique_series(&p, 500, 43).unwrap();
        assert!(!a.equals(&c));
    }

    #[test]
    fn zero_rows_gives_an_empty_typed_column() {
        let p = plan("int64", None, None, None);
        let s = generate_unique_series(&p, 0, 1).unwrap();
        assert_eq!(s.len(), 0);
        assert_eq!(s.dtype(), &DataType::Int64);
    }
}

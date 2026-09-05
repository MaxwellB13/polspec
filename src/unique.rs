//! Filling a column whose values must all differ.
//!
//! The ordinary generators draw each value independently, which is why a
//! `unique` column used to emit duplicates its own spec rejected. These draw
//! *without replacement* instead.
//!
//! Two strategies, picked by how much room the domain has. When a domain of
//! size `D` is barely larger than the `n` values wanted, rejection sampling
//! would spend most of its time rediscovering values it already holds, so the
//! domain is materialised and partially shuffled -- Fisher-Yates, stopped
//! after `n` draws. When `D` is comfortably larger, materialising it would be
//! absurd (a `UInt64` column's domain does not fit in memory), so values are
//! drawn and rejected against a set instead. `CROWDED` is the line between the
//! two: at `D = 8n` rejection expects about 1.07 draws per value.
//!
//! Nulls are exempt, as they are everywhere else in polspec: a null means "no
//! value", and repeating it is not repeating a value. So the null mask is
//! decided first and only the non-null rows draw from the domain.
//!
//! Unlike the ordinary generators, these fill a column in one pass rather than
//! in independently-seeded chunks: distinctness is a property of the whole
//! column, so it cannot be established chunk by chunk.

use std::collections::HashSet;

use polars::prelude::*;
use polars_core::chunked_array::builder::{
    BooleanChunkedBuilder, PrimitiveChunkedBuilder, StringChunkedBuilder,
};
use rand::distributions::{Distribution as _, Uniform};
use rand::prelude::*;
use rand_xoshiro::Xoshiro256PlusPlus;
use rand_xoshiro::rand_core::SeedableRng;

use crate::plan::{ColumnPlan, Kind};
use crate::sample::{CHARSET, random_ascii, seed_for_chunk};

/// Above this ratio of domain size to values wanted, reject rather than shuffle.
const CROWDED: u128 = 8;

/// How many draws a rejection loop may take per value before giving up. A
/// domain wide enough for rejection needs barely more than one; this only
/// stops a domain that is secretly too small from looping forever.
const MAX_DRAWS_PER_VALUE: usize = 64;

/// Which rows are null, and how many are not.
fn null_mask(plan: &ColumnPlan, n: usize, rng: &mut Xoshiro256PlusPlus) -> (Vec<bool>, usize) {
    if !plan.nullable || plan.null_probability <= 0.0 {
        return (vec![false; n], n);
    }
    let mask: Vec<bool> = (0..n)
        .map(|_| rng.gen_bool(plan.null_probability))
        .collect();
    let wanted = mask.iter().filter(|is_null| !**is_null).count();
    (mask, wanted)
}

fn too_small(plan: &ColumnPlan, wanted: usize, domain: u128) -> String {
    format!(
        "Column '{}' is unique, but its domain holds only {domain} distinct value(s) \
         and {wanted} are needed. Widen its bounds or choices, or generate fewer rows.",
        plan.name
    )
}

/// `wanted` distinct offsets into a domain of `domain` values, in random order.
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

    if domain <= CROWDED * wanted as u128 {
        // Small enough to hold: shuffle the domain, keep the first `wanted`.
        // Bounded by CROWDED * wanted, so the allocation stays proportional to
        // the output rather than to the dtype's range.
        let mut pool: Vec<u128> = (0..domain).collect();
        for i in 0..wanted {
            let j = rng.gen_range(i as u128..domain) as usize;
            pool.swap(i, j);
        }
        pool.truncate(wanted);
        return Ok(pool);
    }

    let mut seen: HashSet<u128> = HashSet::with_capacity(wanted);
    let mut out = Vec::with_capacity(wanted);
    let budget = wanted.saturating_mul(MAX_DRAWS_PER_VALUE);
    for _ in 0..budget {
        let candidate = rng.gen_range(0..domain);
        if seen.insert(candidate) {
            out.push(candidate);
            if out.len() == wanted {
                return Ok(out);
            }
        }
    }
    Err(too_small(plan, wanted, domain))
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
            let (mask, wanted) = null_mask(plan, n, &mut rng);
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
            let (mask, wanted) = null_mask(plan, n, &mut rng);

            // A float range is not enumerable, so there is no domain to
            // shuffle: draw and reject on the bit pattern.
            let uniform = Uniform::new_inclusive(lo, hi);
            let mut seen: HashSet<u64> = HashSet::with_capacity(wanted);
            let mut values: Vec<$native_type> = Vec::with_capacity(wanted);
            let budget = wanted.saturating_mul(MAX_DRAWS_PER_VALUE);
            for _ in 0..budget {
                let candidate = uniform.sample(&mut rng);
                if seen.insert((candidate as f64).to_bits()) {
                    values.push(candidate);
                    if values.len() == wanted {
                        break;
                    }
                }
            }
            if values.len() < wanted {
                return Err(format!(
                    "Column '{}' is unique, but {wanted} distinct value(s) could not be \
                     drawn from [{lo}, {hi}]. Widen its bounds, or generate fewer rows.",
                    plan.name
                ));
            }
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
    let (mask, wanted) = null_mask(plan, n, &mut rng);
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
    let (mask, wanted) = null_mask(plan, n, &mut rng);
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
    let len_dist = (max_len > min_len).then(|| Uniform::new_inclusive(min_len, max_len));

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed_for_chunk(seed, 0));
    let (mask, wanted) = null_mask(plan, n, &mut rng);

    let mut seen: HashSet<String> = HashSet::with_capacity(wanted);
    let mut values: Vec<String> = Vec::with_capacity(wanted);
    let budget = wanted.saturating_mul(MAX_DRAWS_PER_VALUE);
    let mut scratch = vec![0u8; max_len];
    for _ in 0..budget {
        let len = len_dist.as_ref().map_or(min_len, |d| d.sample(&mut rng));
        random_ascii(&mut rng, &mut scratch, len);
        // SAFETY: CHARSET holds only ASCII bytes.
        let candidate = unsafe { std::str::from_utf8_unchecked(&scratch[..len]) };
        if !seen.contains(candidate) {
            let owned = candidate.to_owned();
            seen.insert(owned.clone());
            values.push(owned);
            if values.len() == wanted {
                break;
            }
        }
    }
    if values.len() < wanted {
        return Err(format!(
            "Column '{}' is unique, but {wanted} distinct string(s) of length {min_len}..\
             {max_len} could not be drawn from a {}-character alphabet. Allow longer \
             strings, or generate fewer rows.",
            plan.name,
            CHARSET.len()
        ));
    }
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
    use crate::plan::Limit;

    fn plan(
        kind: &str,
        min: Option<Limit>,
        max: Option<Limit>,
        n_categories: Option<usize>,
    ) -> ColumnPlan {
        ColumnPlan::build(
            "c".into(),
            kind,
            false,
            0.0,
            min,
            max,
            n_categories,
            None,
            None,
            None,
            None,
            None,
            true,
        )
        .expect("valid plan")
    }

    fn nullable(kind: &str, min: Option<Limit>, max: Option<Limit>, null_p: f64) -> ColumnPlan {
        ColumnPlan::build(
            "c".into(),
            kind,
            true,
            null_p,
            min,
            max,
            None,
            None,
            None,
            None,
            None,
            None,
            true,
        )
        .expect("valid plan")
    }

    fn distinct_count(s: &Series) -> usize {
        s.n_unique().expect("countable")
    }

    #[test]
    fn a_crowded_domain_is_shuffled_rather_than_rejected() {
        // 100 values from a domain of exactly 100: rejection would stall, so
        // this only terminates because the shuffle path takes it.
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

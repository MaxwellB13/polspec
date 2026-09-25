//! Filling a column whose values must all differ.
//!
//! A column with an enumerable value space -- an integer range, a set of
//! categories, the two booleans, a grid over a float range -- is filled
//! through a keyed permutation of that space (`crate::permute`): the value at
//! row `r` is the `π(r)`-th value of the space. Distinct rows get distinct
//! values, and a row's value depends only on the row and the seed, so the
//! column is unique over the whole frame, any window of rows is the whole
//! column's slice -- a batch of `generate_batches` included -- and chunks
//! fill in parallel with nothing held but the output.
//!
//! Strings and templated strings are still drawn by rejection against a set,
//! in one pass per call, until their value spaces are enumerated too; until
//! then they repeat from batch to batch, and Python says so.
//!
//! Nulls are exempt, as they are everywhere else in polspec: a null means "no
//! value", and repeating it is not repeating a value. The null mask is the
//! ordinary per-row draw, and a null row simply spends its index -- so a
//! permuted column's space has to hold every row, not only the non-null ones.

use std::collections::HashSet;
use std::hash::Hash;

use polars::prelude::*;
use polars_arrow::array::BooleanArray;
use polars_arrow::bitmap::Bitmap;
use polars_arrow::datatypes::ArrowDataType;
use polars_core::chunked_array::builder::StringChunkedBuilder;
use rand::SeedableRng;
use rand::distr::{Bernoulli, Distribution as _, Uniform};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;

use crate::permute::Permutation;
use crate::plan::{ColumnPlan, Kind};
use crate::sample::{
    CHARSET, CHUNK_SIZE, ColumnSeed, VALIDITY_BYTES_PER_CHUNK, chunk_rng, draws_nulls,
    null_bernoulli, random_ascii, seed_for_chunk,
};

/// How many draws a rejection loop may take per value before giving up. A
/// domain wide enough to be rejected against needs barely more than one; this
/// only stops a domain that is secretly too small -- three-character strings
/// -- from looping forever.
const MAX_DRAWS_PER_VALUE: usize = 64;

/// A float grid's points sit this many units in the last place apart at the
/// widest magnitude of its range, so rounding a point to the column's float
/// type can never merge two of them.
const GRID_ULPS: f64 = 4.0;

/// The most points a float grid holds: every integer below it is exact in an
/// `f64`, so the fraction each point sits at is too.
const MAX_GRID_POINTS: f64 = 9_007_199_254_740_992.0; // 2^53

fn too_small(plan: &ColumnPlan, wanted: u128, domain: u128) -> String {
    format!(
        "Column '{}' is unique, but its domain holds only {domain} distinct value(s) \
         and {wanted} are needed. Widen its bounds or choices, or generate fewer rows.",
        plan.name
    )
}

// ---------------------------------------------------------------------------
// The permuted kinds
// ---------------------------------------------------------------------------

/// Rows `[first_row, first_row + n)` of a column whose value at row `r` is
/// `decode(π(r))`, and their validity. `first_row` is where `seed`'s first
/// chunk starts, so the chunks, their null draws and the rows they permute
/// are the whole column's.
fn permuted_buffer<N, F>(
    plan: &ColumnPlan,
    n: usize,
    seed: ColumnSeed,
    domain: u128,
    decode: F,
) -> Result<(Vec<N>, Option<Bitmap>), String>
where
    N: Default + Copy + Send + Sync,
    F: Fn(u128) -> N + Sync,
{
    let first_row = seed.first_chunk * CHUNK_SIZE;
    let rows = (first_row + n) as u128;
    if rows > domain {
        return Err(too_small(plan, rows, domain));
    }
    let mut values: Vec<N> = vec![N::default(); n];
    if n == 0 {
        return Ok((values, None));
    }
    // Keyed apart from the chunk seeds the null mask draws from.
    let permutation = Permutation::new(domain, seed.base ^ 0x5EED_0F5A_BE11_A5E5);
    let at = |i: usize, row: usize| {
        decode(permutation.apply((first_row + i * CHUNK_SIZE + row) as u128))
    };

    if !draws_nulls(plan) {
        values
            .par_chunks_mut(CHUNK_SIZE)
            .enumerate()
            .for_each(|(i, slots)| {
                for (row, slot) in slots.iter_mut().enumerate() {
                    *slot = at(i, row);
                }
            });
        return Ok((values, None));
    }

    let bernoulli = null_bernoulli(plan)?;
    let mut validity: Vec<u8> = vec![0u8; n.div_ceil(8)];
    values
        .par_chunks_mut(CHUNK_SIZE)
        .zip(validity.par_chunks_mut(VALIDITY_BYTES_PER_CHUNK))
        .enumerate()
        .for_each(|(i, (slots, bits))| {
            let mut rng = chunk_rng(seed, i);
            for (row, slot) in slots.iter_mut().enumerate() {
                if bernoulli.sample(&mut rng) {
                    continue; // null: the row spends its index, the bit stays unset
                }
                bits[row / 8] |= 1 << (row % 8);
                *slot = at(i, row);
            }
        });
    Ok((values, Some(Bitmap::from_u8_vec(validity, n))))
}

fn permuted_column<T, F>(
    plan: &ColumnPlan,
    n: usize,
    seed: ColumnSeed,
    domain: u128,
    decode: F,
) -> Result<ChunkedArray<T>, String>
where
    T: PolarsNumericType,
    F: Fn(u128) -> T::Native + Sync,
{
    let (values, validity) = permuted_buffer(plan, n, seed, domain, decode)?;
    Ok(ChunkedArray::from_vec_validity(
        PlSmallStr::from(plan.name.as_str()),
        values,
        validity,
    ))
}

macro_rules! impl_unique_int_column {
    ($fn_name:ident, $polars_type:ident, $native_type:ty) => {
        fn $fn_name(
            plan: &ColumnPlan,
            n: usize,
            seed: ColumnSeed,
        ) -> Result<ChunkedArray<$polars_type>, String> {
            let clamp = |v: i128| -> i128 {
                v.clamp(<$native_type>::MIN as i128, <$native_type>::MAX as i128)
            };
            let lo = plan
                .min
                .map_or(<$native_type>::MIN as i128, |l| clamp(l.as_i128()));
            let hi = plan
                .max
                .map_or(<$native_type>::MAX as i128, |l| clamp(l.as_i128()));
            let (lo, hi) = if lo <= hi { (lo, hi) } else { (hi, lo) };
            let domain = (hi - lo + 1) as u128;
            permuted_column(plan, n, seed, domain, |i| (lo + i as i128) as $native_type)
        }
    };
}

impl_unique_int_column!(unique_int8, Int8Type, i8);
impl_unique_int_column!(unique_int16, Int16Type, i16);
impl_unique_int_column!(unique_int32, Int32Type, i32);
impl_unique_int_column!(unique_int64, Int64Type, i64);
impl_unique_int_column!(unique_uint8, UInt8Type, u8);
impl_unique_int_column!(unique_uint16, UInt16Type, u16);
impl_unique_int_column!(unique_uint32, UInt32Type, u32);
impl_unique_int_column!(unique_uint64, UInt64Type, u64);

/// The grid a unique float column draws from: as many evenly spaced points
/// over `[lo, hi]` as its widest magnitude keeps `GRID_ULPS` apart, capped at
/// 2^53, and the point at an index as a convex combination of the ends -- which
/// cannot overflow, whatever the range.
fn float_grid(lo: f64, hi: f64, epsilon: f64) -> (u128, impl Fn(u128) -> f64 + Sync) {
    let magnitude = lo.abs().max(hi.abs()).max(f64::MIN_POSITIVE);
    let half_span = hi * 0.5 - lo * 0.5;
    let spacing = GRID_ULPS * magnitude * epsilon;
    let points = if half_span > 0.0 {
        ((2.0 * (half_span / spacing))
            .floor()
            .min(MAX_GRID_POINTS - 1.0) as u128)
            + 1
    } else {
        1
    };
    let last = (points - 1).max(1) as f64;
    let point = move |i: u128| {
        let t = i as f64 / last;
        (lo * (1.0 - t) + hi * t).clamp(lo, hi)
    };
    (points, point)
}

macro_rules! impl_unique_float_column {
    ($fn_name:ident, $polars_type:ident, $native_type:ty, $default_bound:expr) => {
        fn $fn_name(
            plan: &ColumnPlan,
            n: usize,
            seed: ColumnSeed,
        ) -> Result<ChunkedArray<$polars_type>, String> {
            let lo = plan.min.map_or(-$default_bound, |l| l.as_f64());
            let hi = plan.max.map_or($default_bound, |l| l.as_f64());
            let (lo, hi) = if lo <= hi { (lo, hi) } else { (hi, lo) };
            let (domain, point) = float_grid(lo, hi, <$native_type>::EPSILON as f64);
            permuted_column(plan, n, seed, domain, |i| point(i) as $native_type)
        }
    };
}

impl_unique_float_column!(unique_float32, Float32Type, f32, 1_000_000.0);
impl_unique_float_column!(unique_float64, Float64Type, f64, 1_000_000.0);

fn unique_bool(plan: &ColumnPlan, n: usize, seed: ColumnSeed) -> Result<BooleanChunked, String> {
    let (values, validity) = permuted_buffer(plan, n, seed, 2, |i| i == 1)?;
    let mut packed = vec![0u8; n.div_ceil(8)];
    for (row, value) in values.iter().enumerate() {
        if *value {
            packed[row / 8] |= 1 << (row % 8);
        }
    }
    let array = BooleanArray::new(
        ArrowDataType::Boolean,
        Bitmap::from_u8_vec(packed, n),
        validity,
    );
    Ok(BooleanChunked::with_chunk(
        PlSmallStr::from(plan.name.as_str()),
        array,
    ))
}

/// Distinct indices into a finite domain; Python gathers the typed values.
///
/// Weights cannot bias a draw without replacement into anything meaningful
/// once the domain is barely larger than the sample, so they are ignored
/// here. Python refuses the combination before it reaches this point.
fn unique_index(plan: &ColumnPlan, n: usize, seed: ColumnSeed) -> Result<UInt32Chunked, String> {
    let domain = plan.n_categories.unwrap_or(0) as u128;
    permuted_column(plan, n, seed, domain, |i| i as u32)
}

// ---------------------------------------------------------------------------
// The rejected kinds: strings, until their value spaces are enumerated
// ---------------------------------------------------------------------------

/// Which rows are null, and how many are not.
fn null_mask(
    plan: &ColumnPlan,
    n: usize,
    rng: &mut Xoshiro256PlusPlus,
) -> Result<(Vec<bool>, usize), String> {
    if !plan.nullable || plan.null_probability <= 0.0 {
        return Ok((vec![false; n], n));
    }
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

/// `wanted` distinct values drawn by rejection, for a domain with no offsets.
///
/// `key` is what "distinct" means for the value type. Returns None when the
/// budget runs out, which is the caller's cue to explain what its domain
/// could not supply.
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

/// A templated string column drawn without replacement.
///
/// Rejection against a set, like `unique_string`. The refusal names the
/// template's own cardinality, so `format="uuid4"` can ask for a billion rows
/// and an `iso_country` column is told at 250 that there is nothing left.
fn unique_template(plan: &ColumnPlan, n: usize, seed: u64) -> Result<StringChunked, String> {
    let name = PlSmallStr::from(plan.name.as_str());
    let mut builder = StringChunkedBuilder::new(name, n);
    if n == 0 {
        return Ok(builder.finish());
    }
    let template = plan
        .template
        .as_ref()
        .ok_or_else(|| format!("Column '{}' has kind 'template' but no template", plan.name))?;
    let sampler = template.sampler();

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed_for_chunk(seed, 0));
    let (mask, wanted) = null_mask(plan, n, &mut rng)?;
    if (wanted as u128) > template.cardinality() {
        return Err(too_small(plan, wanted as u128, template.cardinality()));
    }

    let mut scratch = Vec::with_capacity(template.max_bytes());
    let values = distinct_by_rejection(
        wanted,
        || sampler.draw(&mut rng, &mut scratch),
        |s: &String| s.clone(),
    )
    .ok_or_else(|| too_small(plan, wanted as u128, template.cardinality()))?;
    Ok(place!(builder, mask, values))
}

/// Whether a unique column of `kind` is filled through the permutation -- so
/// a window of it is the whole column's slice -- or drawn per call by
/// rejection.
pub fn is_permuted(kind: Kind) -> bool {
    !matches!(kind, Kind::String | Kind::Template)
}

/// Rows of a permuted unique column, from where `seed`'s first chunk starts.
pub fn generate_permuted_series(
    plan: &ColumnPlan,
    n: usize,
    seed: ColumnSeed,
) -> Result<Series, String> {
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
        Kind::Index => unique_index(plan, n, seed)?.into_series(),
        Kind::String | Kind::Template => {
            return Err(format!(
                "Column '{}': a unique {} column is drawn by rejection, not permuted",
                plan.name,
                plan.kind.name()
            ));
        }
    })
}

/// A unique string or templated column, drawn by rejection in one pass per
/// call -- so, for now, the same rows in every batch.
pub fn generate_rejected_series(plan: &ColumnPlan, n: usize, seed: u64) -> Result<Series, String> {
    Ok(match plan.kind {
        Kind::String => unique_string(plan, n, seed)?.into_series(),
        Kind::Template => unique_template(plan, n, seed)?.into_series(),
        _ => {
            return Err(format!(
                "Column '{}': a unique {} column is permuted, not rejected",
                plan.name,
                plan.kind.name()
            ));
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::{Limit, PlanArgs};
    use crate::sample::generate_series;

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

    fn column(p: &ColumnPlan, n: usize, seed: u64) -> Result<Series, String> {
        generate_series(p, n, seed, 0)
    }

    fn distinct_count(s: &Series) -> usize {
        s.n_unique().expect("countable")
    }

    #[test]
    fn a_whole_domain_is_a_permutation_of_it() {
        // 100 values from a domain of exactly 100: every value, once.
        let p = plan("int64", Some(Limit::Int(1)), Some(Limit::Int(100)), None);
        let s = column(&p, 100, 7).unwrap();
        let mut values: Vec<i64> = s.i64().unwrap().into_no_null_iter().collect();
        values.sort_unstable();
        assert_eq!(values, (1..=100).collect::<Vec<i64>>());

        let p = plan("index", None, None, Some(64));
        let mut values: Vec<u32> = column(&p, 64, 5)
            .unwrap()
            .u32()
            .unwrap()
            .into_no_null_iter()
            .collect();
        values.sort_unstable();
        assert_eq!(values, (0..64).collect::<Vec<u32>>());
    }

    #[test]
    fn a_domain_smaller_than_the_frame_names_the_column() {
        let p = plan("int64", Some(Limit::Int(1)), Some(Limit::Int(10)), None);
        let err = column(&p, 50, 1).unwrap_err();
        assert!(err.contains("'c' is unique"), "{err}");
        assert!(err.contains("10 distinct value(s)"), "{err}");
        assert!(err.contains("50 are needed"), "{err}");
    }

    #[test]
    fn a_window_past_the_end_of_the_domain_is_refused_too() {
        // Rows 90..110 of a column whose domain holds 100: the window is
        // small, but the frame it is a window onto is not.
        let p = plan("int64", Some(Limit::Int(1)), Some(Limit::Int(100)), None);
        let err = generate_series(&p, 20, 1, 90).unwrap_err();
        assert!(err.contains("110 are needed"), "{err}");
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
            let s = column(&p, 150, 11).unwrap_or_else(|e| panic!("{kind}: {e}"));
            assert_eq!(distinct_count(&s), 150, "{kind} repeated a value");
        }
    }

    #[test]
    fn a_permuted_column_is_unique_across_chunks_and_a_window_is_its_slice() {
        let whole_n = CHUNK_SIZE * 3 + 500;
        for (kind, min, max, cats) in [
            ("int64", None, None, None),
            ("uint32", None, None, None),
            (
                "float64",
                Some(Limit::Float(0.0)),
                Some(Limit::Float(1.0)),
                None,
            ),
            ("float32", None, None, None),
            ("index", None, None, Some(whole_n + 10)),
        ] {
            let p = plan(kind, min, max, cats);
            let whole = column(&p, whole_n, 9).unwrap();
            assert_eq!(
                distinct_count(&whole),
                whole_n,
                "{kind} repeated across chunks"
            );
            for (offset, n) in [
                (0, 10),
                (17, 100),
                (CHUNK_SIZE - 3, 10),
                (CHUNK_SIZE + 1234, CHUNK_SIZE * 2 - 2000),
                (whole_n - 5, 5),
            ] {
                let window = generate_series(&p, n, 9, offset).unwrap();
                assert!(
                    window.equals_missing(&whole.slice(offset as i64, n)),
                    "{kind} at offset {offset} for {n} rows"
                );
            }
        }
    }

    #[test]
    fn a_narrow_float_range_still_gives_distinct_values_inside_it() {
        let p = plan(
            "float32",
            Some(Limit::Float(0.1)),
            Some(Limit::Float(0.1001)),
            None,
        );
        let s = column(&p, 50, 3).unwrap();
        assert_eq!(distinct_count(&s), 50);
        let ca = s.f32().unwrap();
        assert!(ca.min().unwrap() >= 0.1f32 && ca.max().unwrap() <= 0.1001f32);
    }

    #[test]
    fn values_are_not_returned_in_ascending_order() {
        let p = plan("int64", Some(Limit::Int(0)), Some(Limit::Int(9_999)), None);
        let s = column(&p, 1_000, 13).unwrap();
        let values: Vec<i64> = s.i64().unwrap().into_no_null_iter().collect();
        let mut sorted = values.clone();
        sorted.sort_unstable();
        assert_ne!(values, sorted, "the draw came back in ascending order");
    }

    #[test]
    fn a_bool_column_holds_at_most_its_two_values() {
        let p = plan("bool", None, None, None);
        let s = column(&p, 2, 5).unwrap();
        assert_eq!(distinct_count(&s), 2);
        assert!(column(&p, 3, 5).is_err());
    }

    #[test]
    fn nulls_repeat_but_values_do_not() {
        let p = nullable("int64", Some(Limit::Int(1)), Some(Limit::Int(60)), 0.5);
        let s = column(&p, 60, 9).unwrap();
        assert_eq!(s.len(), 60);
        assert!(s.null_count() > 0, "the null probability did nothing");
        let present = s.drop_nulls();
        assert_eq!(distinct_count(&present), present.len());
        // A null row spends its index, so the domain has to hold every row.
        assert!(column(&p, 61, 9).is_err());
    }

    #[test]
    fn the_same_seed_gives_the_same_column() {
        let p = plan("int64", Some(Limit::Int(1)), Some(Limit::Int(10_000)), None);
        let a = column(&p, 500, 42).unwrap();
        let b = column(&p, 500, 42).unwrap();
        assert!(a.equals(&b));
        let c = column(&p, 500, 43).unwrap();
        assert!(!a.equals(&c));
    }

    #[test]
    fn zero_rows_gives_an_empty_typed_column() {
        let p = plan("int64", None, None, None);
        let s = column(&p, 0, 1).unwrap();
        assert_eq!(s.len(), 0);
        assert_eq!(s.dtype(), &DataType::Int64);
    }
}

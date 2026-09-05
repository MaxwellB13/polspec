//! Filling a column with values.
//!
//! Nothing here touches Python: a `ColumnPlan` comes in, a Polars `Series`
//! goes out, so every generator is a plain function `cargo test` can run.
//!
//! A column is filled in fixed-size chunks, each seeded from the column seed
//! and its chunk index, so the output for a given seed is identical whatever
//! the thread count. The column seed itself is derived from the frame seed and
//! the column *name*, so inserting a column never reshuffles its neighbours.

use polars::prelude::*;
use polars_core::chunked_array::builder::{
    BooleanChunkedBuilder, PrimitiveChunkedBuilder, StringChunkedBuilder,
};
use rand::distributions::{Distribution as _, Uniform, WeightedIndex};
use rand::prelude::*;
use rand_xoshiro::Xoshiro256PlusPlus;
use rand_xoshiro::rand_core::SeedableRng;
use rayon::prelude::*;

use crate::dist::Distribution;
use crate::plan::{ColumnPlan, Kind};

pub const CHARSET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
const DEFAULT_WIDE_INT_BOUND: i64 = 1_000_000;
const DEFAULT_WIDE_UINT_BOUND: u64 = 1_000_000;
const DEFAULT_FLOAT_BOUND: f64 = 1_000_000.0;
const DEFAULT_FLOAT32_BOUND: f32 = 1_000_000.0;
pub const CHUNK_SIZE: usize = 65_536;

/// The seed one chunk of a column is filled from.
pub fn seed_for_chunk(base_seed: u64, chunk_index: usize) -> u64 {
    base_seed
        ^ (chunk_index as u64)
            .wrapping_mul(0x9E3779B97F4A7C15)
            .wrapping_add(0x1)
}

/// The seed one column is filled from: the frame seed mixed with the column
/// name (FNV-1a, then a splitmix64 finaliser), never with its position.
pub fn seed_for_column(base_seed: u64, name: &str) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in name.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    let mut z = base_seed ^ hash;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

/// Runs `fill(chunk_index, chunk_len)` over every chunk of `n` rows in parallel.
fn par_chunks<T, F>(n: usize, fill: F) -> Vec<T>
where
    T: Send,
    F: Fn(usize, usize) -> T + Sync,
{
    let n_chunks = n.div_ceil(CHUNK_SIZE).max(1);
    (0..n_chunks)
        .into_par_iter()
        .map(|i| {
            let start = i * CHUNK_SIZE;
            let end = (start + CHUNK_SIZE).min(n);
            fill(i, end - start)
        })
        .collect()
}

macro_rules! concat_chunks {
    ($parts:expr) => {{
        let mut parts = $parts;
        let mut result = parts.remove(0);
        for ca in parts {
            result
                .append_owned(ca)
                .expect("all chunks share the same dtype");
        }
        result
    }};
}

/// Appends `len` values from `sample`, replacing each with a null with the
/// plan's probability when the column is nullable.
macro_rules! fill_builder {
    ($builder:expr, $len:expr, $plan:expr, $rng:expr, $sample:expr) => {
        if $plan.nullable {
            let null_p = $plan.null_probability;
            for _ in 0..$len {
                if $rng.gen_bool(null_p) {
                    $builder.append_null();
                } else {
                    $builder.append_value($sample(&mut $rng));
                }
            }
        } else {
            for _ in 0..$len {
                $builder.append_value($sample(&mut $rng));
            }
        }
    };
}

macro_rules! impl_gen_int_column {
    ($fn_name:ident, $polars_type:ident, $native_type:ty, $default_min:expr, $default_max:expr) => {
        fn $fn_name(plan: &ColumnPlan, n: usize, seed: u64) -> ChunkedArray<$polars_type> {
            let name = PlSmallStr::from(plan.name.as_str());
            if n == 0 {
                return PrimitiveChunkedBuilder::<$polars_type>::new(name, 0).finish();
            }
            // A limit is clamped into the native range rather than rejected:
            // Python has already checked declared bounds against the dtype.
            let clamp = |v: i128| -> $native_type {
                v.clamp(<$native_type>::MIN as i128, <$native_type>::MAX as i128) as $native_type
            };
            let lo = plan.min.map(|l| clamp(l.as_i128())).unwrap_or($default_min);
            let hi = plan.max.map(|l| clamp(l.as_i128())).unwrap_or($default_max);
            let (lo, hi) = if lo <= hi { (lo, hi) } else { (hi, lo) };
            let uniform = Uniform::new_inclusive(lo, hi);
            let dist = plan.distribution;

            let parts = par_chunks(n, |i, len| {
                let mut builder = PrimitiveChunkedBuilder::<$polars_type>::new(name.clone(), len);
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed_for_chunk(seed, i));
                let sample = |rng: &mut Xoshiro256PlusPlus| -> $native_type {
                    match dist {
                        Distribution::Uniform => uniform.sample(rng),
                        _ => {
                            let val = dist.sample(rng);
                            let rounded = if val.is_nan() { lo as f64 } else { val.round() };
                            // The f64 clamp tames a sample of any magnitude; the
                            // integer clamp undoes the rounding `lo as f64` and
                            // `hi as f64` suffer near the ends of i64/u64.
                            (rounded.clamp(lo as f64, hi as f64) as $native_type).clamp(lo, hi)
                        }
                    }
                };
                fill_builder!(builder, len, plan, rng, sample);
                builder.finish()
            });
            concat_chunks!(parts)
        }
    };
}

impl_gen_int_column!(gen_int8_column, Int8Type, i8, i8::MIN, i8::MAX);
impl_gen_int_column!(gen_int16_column, Int16Type, i16, i16::MIN, i16::MAX);
impl_gen_int_column!(gen_int32_column, Int32Type, i32, i32::MIN, i32::MAX);
impl_gen_int_column!(
    gen_int64_column,
    Int64Type,
    i64,
    -DEFAULT_WIDE_INT_BOUND,
    DEFAULT_WIDE_INT_BOUND
);
impl_gen_int_column!(gen_uint8_column, UInt8Type, u8, u8::MIN, u8::MAX);
impl_gen_int_column!(gen_uint16_column, UInt16Type, u16, u16::MIN, u16::MAX);
impl_gen_int_column!(gen_uint32_column, UInt32Type, u32, u32::MIN, u32::MAX);
impl_gen_int_column!(
    gen_uint64_column,
    UInt64Type,
    u64,
    0,
    DEFAULT_WIDE_UINT_BOUND
);

macro_rules! impl_gen_float_column {
    ($fn_name:ident, $polars_type:ident, $native_type:ty, $default_bound:expr) => {
        fn $fn_name(plan: &ColumnPlan, n: usize, seed: u64) -> ChunkedArray<$polars_type> {
            let name = PlSmallStr::from(plan.name.as_str());
            if n == 0 {
                return PrimitiveChunkedBuilder::<$polars_type>::new(name, 0).finish();
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
            let uniform = Uniform::new_inclusive(lo, hi);
            let dist = plan.distribution;

            let parts = par_chunks(n, |i, len| {
                let mut builder = PrimitiveChunkedBuilder::<$polars_type>::new(name.clone(), len);
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed_for_chunk(seed, i));
                let sample = |rng: &mut Xoshiro256PlusPlus| -> $native_type {
                    match dist {
                        Distribution::Uniform => uniform.sample(rng),
                        _ => {
                            let val = dist.sample(rng) as $native_type;
                            if val.is_nan() { lo } else { val.clamp(lo, hi) }
                        }
                    }
                };
                fill_builder!(builder, len, plan, rng, sample);
                builder.finish()
            });
            concat_chunks!(parts)
        }
    };
}

impl_gen_float_column!(gen_float32_column, Float32Type, f32, DEFAULT_FLOAT32_BOUND);
impl_gen_float_column!(gen_float64_column, Float64Type, f64, DEFAULT_FLOAT_BOUND);

fn gen_bool_column(plan: &ColumnPlan, n: usize, seed: u64) -> BooleanChunked {
    let name = PlSmallStr::from(plan.name.as_str());
    if n == 0 {
        return BooleanChunkedBuilder::new(name, 0).finish();
    }
    let p_true = plan.p_true;
    let parts = par_chunks(n, |i, len| {
        let mut builder = BooleanChunkedBuilder::new(name.clone(), len);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed_for_chunk(seed, i));
        let sample = |rng: &mut Xoshiro256PlusPlus| -> bool { rng.gen_bool(p_true) };
        fill_builder!(builder, len, plan, rng, sample);
        builder.finish()
    });
    concat_chunks!(parts)
}

/// Draws category indices, uniformly or by weight.
#[derive(Clone)]
enum IndexSampler {
    Uniform(Uniform<u32>),
    Weighted(WeightedIndex<f64>),
}

impl IndexSampler {
    fn new(plan: &ColumnPlan) -> Result<Self, String> {
        let n = plan.n_categories.unwrap_or(0);
        if n == 0 {
            return Err(format!(
                "Column '{}' has kind 'index' with an empty domain",
                plan.name
            ));
        }
        match &plan.weights {
            Some(w) => WeightedIndex::new(w)
                .map(IndexSampler::Weighted)
                .map_err(|e| format!("Invalid weights for column '{}': {e}", plan.name)),
            None => Ok(IndexSampler::Uniform(Uniform::new(0, n as u32))),
        }
    }

    #[inline(always)]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> u32 {
        match self {
            IndexSampler::Uniform(u) => u.sample(rng),
            IndexSampler::Weighted(w) => w.sample(rng) as u32,
        }
    }
}

/// Indices into a finite domain; Python gathers the typed values.
fn gen_index_column(plan: &ColumnPlan, n: usize, seed: u64) -> Result<UInt32Chunked, String> {
    let name = PlSmallStr::from(plan.name.as_str());
    if n == 0 {
        return Ok(PrimitiveChunkedBuilder::<UInt32Type>::new(name, 0).finish());
    }
    let sampler = IndexSampler::new(plan)?;
    let parts = par_chunks(n, |i, len| {
        let mut builder = PrimitiveChunkedBuilder::<UInt32Type>::new(name.clone(), len);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed_for_chunk(seed, i));
        let sample = |rng: &mut Xoshiro256PlusPlus| -> u32 { sampler.sample(rng) };
        fill_builder!(builder, len, plan, rng, sample);
        builder.finish()
    });
    Ok(concat_chunks!(parts))
}

/// Writes a random alphanumeric string of `len` bytes into `scratch`.
#[inline(always)]
pub fn random_ascii(rng: &mut Xoshiro256PlusPlus, scratch: &mut [u8], len: usize) {
    let mut rand_val = rng.next_u64();
    let mut bits_left = 64;
    for slot in scratch.iter_mut().take(len) {
        if bits_left < 6 {
            rand_val = rng.next_u64();
            bits_left = 64;
        }
        let mut idx = (rand_val & 0x3F) as usize;
        rand_val >>= 6;
        bits_left -= 6;
        if idx >= CHARSET.len() {
            idx = (rng.next_u32() as usize) % CHARSET.len();
        }
        *slot = CHARSET[idx];
    }
}

fn gen_string_column(plan: &ColumnPlan, n: usize, seed: u64) -> StringChunked {
    let name = PlSmallStr::from(plan.name.as_str());
    if n == 0 {
        return StringChunkedBuilder::new(name, 0).finish();
    }
    let min_len = plan.str_min_len;
    let max_len = plan.str_max_len.max(min_len);
    let len_dist = (max_len > min_len).then(|| Uniform::new_inclusive(min_len, max_len));

    let parts = par_chunks(n, |i, len| {
        let mut builder = StringChunkedBuilder::new(name.clone(), len);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed_for_chunk(seed, i));
        let mut scratch: Vec<u8> = vec![0u8; max_len];
        let null_p = plan.null_probability;
        for _ in 0..len {
            if plan.nullable && rng.gen_bool(null_p) {
                builder.append_null();
                continue;
            }
            let str_len = len_dist
                .as_ref()
                .map_or(min_len, |dist| dist.sample(&mut rng));
            random_ascii(&mut rng, &mut scratch, str_len);
            // SAFETY: CHARSET contains only ASCII bytes (A-Z, a-z, 0-9).
            let s = unsafe { std::str::from_utf8_unchecked(&scratch[..str_len]) };
            builder.append_value(s);
        }
        builder.finish()
    });
    concat_chunks!(parts)
}

/// Fills one column according to its plan.
pub fn generate_series(plan: &ColumnPlan, n: usize, seed: u64) -> Result<Series, String> {
    if plan.unique {
        // Distinctness is a property of the whole column, so a unique column
        // is filled in one pass rather than in independent chunks.
        return crate::unique::generate_unique_series(plan, n, seed);
    }
    Ok(match plan.kind {
        Kind::Int64 => gen_int64_column(plan, n, seed).into_series(),
        Kind::Int32 => gen_int32_column(plan, n, seed).into_series(),
        Kind::Int16 => gen_int16_column(plan, n, seed).into_series(),
        Kind::Int8 => gen_int8_column(plan, n, seed).into_series(),
        Kind::UInt64 => gen_uint64_column(plan, n, seed).into_series(),
        Kind::UInt32 => gen_uint32_column(plan, n, seed).into_series(),
        Kind::UInt16 => gen_uint16_column(plan, n, seed).into_series(),
        Kind::UInt8 => gen_uint8_column(plan, n, seed).into_series(),
        Kind::Float64 => gen_float64_column(plan, n, seed).into_series(),
        Kind::Float32 => gen_float32_column(plan, n, seed).into_series(),
        Kind::Bool => gen_bool_column(plan, n, seed).into_series(),
        Kind::String => gen_string_column(plan, n, seed).into_series(),
        Kind::Index => gen_index_column(plan, n, seed)?.into_series(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::Limit;
    use std::collections::HashMap;

    #[allow(clippy::too_many_arguments)]
    fn plan(
        kind: &str,
        min: Option<Limit>,
        max: Option<Limit>,
        n_categories: Option<usize>,
        weights: Option<Vec<f64>>,
        distribution: Option<&str>,
        params: Option<HashMap<String, f64>>,
    ) -> ColumnPlan {
        ColumnPlan::build(
            "c".into(),
            kind,
            false,
            0.0,
            min,
            max,
            n_categories,
            weights,
            None,
            None,
            distribution,
            params.as_ref(),
            false,
        )
        .unwrap()
    }

    fn simple(kind: &str) -> ColumnPlan {
        plan(kind, None, None, None, None, None, None)
    }

    #[test]
    fn same_seed_same_values_across_chunk_boundaries() {
        let p = simple("int64");
        let n = CHUNK_SIZE * 2 + 17;
        let a = generate_series(&p, n, 42).unwrap();
        let b = generate_series(&p, n, 42).unwrap();
        assert!(a.equals(&b));
        assert_eq!(a.len(), n);
        let c = generate_series(&p, n, 43).unwrap();
        assert!(!a.equals(&c));
    }

    #[test]
    fn a_shorter_frame_is_a_prefix_of_a_longer_one() {
        // Chunk seeds depend on the chunk index alone, so the first rows of a
        // column never change when more rows are asked for.
        let p = simple("float64");
        let short = generate_series(&p, 1000, 7).unwrap();
        let long = generate_series(&p, CHUNK_SIZE + 1000, 7).unwrap();
        assert!(short.equals(&long.slice(0, 1000)));
    }

    #[test]
    fn integer_bounds_hold_at_the_extremes_of_i64_and_u64() {
        let lo = 9_007_199_254_740_990i64; // 2**53 - 2
        let hi = 9_007_199_254_740_999i64;
        let p = plan(
            "int64",
            Some(Limit::Int(lo)),
            Some(Limit::Int(hi)),
            None,
            None,
            None,
            None,
        );
        let s = generate_series(&p, 5000, 1).unwrap();
        let ca = s.i64().unwrap();
        assert_eq!(ca.min().unwrap(), lo);
        assert_eq!(ca.max().unwrap(), hi);

        let lo = u64::MAX - 15;
        let p = plan(
            "uint64",
            Some(Limit::UInt(lo)),
            Some(Limit::UInt(u64::MAX)),
            None,
            None,
            None,
            None,
        );
        let s = generate_series(&p, 5000, 1).unwrap();
        let ca = s.u64().unwrap();
        assert!(ca.min().unwrap() >= lo);
        assert!(
            ca.n_unique().unwrap() > 1,
            "a 16-value range must not collapse"
        );
    }

    #[test]
    fn a_float_limit_on_an_integer_column_truncates() {
        let p = plan(
            "int32",
            Some(Limit::Float(2.9)),
            Some(Limit::Float(2.9)),
            None,
            None,
            None,
            None,
        );
        let s = generate_series(&p, 10, 1).unwrap();
        assert_eq!(s.i32().unwrap().max().unwrap(), 2);
    }

    #[test]
    fn index_values_stay_inside_the_domain_and_follow_weights() {
        let p = plan("index", None, None, Some(4), None, None, None);
        let s = generate_series(&p, 10_000, 3).unwrap();
        let ca = s.u32().unwrap();
        assert_eq!(ca.min().unwrap(), 0);
        assert_eq!(ca.max().unwrap(), 3);

        let p = plan(
            "index",
            None,
            None,
            Some(3),
            Some(vec![0.0, 0.0, 1.0]),
            None,
            None,
        );
        let s = generate_series(&p, 1000, 3).unwrap();
        assert_eq!(s.u32().unwrap().min().unwrap(), 2);
    }

    #[test]
    fn nulls_appear_with_the_declared_probability() {
        let p = ColumnPlan::build(
            "c".into(),
            "int64",
            true,
            0.5,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            false,
        )
        .unwrap();
        let s = generate_series(&p, 20_000, 9).unwrap();
        let share = s.null_count() as f64 / 20_000.0;
        assert!((share - 0.5).abs() < 0.03, "null share {share}");
    }

    #[test]
    fn strings_respect_their_length_range() {
        let p = ColumnPlan::build(
            "c".into(),
            "string",
            false,
            0.0,
            None,
            None,
            None,
            None,
            Some(3),
            Some(6),
            None,
            None,
            false,
        )
        .unwrap();
        let s = generate_series(&p, 2000, 5).unwrap();
        let ca = s.str().unwrap();
        for v in (0..ca.len()).filter_map(|i| ca.get(i)) {
            assert!((3..=6).contains(&v.len()), "{v}");
            assert!(v.bytes().all(|b| CHARSET.contains(&b)));
        }
    }

    #[test]
    fn a_distribution_is_clamped_into_the_bounds() {
        let mut params = HashMap::new();
        params.insert("mean".to_string(), 1000.0);
        params.insert("std".to_string(), 1.0);
        let p = plan(
            "int64",
            Some(Limit::Int(0)),
            Some(Limit::Int(10)),
            None,
            None,
            Some("normal"),
            Some(params),
        );
        let s = generate_series(&p, 100, 1).unwrap();
        assert_eq!(s.i64().unwrap().max().unwrap(), 10);
    }

    #[test]
    fn column_seeds_depend_on_the_name_not_the_position() {
        assert_eq!(seed_for_column(1, "a"), seed_for_column(1, "a"));
        assert_ne!(seed_for_column(1, "a"), seed_for_column(1, "b"));
        assert_ne!(seed_for_column(1, "a"), seed_for_column(2, "a"));
        // Golden values pin the mapping: changing them is a breaking change.
        assert_eq!(
            seed_for_chunk(42, 3),
            42 ^ 3u64.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1)
        );
        assert_eq!(seed_for_column(42, "order_id"), 26_122_605_474_442_453);
    }

    #[test]
    fn zero_rows_gives_an_empty_typed_series() {
        for kind in ["int8", "uint64", "float32", "bool", "string"] {
            let s = generate_series(&simple(kind), 0, 1).unwrap();
            assert_eq!(s.len(), 0);
        }
        let p = plan("index", None, None, Some(2), None, None, None);
        assert_eq!(
            generate_series(&p, 0, 1).unwrap().dtype(),
            &DataType::UInt32
        );
    }
}

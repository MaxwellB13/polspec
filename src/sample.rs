//! Filling a column with values.
//!
//! Nothing here touches Python: a `ColumnPlan` comes in, a Polars `Series`
//! goes out, so every generator is a plain function `cargo test` can run.
//!
//! A column is one allocation, not a pile of them. The values buffer and the
//! validity bitmap are each allocated once at full length and then divided
//! into fixed-size chunks that are filled in parallel, every chunk writing
//! only its own disjoint slice. `CHUNK_SIZE` is a multiple of 8, which is what
//! lets the bitmap be split on a byte boundary alongside the values, so
//! neither buffer needs synchronising and neither needs stitching together
//! afterwards -- the column reaches Polars as a single chunk.
//!
//! Each chunk is seeded from the column seed and its chunk index, so the
//! output for a given seed is identical whatever the thread count, and the
//! first rows of a column never change when more rows are asked for. The
//! column seed itself is derived from the frame seed and the column *name*, so
//! inserting a column never reshuffles its neighbours.

use polars::prelude::*;
use polars_arrow::array::BooleanArray;
use polars_arrow::bitmap::Bitmap;
use polars_arrow::datatypes::ArrowDataType;
use polars_core::chunked_array::builder::StringChunkedBuilder;
use rand::distr::weighted::WeightedIndex;
use rand::distr::{Bernoulli, Distribution as _, Uniform};
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;

use crate::dist::Distribution;
use crate::plan::{ColumnPlan, Kind};

pub const CHARSET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
const DEFAULT_WIDE_INT_BOUND: i64 = 1_000_000;
const DEFAULT_WIDE_UINT_BOUND: u64 = 1_000_000;
const DEFAULT_FLOAT_BOUND: f64 = 1_000_000.0;
const DEFAULT_FLOAT32_BOUND: f32 = 1_000_000.0;
pub const CHUNK_SIZE: usize = 65_536;

/// Validity bytes covering exactly the rows of one chunk. The assertion below
/// is what makes that division exact -- without it a chunk would own a
/// fraction of a byte and two threads could write the same one.
const VALIDITY_BYTES_PER_CHUNK: usize = CHUNK_SIZE / 8;
const _: () = assert!(
    CHUNK_SIZE.is_multiple_of(8),
    "a chunk must cover a whole number of validity bytes"
);

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

/// The RNG one chunk draws from.
#[inline]
fn chunk_rng(seed: u64, chunk_index: usize) -> Xoshiro256PlusPlus {
    Xoshiro256PlusPlus::seed_from_u64(seed_for_chunk(seed, chunk_index))
}

/// Whether this column actually produces nulls.
///
/// `nullable` with a probability of zero never does, so it takes the cheaper
/// path: no bitmap allocated, and no Bernoulli draw per row.
fn draws_nulls(plan: &ColumnPlan) -> bool {
    plan.nullable && plan.null_probability > 0.0
}

/// The Bernoulli a nullable column decides each row with.
///
/// Built once per column rather than once per row: `Rng::random_bool`
/// constructs one of these on every call, which for a nullable column is a
/// float conversion and a `Result` per row of the frame.
fn null_bernoulli(plan: &ColumnPlan) -> Result<Bernoulli, String> {
    Bernoulli::new(plan.null_probability).map_err(|e| {
        format!(
            "Invalid null_probability {} for column '{}': {e}",
            plan.null_probability, plan.name
        )
    })
}

/// Fills `n` values and their validity, chunk by chunk in parallel.
///
/// One allocation for the values and one for the validity bytes; chunk `i`
/// owns `values[i * CHUNK_SIZE..]` and the validity bytes covering the same
/// rows, so the two are written without synchronisation and the result needs
/// no concatenation. A null row leaves its value slot at `N::default()`, which
/// the validity bitmap marks unset -- Polars never reads it.
///
/// `sample` is called exactly once per non-null row, in row order within a
/// chunk, which is what keeps a shorter frame a prefix of a longer one.
fn fill_buffer<N, F>(
    plan: &ColumnPlan,
    n: usize,
    seed: u64,
    sample: F,
) -> Result<(Vec<N>, Option<Bitmap>), String>
where
    N: Default + Copy + Send + Sync,
    F: Fn(&mut Xoshiro256PlusPlus) -> N + Sync,
{
    let mut values: Vec<N> = vec![N::default(); n];

    if !draws_nulls(plan) {
        values
            .par_chunks_mut(CHUNK_SIZE)
            .enumerate()
            .for_each(|(i, slots)| {
                let mut rng = chunk_rng(seed, i);
                for slot in slots.iter_mut() {
                    *slot = sample(&mut rng);
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
                    continue; // null: the value stays default and the bit unset
                }
                bits[row / 8] |= 1 << (row % 8);
                *slot = sample(&mut rng);
            }
        });
    Ok((values, Some(Bitmap::from_u8_vec(validity, n))))
}

/// A numeric column from one buffer, as the Polars type that owns it.
fn numeric_column<T, F>(
    plan: &ColumnPlan,
    n: usize,
    seed: u64,
    sample: F,
) -> Result<ChunkedArray<T>, String>
where
    T: PolarsNumericType,
    F: Fn(&mut Xoshiro256PlusPlus) -> T::Native + Sync,
{
    let (values, validity) = fill_buffer(plan, n, seed, sample)?;
    Ok(ChunkedArray::from_vec_validity(
        PlSmallStr::from(plan.name.as_str()),
        values,
        validity,
    ))
}

/// A `Uniform` over an already-ordered range, naming the column if it refuses.
fn uniform<X>(lo: X, hi: X, plan: &ColumnPlan) -> Result<Uniform<X>, String>
where
    X: rand::distr::uniform::SampleUniform + std::fmt::Display + Copy,
{
    Uniform::new_inclusive(lo, hi).map_err(|e| {
        format!(
            "Cannot sample column '{}' over the range [{lo}, {hi}]: {e}",
            plan.name
        )
    })
}

macro_rules! impl_gen_int_column {
    ($fn_name:ident, $polars_type:ident, $native_type:ty, $default_min:expr, $default_max:expr) => {
        fn $fn_name(
            plan: &ColumnPlan,
            n: usize,
            seed: u64,
        ) -> Result<ChunkedArray<$polars_type>, String> {
            // A limit is clamped into the native range rather than rejected:
            // Python has already checked declared bounds against the dtype.
            let clamp = |v: i128| -> $native_type {
                v.clamp(<$native_type>::MIN as i128, <$native_type>::MAX as i128) as $native_type
            };
            let lo = plan.min.map(|l| clamp(l.as_i128())).unwrap_or($default_min);
            let hi = plan.max.map(|l| clamp(l.as_i128())).unwrap_or($default_max);
            let (lo, hi) = if lo <= hi { (lo, hi) } else { (hi, lo) };
            let range = uniform(lo, hi, plan)?;
            let dist = plan.distribution;

            numeric_column::<$polars_type, _>(plan, n, seed, move |rng| match dist {
                Distribution::Uniform => range.sample(rng),
                _ => {
                    let val = dist.sample(rng);
                    let rounded = if val.is_nan() { lo as f64 } else { val.round() };
                    // The f64 clamp tames a sample of any magnitude; the
                    // integer clamp undoes the rounding `lo as f64` and
                    // `hi as f64` suffer near the ends of i64/u64.
                    (rounded.clamp(lo as f64, hi as f64) as $native_type).clamp(lo, hi)
                }
            })
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
        fn $fn_name(
            plan: &ColumnPlan,
            n: usize,
            seed: u64,
        ) -> Result<ChunkedArray<$polars_type>, String> {
            let lo = plan
                .min
                .map(|l| l.as_f64() as $native_type)
                .unwrap_or(-$default_bound);
            let hi = plan
                .max
                .map(|l| l.as_f64() as $native_type)
                .unwrap_or($default_bound);
            let (lo, hi) = if lo <= hi { (lo, hi) } else { (hi, lo) };
            let range = uniform(lo, hi, plan)?;
            let dist = plan.distribution;

            numeric_column::<$polars_type, _>(plan, n, seed, move |rng| match dist {
                Distribution::Uniform => range.sample(rng),
                _ => {
                    let val = dist.sample(rng) as $native_type;
                    if val.is_nan() { lo } else { val.clamp(lo, hi) }
                }
            })
        }
    };
}

impl_gen_float_column!(gen_float32_column, Float32Type, f32, DEFAULT_FLOAT32_BOUND);
impl_gen_float_column!(gen_float64_column, Float64Type, f64, DEFAULT_FLOAT_BOUND);

/// A boolean column, values and validity both bit-packed in one pass.
///
/// The same byte-per-chunk division as `fill_buffer`, except that the values
/// are a bitmap too, so a chunk owns one byte range in each of the two.
fn gen_bool_column(plan: &ColumnPlan, n: usize, seed: u64) -> Result<BooleanChunked, String> {
    let name = PlSmallStr::from(plan.name.as_str());
    let n_bytes = n.div_ceil(8);
    let mut values: Vec<u8> = vec![0u8; n_bytes];
    let truth = Bernoulli::new(plan.p_true).map_err(|e| {
        format!(
            "Invalid boolean probability {} for column '{}': {e}",
            plan.p_true, plan.name
        )
    })?;

    let validity = if draws_nulls(plan) {
        let bernoulli = null_bernoulli(plan)?;
        let mut validity: Vec<u8> = vec![0u8; n_bytes];
        values
            .par_chunks_mut(VALIDITY_BYTES_PER_CHUNK)
            .zip(validity.par_chunks_mut(VALIDITY_BYTES_PER_CHUNK))
            .enumerate()
            .for_each(|(i, (vals, bits))| {
                let mut rng = chunk_rng(seed, i);
                let rows = (vals.len() * 8).min(n.saturating_sub(i * CHUNK_SIZE));
                for row in 0..rows {
                    if bernoulli.sample(&mut rng) {
                        continue;
                    }
                    bits[row / 8] |= 1 << (row % 8);
                    if truth.sample(&mut rng) {
                        vals[row / 8] |= 1 << (row % 8);
                    }
                }
            });
        Some(Bitmap::from_u8_vec(validity, n))
    } else {
        values
            .par_chunks_mut(VALIDITY_BYTES_PER_CHUNK)
            .enumerate()
            .for_each(|(i, vals)| {
                let mut rng = chunk_rng(seed, i);
                let rows = (vals.len() * 8).min(n.saturating_sub(i * CHUNK_SIZE));
                for row in 0..rows {
                    if truth.sample(&mut rng) {
                        vals[row / 8] |= 1 << (row % 8);
                    }
                }
            });
        None
    };

    let array = BooleanArray::new(
        ArrowDataType::Boolean,
        Bitmap::from_u8_vec(values, n),
        validity,
    );
    Ok(BooleanChunked::with_chunk(name, array))
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
            Some(w) => WeightedIndex::new(w.clone())
                .map(IndexSampler::Weighted)
                .map_err(|e| format!("Invalid weights for column '{}': {e}", plan.name)),
            None => Ok(IndexSampler::Uniform(uniform(0, n as u32 - 1, plan)?)),
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
    let sampler = IndexSampler::new(plan)?;
    numeric_column::<UInt32Type, _>(plan, n, seed, move |rng| sampler.sample(rng))
}

/// One character index, drawn without bias, for the rare six-bit draw that
/// lands past the end of the alphabet.
///
/// Kept out of line on purpose. Two draws in sixty-four reach it, and inlining
/// it costs far more than it saves: an inner rejection loop around the whole
/// per-character step measured four times slower over a column of twenty
/// million strings, because the straight-line body below stops being
/// straight-line. `#[cold]` says the same thing to the optimiser.
///
/// `bound` is the largest multiple of the alphabet size that fits in a `u32`,
/// so `% len` over an accepted draw is exactly uniform. It rejects four values
/// out of 2^32, which is to say never.
#[cold]
#[inline(never)]
fn unbiased_index(rng: &mut Xoshiro256PlusPlus) -> usize {
    let len = CHARSET.len() as u32;
    let bound = (u32::MAX / len) * len;
    loop {
        let draw = rng.next_u32();
        if draw < bound {
            return (draw % len) as usize;
        }
    }
}

/// Writes a random alphanumeric string of `len` bytes into `scratch`.
///
/// Six bits a character, which covers a 62-character alphabet with two values
/// to spare; those two go to `unbiased_index` rather than being folded back on
/// to the alphabet, so no character is likelier than another.
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
            idx = unbiased_index(rng);
        }
        *slot = CHARSET[idx];
    }
}

/// A string column, built one chunk at a time.
///
/// Strings are the one kind that cannot be written into a preallocated buffer:
/// a row's width is not known until it is drawn, and Polars backs a string
/// column with view arrays rather than one contiguous run of bytes. So each
/// chunk builds its own array in parallel and the column keeps them.
///
/// `from_chunk_iter` rather than appending the chunks one by one: an append
/// rescans both sides for their first and last non-null value to maintain the
/// sorted flag, so hundreds of chunks mean hundreds of passes over the data to
/// maintain a flag a column of random strings will not have set anyway.
///
/// Merging the chunks into one was tried and reverted. It copies no string
/// bytes -- view arrays concatenate by merging their buffer lists -- but it
/// does copy the views, sixteen bytes a row, and at twenty million rows that
/// costs appreciably more than leaving the column split ever did.
fn gen_string_column(plan: &ColumnPlan, n: usize, seed: u64) -> Result<StringChunked, String> {
    let name = PlSmallStr::from(plan.name.as_str());
    if n == 0 {
        return Ok(StringChunkedBuilder::new(name, 0).finish());
    }
    let min_len = plan.str_min_len;
    let max_len = plan.str_max_len.max(min_len);
    let len_dist = match max_len > min_len {
        true => Some(uniform(min_len, max_len, plan)?),
        false => None,
    };
    let bernoulli = draws_nulls(plan)
        .then(|| null_bernoulli(plan))
        .transpose()?;

    let chunks: Vec<StringChunked> = (0..n.div_ceil(CHUNK_SIZE))
        .into_par_iter()
        .map(|i| {
            let start = i * CHUNK_SIZE;
            let len = (start + CHUNK_SIZE).min(n) - start;
            let mut builder = StringChunkedBuilder::new(name.clone(), len);
            let mut rng = chunk_rng(seed, i);
            let mut scratch: Vec<u8> = vec![0u8; max_len];
            for _ in 0..len {
                if bernoulli.as_ref().is_some_and(|b| b.sample(&mut rng)) {
                    builder.append_null();
                    continue;
                }
                let str_len = len_dist.as_ref().map_or(min_len, |d| d.sample(&mut rng));
                random_ascii(&mut rng, &mut scratch, str_len);
                // SAFETY: CHARSET contains only ASCII bytes (A-Z, a-z, 0-9).
                let s = unsafe { std::str::from_utf8_unchecked(&scratch[..str_len]) };
                builder.append_value(s);
            }
            builder.finish()
        })
        .collect();

    Ok(StringChunked::from_chunk_iter(
        name,
        chunks.iter().map(|ca| {
            ca.downcast_iter()
                .next()
                .expect("one chunk per part")
                .clone()
        }),
    ))
}

/// A templated string column: each row filled from the plan's `Template`.
///
/// Chunked and collected the same way as `gen_string_column`, and for the
/// same reasons; the only difference is where the bytes come from.
fn gen_template_column(plan: &ColumnPlan, n: usize, seed: u64) -> Result<StringChunked, String> {
    let name = PlSmallStr::from(plan.name.as_str());
    if n == 0 {
        return Ok(StringChunkedBuilder::new(name, 0).finish());
    }
    let template = plan
        .template
        .as_ref()
        .ok_or_else(|| format!("Column '{}' has kind 'template' but no template", plan.name))?;
    let sampler = template.sampler();
    let bernoulli = draws_nulls(plan)
        .then(|| null_bernoulli(plan))
        .transpose()?;

    let chunks: Vec<StringChunked> = (0..n.div_ceil(CHUNK_SIZE))
        .into_par_iter()
        .map(|i| {
            let start = i * CHUNK_SIZE;
            let len = (start + CHUNK_SIZE).min(n) - start;
            let mut builder = StringChunkedBuilder::new(name.clone(), len);
            let mut rng = chunk_rng(seed, i);
            let mut scratch: Vec<u8> = Vec::with_capacity(template.max_bytes());
            for _ in 0..len {
                if bernoulli.as_ref().is_some_and(|b| b.sample(&mut rng)) {
                    builder.append_null();
                    continue;
                }
                sampler.fill(&mut rng, &mut scratch);
                // SAFETY: every template part is ASCII or a `String`, so the
                // buffer is valid UTF-8 -- see `format::Template::compile`.
                let s = unsafe { std::str::from_utf8_unchecked(&scratch) };
                builder.append_value(s);
            }
            builder.finish()
        })
        .collect();

    Ok(StringChunked::from_chunk_iter(
        name,
        chunks.iter().map(|ca| {
            ca.downcast_iter()
                .next()
                .expect("one chunk per part")
                .clone()
        }),
    ))
}

/// Fills one column according to its plan.
pub fn generate_series(plan: &ColumnPlan, n: usize, seed: u64) -> Result<Series, String> {
    if plan.unique {
        // Distinctness is a property of the whole column, so a unique column
        // is filled in one pass rather than in independent chunks.
        return crate::unique::generate_unique_series(plan, n, seed);
    }
    Ok(match plan.kind {
        Kind::Int64 => gen_int64_column(plan, n, seed)?.into_series(),
        Kind::Int32 => gen_int32_column(plan, n, seed)?.into_series(),
        Kind::Int16 => gen_int16_column(plan, n, seed)?.into_series(),
        Kind::Int8 => gen_int8_column(plan, n, seed)?.into_series(),
        Kind::UInt64 => gen_uint64_column(plan, n, seed)?.into_series(),
        Kind::UInt32 => gen_uint32_column(plan, n, seed)?.into_series(),
        Kind::UInt16 => gen_uint16_column(plan, n, seed)?.into_series(),
        Kind::UInt8 => gen_uint8_column(plan, n, seed)?.into_series(),
        Kind::Float64 => gen_float64_column(plan, n, seed)?.into_series(),
        Kind::Float32 => gen_float32_column(plan, n, seed)?.into_series(),
        Kind::Bool => gen_bool_column(plan, n, seed)?.into_series(),
        Kind::String => gen_string_column(plan, n, seed)?.into_series(),
        Kind::Template => gen_template_column(plan, n, seed)?.into_series(),
        Kind::Index => gen_index_column(plan, n, seed)?.into_series(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::{Limit, PlanArgs};
    use std::collections::HashMap;

    fn simple(kind: &str) -> ColumnPlan {
        ColumnPlan::build(PlanArgs {
            name: "c".into(),
            kind,
            ..Default::default()
        })
        .unwrap()
    }

    fn bounded(kind: &str, min: Limit, max: Limit) -> ColumnPlan {
        ColumnPlan::build(PlanArgs {
            name: "c".into(),
            kind,
            min: Some(min),
            max: Some(max),
            ..Default::default()
        })
        .unwrap()
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
    fn a_fixed_width_column_arrives_as_one_chunk() {
        // The whole point of filling into one buffer: nothing downstream --
        // a gather, a cast, a write -- pays for a column split 40 ways.
        let n = CHUNK_SIZE * 40 + 3;
        for kind in ["int64", "float32", "bool"] {
            let s = generate_series(&simple(kind), n, 3).unwrap();
            assert_eq!(s.n_chunks(), 1, "{kind} came back split");
            assert_eq!(s.len(), n, "{kind}");
        }
        let nullable = ColumnPlan::build(PlanArgs {
            name: "c".into(),
            kind: "int64",
            nullable: true,
            null_probability: 0.3,
            ..Default::default()
        })
        .unwrap();
        assert_eq!(generate_series(&nullable, n, 3).unwrap().n_chunks(), 1);
    }

    #[test]
    fn a_string_column_stays_chunked() {
        // Deliberate, and the opposite of every other kind: merging a view
        // array copies sixteen bytes of view per row, which costs more than
        // the split it removes. Pinned so the merge is not quietly re-added.
        let n = CHUNK_SIZE * 40 + 3;
        let s = generate_series(&simple("string"), n, 3).unwrap();
        assert_eq!(s.len(), n);
        assert!(s.n_chunks() > 1, "a long string column is left split");
    }

    #[test]
    fn integer_bounds_hold_at_the_extremes_of_i64_and_u64() {
        let lo = 9_007_199_254_740_990i64; // 2**53 - 2
        let hi = 9_007_199_254_740_999i64;
        let p = bounded("int64", Limit::Int(lo), Limit::Int(hi));
        let s = generate_series(&p, 5000, 1).unwrap();
        let ca = s.i64().unwrap();
        assert_eq!(ca.min().unwrap(), lo);
        assert_eq!(ca.max().unwrap(), hi);

        let lo = u64::MAX - 15;
        let p = bounded("uint64", Limit::UInt(lo), Limit::UInt(u64::MAX));
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
        let p = bounded("int32", Limit::Float(2.9), Limit::Float(2.9));
        let s = generate_series(&p, 10, 1).unwrap();
        assert_eq!(s.i32().unwrap().max().unwrap(), 2);
    }

    #[test]
    fn index_values_stay_inside_the_domain_and_follow_weights() {
        let p = ColumnPlan::build(PlanArgs {
            name: "c".into(),
            kind: "index",
            n_categories: Some(4),
            ..Default::default()
        })
        .unwrap();
        let s = generate_series(&p, 10_000, 3).unwrap();
        let ca = s.u32().unwrap();
        assert_eq!(ca.min().unwrap(), 0);
        assert_eq!(ca.max().unwrap(), 3);

        let p = ColumnPlan::build(PlanArgs {
            name: "c".into(),
            kind: "index",
            n_categories: Some(3),
            weights: Some(vec![0.0, 0.0, 1.0]),
            ..Default::default()
        })
        .unwrap();
        let s = generate_series(&p, 1000, 3).unwrap();
        assert_eq!(s.u32().unwrap().min().unwrap(), 2);
    }

    #[test]
    fn nulls_appear_with_the_declared_probability() {
        let p = ColumnPlan::build(PlanArgs {
            name: "c".into(),
            kind: "int64",
            nullable: true,
            null_probability: 0.5,
            ..Default::default()
        })
        .unwrap();
        let s = generate_series(&p, 20_000, 9).unwrap();
        let share = s.null_count() as f64 / 20_000.0;
        assert!((share - 0.5).abs() < 0.03, "null share {share}");
    }

    #[test]
    fn a_nullable_column_with_no_nulls_declared_produces_none() {
        // The zero-probability path skips the bitmap entirely; it still has to
        // produce a column of the right length with every value present.
        let p = ColumnPlan::build(PlanArgs {
            name: "c".into(),
            kind: "int64",
            nullable: true,
            null_probability: 0.0,
            ..Default::default()
        })
        .unwrap();
        let s = generate_series(&p, 5_000, 4).unwrap();
        assert_eq!(s.null_count(), 0);
        assert_eq!(s.len(), 5_000);
    }

    #[test]
    fn strings_respect_their_length_range() {
        let p = ColumnPlan::build(PlanArgs {
            name: "c".into(),
            kind: "string",
            str_min_len: Some(3),
            str_max_len: Some(6),
            ..Default::default()
        })
        .unwrap();
        let s = generate_series(&p, 2000, 5).unwrap();
        let ca = s.str().unwrap();
        for v in (0..ca.len()).filter_map(|i| ca.get(i)) {
            assert!((3..=6).contains(&v.len()), "{v}");
            assert!(v.bytes().all(|b| CHARSET.contains(&b)));
        }
    }

    #[test]
    fn every_character_of_the_alphabet_is_reachable() {
        // The 6-bit draw has to reject 62 and 63 rather than fold them back
        // onto the alphabet, which would make two characters twice as likely.
        let p = ColumnPlan::build(PlanArgs {
            name: "c".into(),
            kind: "string",
            str_min_len: Some(20),
            str_max_len: Some(20),
            ..Default::default()
        })
        .unwrap();
        let s = generate_series(&p, 5_000, 11).unwrap();
        let ca = s.str().unwrap();
        let mut counts = [0usize; 62];
        for v in (0..ca.len()).filter_map(|i| ca.get(i)) {
            for b in v.bytes() {
                counts[CHARSET.iter().position(|c| *c == b).unwrap()] += 1;
            }
        }
        let expected = (5_000 * 20) as f64 / 62.0;
        for (index, count) in counts.iter().enumerate() {
            let ratio = *count as f64 / expected;
            assert!(
                (0.85..1.15).contains(&ratio),
                "character {:?} appeared {count} times, expected about {expected:.0}",
                CHARSET[index] as char
            );
        }
    }

    #[test]
    fn a_distribution_is_clamped_into_the_bounds() {
        let params = HashMap::from([("mean".to_string(), 1000.0), ("std".to_string(), 1.0)]);
        let p = ColumnPlan::build(PlanArgs {
            name: "c".into(),
            kind: "int64",
            min: Some(Limit::Int(0)),
            max: Some(Limit::Int(10)),
            distribution: Some("normal"),
            params: Some(&params),
            ..Default::default()
        })
        .unwrap();
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
            assert_eq!(s.len(), 0, "{kind}");
        }
        let p = ColumnPlan::build(PlanArgs {
            name: "c".into(),
            kind: "index",
            n_categories: Some(2),
            ..Default::default()
        })
        .unwrap();
        assert_eq!(
            generate_series(&p, 0, 1).unwrap().dtype(),
            &DataType::UInt32
        );
    }
}

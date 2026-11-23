//! Filling a column with values.
//!
//! Nothing here touches Python: a `ColumnPlan` comes in, a Polars `Series`
//! goes out, so every generator is a plain function `cargo test` can run.
//!
//! A column is one allocation, not a pile of them. The values buffer and the
//! validity bitmap are each allocated once at full length and then divided
//! into tasks that are filled in parallel, every task writing only its own
//! disjoint slice. Neither buffer needs synchronising and neither needs
//! stitching together afterwards -- the column reaches Polars as a single
//! chunk.
//!
//! Two sizes, kept apart on purpose (PLAN-2.0.0.md, S1):
//!
//! - **What is seeded**: a *seed chunk* of `SEED_CHUNK` rows draws from its
//!   own generator, seeded from the column seed and the chunk's index. This
//!   alone decides the output: the first rows of a column never change when
//!   more rows are asked for, and a window onto the frame regenerates at most
//!   one partial seed chunk.
//! - **What is scheduled**: a *task* is a run of whole seed chunks, sized
//!   from the frame and the thread count (`task_rows`), filled on one thread,
//!   in order. A small frame gets many tasks, so it fills on every core; a
//!   large one gets tasks of up to `MAX_TASK_ROWS`, so it pays no more per
//!   task than it ever did. Since each seed chunk's generator is its own, the
//!   task size never reaches the output: a seed gives one frame on one thread
//!   or thirty-two.
//!
//! A nullable column draws its nulls from a stream of their own, one
//! generator per seed chunk, and its values from the value stream for every
//! row, null or not. So a column's values do not move when its null rate
//! does, and the value loop has no branch in it.
//!
//! The column seed is derived from the frame seed and the column *name* --
//! or the `seed_name` a spec declares in its place -- so inserting a column
//! never reshuffles its neighbours, and renaming one need not change it.

use polars::prelude::*;
use polars_arrow::array::{BooleanArray, Utf8ViewArray, View};
use polars_arrow::bitmap::Bitmap;
use polars_arrow::datatypes::ArrowDataType;
use polars_core::chunked_array::builder::StringChunkedBuilder;
use rand::distr::{Bernoulli, Distribution as _, Uniform};
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;

use crate::alias::Alias;
use crate::chars::{Alphabet, SLACK};
use crate::dist::with_sampler;
use crate::plan::{ColumnPlan, Kind, Limit};
use crate::samplers::Portable;
use crate::views::Views;

pub const CHARSET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
const DEFAULT_WIDE_INT_BOUND: i64 = 1_000_000;
const DEFAULT_WIDE_UINT_BOUND: u64 = 1_000_000;
const DEFAULT_FLOAT_BOUND: f64 = 1_000_000.0;
const DEFAULT_FLOAT32_BOUND: f32 = 1_000_000.0;
/// The rows one generator draws: what the output depends on.
pub const SEED_CHUNK: usize = 4_096;

/// The most rows one parallel task fills, and so the most rows in one string
/// array: 1.x's chunk, which large frames are tuned to.
const MAX_TASK_ROWS: usize = 65_536;

/// Tasks per thread a column is split into, when it has rows enough. One:
/// a frame's columns already run in parallel with each other, and pass 2 of
/// PLAN-2.0.0.md measured four a thread 15% slower on a million-row list
/// column and no faster at 65,536 rows.
const TASKS_PER_THREAD: usize = 1;

/// Validity bytes covering exactly the rows of one seed chunk. The assertion
/// below is what makes that division exact -- without it a chunk would own a
/// fraction of a byte and two threads could write the same one.
pub(crate) const VALIDITY_BYTES_PER_CHUNK: usize = SEED_CHUNK / 8;
const _: () = assert!(
    SEED_CHUNK.is_multiple_of(8) && MAX_TASK_ROWS.is_multiple_of(SEED_CHUNK),
    "a seed chunk covers whole validity bytes, and a task whole seed chunks"
);

/// Mixed into a column seed to give its null stream a seed of its own.
const NULL_STREAM: u64 = 0x6E75_6C6C_5EED_0001;

/// Mixed into a column seed to give a plain string's characters a stream of
/// their own, apart from its lengths.
const CHAR_STREAM: u64 = 0x6368_6172_5EED_0002;

/// The rows one parallel task fills for a column of `n` rows: whole seed
/// chunks, `TASKS_PER_THREAD` tasks a thread, at most `MAX_TASK_ROWS`.
/// Scheduling only -- the output never depends on it.
pub(crate) fn task_rows(n: usize) -> usize {
    let threads = rayon::current_num_threads().max(1);
    let wanted = n.div_ceil(threads * TASKS_PER_THREAD);
    wanted
        .div_ceil(SEED_CHUNK)
        .clamp(1, MAX_TASK_ROWS / SEED_CHUNK)
        * SEED_CHUNK
}

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

/// What a column's chunks are seeded from: the column seed, and the index
/// of the first chunk this call fills. A call that generates rows
/// `[offset, offset + n)` of a frame numbers its chunks from
/// `offset / SEED_CHUNK`, so it draws exactly the values the whole frame
/// holds there -- a batch is a window onto one frame, whatever its size.
#[derive(Clone, Copy, Debug)]
pub struct ColumnSeed {
    pub base: u64,
    pub first_chunk: usize,
}

/// The generator one seed chunk draws its values from. `chunk_index` counts
/// from the call's first chunk.
#[inline]
pub(crate) fn chunk_rng(seed: ColumnSeed, chunk_index: usize) -> Xoshiro256PlusPlus {
    Xoshiro256PlusPlus::seed_from_u64(seed_for_chunk(seed.base, seed.first_chunk + chunk_index))
}

/// The generator one seed chunk draws its nulls from: the value stream's
/// chunk, under a column seed of its own.
#[inline]
pub(crate) fn null_rng(seed: ColumnSeed, chunk_index: usize) -> Xoshiro256PlusPlus {
    Xoshiro256PlusPlus::seed_from_u64(seed_for_chunk(
        seed.base ^ NULL_STREAM,
        seed.first_chunk + chunk_index,
    ))
}

/// Task `t`'s validity bytes, from its seed chunks' null streams. The values'
/// own loops call this inside the same task, so a nullable column is still
/// one pass over its rows.
pub(crate) fn fill_task_validity(
    bytes: &mut [u8],
    t: usize,
    task: usize,
    n: usize,
    seed: ColumnSeed,
    bernoulli: &Bernoulli,
) {
    let first = t * (task / SEED_CHUNK);
    for (j, bits) in bytes.chunks_mut(VALIDITY_BYTES_PER_CHUNK).enumerate() {
        let rows = (bits.len() * 8).min(n - (first + j) * SEED_CHUNK);
        let mut rng = null_rng(seed, first + j);
        for row in 0..rows {
            if !bernoulli.sample(&mut rng) {
                bits[row / 8] |= 1 << (row % 8);
            }
        }
    }
}

/// The generator one seed chunk draws a plain string's characters from.
#[inline]
fn char_rng(seed: ColumnSeed, chunk_index: usize) -> Xoshiro256PlusPlus {
    Xoshiro256PlusPlus::seed_from_u64(seed_for_chunk(
        seed.base ^ CHAR_STREAM,
        seed.first_chunk + chunk_index,
    ))
}

/// Whether row `row` of a seed chunk is null, for the builders that append a
/// row at a time: `nulls` is that chunk's null generator, drawn once a row.
#[inline(always)]
fn is_null(nulls: &mut Option<(Bernoulli, Xoshiro256PlusPlus)>) -> bool {
    nulls.as_mut().is_some_and(|(b, rng)| b.sample(rng))
}

/// One task's run of seed chunks, as `(chunk index, rows)` from the call's
/// first chunk: the rows `[task * task_rows, ...)` of a column of `n`.
fn task_chunks(task: usize, task_rows: usize, n: usize) -> impl Iterator<Item = (usize, usize)> {
    let first = task * (task_rows / SEED_CHUNK);
    let start = task * task_rows;
    let end = (start + task_rows).min(n);
    (0..(end - start).div_ceil(SEED_CHUNK)).map(move |j| {
        let chunk_start = start + j * SEED_CHUNK;
        (first + j, (chunk_start + SEED_CHUNK).min(end) - chunk_start)
    })
}

/// Whether this column actually produces nulls.
///
/// `nullable` with a probability of zero never does, so it takes the cheaper
/// path: no bitmap allocated, and no Bernoulli draw per row.
pub(crate) fn draws_nulls(plan: &ColumnPlan) -> bool {
    plan.nullable && plan.null_probability > 0.0
}

/// The Bernoulli a nullable column decides each row with.
///
/// Built once per column rather than once per row: `Rng::random_bool`
/// constructs one of these on every call, which for a nullable column is a
/// float conversion and a `Result` per row of the frame.
pub(crate) fn null_bernoulli(plan: &ColumnPlan) -> Result<Bernoulli, String> {
    Bernoulli::new(plan.null_probability).map_err(|e| {
        format!(
            "Invalid null_probability {} for column '{}': {e}",
            plan.null_probability, plan.name
        )
    })
}

/// Fills `n` values and their validity, task by task in parallel.
///
/// One allocation for the values and one for the validity bytes; a task owns
/// its run of `values` and the validity bytes covering the same rows, so both
/// are written without synchronisation and the result needs no
/// concatenation.
///
/// `sample` is called exactly once per row, null or not, in row order within
/// a seed chunk from that chunk's generator -- which keeps a shorter frame a
/// prefix of a longer one, and a value where it is when the null rate moves.
/// A null row's value is drawn and left in its slot, which the validity
/// bitmap marks unset; Polars never reads it.
fn fill_buffer<N, F>(
    plan: &ColumnPlan,
    n: usize,
    seed: ColumnSeed,
    sample: F,
) -> Result<(Vec<N>, Option<Bitmap>), String>
where
    N: Default + Copy + Send + Sync,
    F: Fn(&mut Xoshiro256PlusPlus) -> N + Sync,
{
    let mut values: Vec<N> = vec![N::default(); n];
    let task = task_rows(n);
    let fill = |t: usize, slots: &mut [N]| {
        let first = t * (task / SEED_CHUNK);
        for (j, chunk) in slots.chunks_mut(SEED_CHUNK).enumerate() {
            let mut rng = chunk_rng(seed, first + j);
            for slot in chunk.iter_mut() {
                *slot = sample(&mut rng);
            }
        }
    };
    if !draws_nulls(plan) {
        values
            .par_chunks_mut(task)
            .enumerate()
            .for_each(|(t, slots)| fill(t, slots));
        return Ok((values, None));
    }
    let bernoulli = null_bernoulli(plan)?;
    let mut validity: Vec<u8> = vec![0u8; n.div_ceil(8)];
    values
        .par_chunks_mut(task)
        .zip(validity.par_chunks_mut(task / 8))
        .enumerate()
        .for_each(|(t, (slots, bits))| {
            fill(t, slots);
            fill_task_validity(bits, t, task, n, seed, &bernoulli);
        });
    Ok((values, Some(Bitmap::from_u8_vec(validity, n))))
}

/// A numeric column from one buffer, as the Polars type that owns it.
fn numeric_column<T, F>(
    plan: &ColumnPlan,
    n: usize,
    seed: ColumnSeed,
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
            seed: ColumnSeed,
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

            // One loop per distribution (see `with_sampler!`).
            with_sampler!(
                dist,
                numeric_column::<$polars_type, _>(plan, n, seed, move |rng| range.sample(rng)),
                |d| numeric_column::<$polars_type, _>(plan, n, seed, move |rng| {
                    let val = d.sample::<Portable, _>(rng);
                    let rounded = if val.is_nan() { lo as f64 } else { val.round() };
                    // The f64 clamp tames a sample of any magnitude; the
                    // integer clamp undoes the rounding `lo as f64` and
                    // `hi as f64` suffer near the ends of i64/u64.
                    (rounded.clamp(lo as f64, hi as f64) as $native_type).clamp(lo, hi)
                })
            )
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

/// A uniform offset in `[0, span]`, for a 128-bit column: one 64-bit draw
/// when the span fits one, as almost every declared range does, and a full
/// 128-bit draw only when it does not.
#[derive(Clone, Copy)]
enum WideOffset {
    Narrow(Uniform<u64>),
    Full(Uniform<u128>),
}

impl WideOffset {
    fn new(span: u128, plan: &ColumnPlan) -> Result<Self, String> {
        Ok(match u64::try_from(span) {
            Ok(narrow) => WideOffset::Narrow(uniform(0, narrow, plan)?),
            Err(_) => WideOffset::Full(uniform(0, span, plan)?),
        })
    }

    #[inline(always)]
    fn sample(&self, rng: &mut Xoshiro256PlusPlus) -> u128 {
        match self {
            WideOffset::Narrow(u) => u.sample(rng) as u128,
            WideOffset::Full(u) => u.sample(rng),
        }
    }
}

/// An `Int128` column -- and a `Decimal`'s physical integers -- drawn at
/// full width: any bounds an `i128` holds.
fn gen_int128_column(
    plan: &ColumnPlan,
    n: usize,
    seed: ColumnSeed,
) -> Result<ChunkedArray<Int128Type>, String> {
    let lo = plan
        .min
        .map_or(-(DEFAULT_WIDE_INT_BOUND as i128), Limit::as_i128);
    let hi = plan
        .max
        .map_or(DEFAULT_WIDE_INT_BOUND as i128, Limit::as_i128);
    let (lo, hi) = if lo <= hi { (lo, hi) } else { (hi, lo) };
    let offset = WideOffset::new(hi.wrapping_sub(lo) as u128, plan)?;
    with_sampler!(
        plan.distribution,
        numeric_column::<Int128Type, _>(plan, n, seed, move |rng| {
            lo.wrapping_add(offset.sample(rng) as i128)
        }),
        |d| numeric_column::<Int128Type, _>(plan, n, seed, move |rng| {
            let val = d.sample::<Portable, _>(rng);
            let rounded = if val.is_nan() { lo as f64 } else { val.round() };
            (rounded.clamp(lo as f64, hi as f64) as i128).clamp(lo, hi)
        })
    )
}

/// A `UInt128` column, drawn at full width.
fn gen_uint128_column(
    plan: &ColumnPlan,
    n: usize,
    seed: ColumnSeed,
) -> Result<ChunkedArray<UInt128Type>, String> {
    let lo = plan.min.map_or(0, Limit::as_u128);
    let hi = plan
        .max
        .map_or(DEFAULT_WIDE_UINT_BOUND as u128, Limit::as_u128);
    let (lo, hi) = if lo <= hi { (lo, hi) } else { (hi, lo) };
    let offset = WideOffset::new(hi - lo, plan)?;
    with_sampler!(
        plan.distribution,
        numeric_column::<UInt128Type, _>(plan, n, seed, move |rng| lo + offset.sample(rng)),
        |d| numeric_column::<UInt128Type, _>(plan, n, seed, move |rng| {
            let val = d.sample::<Portable, _>(rng);
            let rounded = if val.is_nan() { lo as f64 } else { val.round() };
            (rounded.clamp(lo as f64, hi as f64) as u128).clamp(lo, hi)
        })
    )
}

macro_rules! impl_gen_float_column {
    ($fn_name:ident, $polars_type:ident, $native_type:ty, $default_bound:expr) => {
        fn $fn_name(
            plan: &ColumnPlan,
            n: usize,
            seed: ColumnSeed,
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

            with_sampler!(
                dist,
                numeric_column::<$polars_type, _>(plan, n, seed, move |rng| range.sample(rng)),
                |d| numeric_column::<$polars_type, _>(plan, n, seed, move |rng| {
                    let val = d.sample::<Portable, _>(rng) as $native_type;
                    if val.is_nan() { lo } else { val.clamp(lo, hi) }
                })
            )
        }
    };
}

impl_gen_float_column!(gen_float32_column, Float32Type, f32, DEFAULT_FLOAT32_BOUND);
impl_gen_float_column!(gen_float64_column, Float64Type, f64, DEFAULT_FLOAT_BOUND);

/// A boolean column, its values bit-packed.
///
/// The same division as `fill_buffer`, except that the values are a bitmap
/// too, so a task owns a byte range of it; nulls come from `null_mask`.
fn gen_bool_column(
    plan: &ColumnPlan,
    n: usize,
    seed: ColumnSeed,
) -> Result<BooleanChunked, String> {
    let name = PlSmallStr::from(plan.name.as_str());
    let mut values: Vec<u8> = vec![0u8; n.div_ceil(8)];
    let truth = Bernoulli::new(plan.p_true).map_err(|e| {
        format!(
            "Invalid boolean probability {} for column '{}': {e}",
            plan.p_true, plan.name
        )
    })?;
    let task = task_rows(n);
    let fill = |t: usize, bytes: &mut [u8]| {
        let first = t * (task / SEED_CHUNK);
        for (j, vals) in bytes.chunks_mut(VALIDITY_BYTES_PER_CHUNK).enumerate() {
            let rows = (vals.len() * 8).min(n - (first + j) * SEED_CHUNK);
            let mut rng = chunk_rng(seed, first + j);
            for row in 0..rows {
                if truth.sample(&mut rng) {
                    vals[row / 8] |= 1 << (row % 8);
                }
            }
        }
    };
    let validity = if draws_nulls(plan) {
        let bernoulli = null_bernoulli(plan)?;
        let mut validity: Vec<u8> = vec![0u8; n.div_ceil(8)];
        values
            .par_chunks_mut(task / 8)
            .zip(validity.par_chunks_mut(task / 8))
            .enumerate()
            .for_each(|(t, (bytes, bits))| {
                fill(t, bytes);
                fill_task_validity(bits, t, task, n, seed, &bernoulli);
            });
        Some(Bitmap::from_u8_vec(validity, n))
    } else {
        values
            .par_chunks_mut(task / 8)
            .enumerate()
            .for_each(|(t, bytes)| fill(t, bytes));
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
    Weighted(Alias),
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
            Some(w) => Alias::new(w)
                .map(IndexSampler::Weighted)
                .map_err(|e| format!("Invalid weights for column '{}': {e}", plan.name)),
            None => Ok(IndexSampler::Uniform(uniform(0, n as u32 - 1, plan)?)),
        }
    }

    #[inline(always)]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> u32 {
        match self {
            IndexSampler::Uniform(u) => u.sample(rng),
            IndexSampler::Weighted(w) => w.sample(rng),
        }
    }
}

/// Indices into a finite domain; Python gathers the typed values.
fn gen_index_column(
    plan: &ColumnPlan,
    n: usize,
    seed: ColumnSeed,
) -> Result<UInt32Chunked, String> {
    let sampler = IndexSampler::new(plan)?;
    numeric_column::<UInt32Type, _>(plan, n, seed, move |rng| sampler.sample(rng))
}

/// A string column, built task by task.
///
/// Strings are the one kind that cannot be written into a preallocated buffer:
/// a row's width is not known until it is drawn, and Polars backs a string
/// column with view arrays rather than one contiguous run of bytes. So each
/// task builds its own array in parallel and the column keeps them: a column
/// of up to `MAX_TASK_ROWS` rows a piece, as 1.x's chunks were.
///
/// `from_chunk_iter` rather than appending the arrays one by one: an append
/// rescans both sides for their first and last non-null value to maintain the
/// sorted flag, so hundreds of arrays mean hundreds of passes over the data to
/// maintain a flag a column of random strings will not have set anyway.
///
/// Merging the arrays into one was tried and reverted. It copies no string
/// bytes -- view arrays concatenate by merging their buffer lists -- but it
/// does copy the views, sixteen bytes a row, and at twenty million rows that
/// costs appreciably more than leaving the column split ever did.
///
/// `draw` fills `scratch` with one row's string from the chunk's value
/// stream and its character stream; it runs for every row, null or not, so
/// the values stay where they are when the null rate moves. `text` reads the
/// string back out of `scratch`.
fn build_strings<S>(
    plan: &ColumnPlan,
    n: usize,
    seed: ColumnSeed,
    scratch: impl Fn() -> S + Sync,
    draw: impl Fn(&mut Xoshiro256PlusPlus, &mut Xoshiro256PlusPlus, &mut S) + Sync,
    text: impl Fn(&S) -> &str + Sync,
) -> Result<StringChunked, String> {
    let name = PlSmallStr::from(plan.name.as_str());
    if n == 0 {
        return Ok(StringChunkedBuilder::new(name, 0).finish());
    }
    let bernoulli = draws_nulls(plan)
        .then(|| null_bernoulli(plan))
        .transpose()?;
    let task = task_rows(n);
    let arrays: Vec<Utf8ViewArray> = (0..n.div_ceil(task))
        .into_par_iter()
        .map(|t| {
            let rows = (t * task + task).min(n) - t * task;
            let mut views = Views::with_capacity(rows, bernoulli.is_some(), 0);
            let mut buffer = scratch();
            for (chunk, rows) in task_chunks(t, task, n) {
                let mut rng = chunk_rng(seed, chunk);
                let mut chars = char_rng(seed, chunk);
                let mut nulls = bernoulli.map(|b| (b, null_rng(seed, chunk)));
                for _ in 0..rows {
                    draw(&mut rng, &mut chars, &mut buffer);
                    if is_null(&mut nulls) {
                        views.push_null();
                    } else {
                        views.push(text(&buffer));
                    }
                }
            }
            views.finish()
        })
        .collect();
    Ok(StringChunked::from_chunk_iter(name, arrays))
}

/// A plain string column, drawn in bulk a seed chunk at a time: every row's
/// length from the chunk's value stream, then every character of those rows
/// as one run from its character stream (`fill_ascii`). Two streams, so the
/// first rows of a chunk hold the same strings however many of its rows are
/// asked for. A null row's length and characters are drawn too, so the
/// values stay where they are when the null rate moves.
fn gen_string_column(
    plan: &ColumnPlan,
    n: usize,
    seed: ColumnSeed,
) -> Result<StringChunked, String> {
    let name = PlSmallStr::from(plan.name.as_str());
    if n == 0 {
        return Ok(StringChunked::with_chunk(
            name,
            Views::with_capacity(0, false, 0).finish(),
        ));
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
    // Bytes of a task's strings that will not fit inline in their views; and
    // whether none will, so the characters can be drawn where they are kept.
    let long = if max_len > View::MAX_INLINE_SIZE as usize {
        (min_len + max_len) / 2
    } else {
        0
    };
    let all_stored = min_len > View::MAX_INLINE_SIZE as usize;
    let alphabet = Alphabet::new(CHARSET);
    let task = task_rows(n);
    let arrays: Vec<Utf8ViewArray> = (0..n.div_ceil(task))
        .into_par_iter()
        .map(|t| {
            let rows = (t * task + task).min(n) - t * task;
            let mut views = Views::with_capacity(rows, bernoulli.is_some(), rows * long);
            let mut lengths = vec![0usize; SEED_CHUNK];
            let mut chars: Vec<u8> = Vec::with_capacity(SEED_CHUNK * max_len + SLACK);
            for (chunk, rows) in task_chunks(t, task, n) {
                let mut rng = chunk_rng(seed, chunk);
                let lengths = &mut lengths[..rows];
                match &len_dist {
                    Some(d) => lengths.iter_mut().for_each(|l| *l = d.sample(&mut rng)),
                    None => lengths.fill(min_len),
                }
                let total: usize = lengths.iter().sum();
                let mut chars_rng = char_rng(seed, chunk);
                let mut nulls = bernoulli.map(|b| (b, null_rng(seed, chunk)));
                // CHARSET is ASCII, so every run of it is UTF-8.
                if all_stored {
                    // Every string is too long to inline: draw the chunk's
                    // characters into the data buffer itself, and point at
                    // them. A null row's characters sit there unread.
                    let (mut at, room) = views.reserve_stored(total + SLACK);
                    alphabet.fill(&mut chars_rng, room, total);
                    views.release_stored(SLACK);
                    for &len in lengths.iter() {
                        if is_null(&mut nulls) {
                            views.push_null();
                        } else {
                            views.push_stored(at, len);
                        }
                        at += len;
                    }
                    continue;
                }
                // The fill's slack doubles as the sixteen bytes past the last
                // string that `push_padded` reads.
                chars.resize(total + SLACK, 0);
                alphabet.fill(&mut chars_rng, &mut chars, total);
                let mut at = 0;
                for &len in lengths.iter() {
                    if is_null(&mut nulls) {
                        views.push_null();
                    } else {
                        views.push_padded(&chars, at, len);
                    }
                    at += len;
                }
            }
            views.finish()
        })
        .collect();
    Ok(StringChunked::from_chunk_iter(name, arrays))
}

/// A templated string column: each row filled from the plan's `Template`.
fn gen_template_column(
    plan: &ColumnPlan,
    n: usize,
    seed: ColumnSeed,
) -> Result<StringChunked, String> {
    let template = plan
        .template
        .as_ref()
        .ok_or_else(|| format!("Column '{}' has kind 'template' but no template", plan.name))?;
    let sampler = template.sampler();
    build_strings(
        plan,
        n,
        seed,
        || Vec::<u8>::with_capacity(template.max_bytes()),
        |rng, chars, bytes| sampler.fill(rng, chars, bytes),
        // SAFETY: every template part is ASCII or a `String`, so the buffer
        // is valid UTF-8 -- see `format::Template::compile`.
        |bytes| unsafe { std::str::from_utf8_unchecked(bytes) },
    )
}

/// A pattern column: each row a full match of the plan's pattern, from a
/// walk through its automaton -- see `pattern.rs`.
fn gen_pattern_column(
    plan: &ColumnPlan,
    n: usize,
    seed: ColumnSeed,
) -> Result<StringChunked, String> {
    let sampler = plan
        .pattern
        .as_ref()
        .ok_or_else(|| format!("Column '{}' has kind 'pattern' but no pattern", plan.name))?;
    build_strings(
        plan,
        n,
        seed,
        || Vec::<u8>::with_capacity(sampler.max_bytes()),
        |rng, _, bytes| sampler.fill(rng, bytes),
        // SAFETY: every unit of a pattern is a `str`, so a value made of
        // them is valid UTF-8 -- see `PatternSampler::fill`.
        |bytes| unsafe { std::str::from_utf8_unchecked(bytes) },
    )
}

/// Fills one column according to its plan.
/// Rows `[row_offset, row_offset + n)` of the column `plan` describes.
///
/// Seed chunks are numbered from `row_offset / SEED_CHUNK`, so the values are
/// the ones the whole column holds at those rows whatever `n` and `row_offset`
/// are; a call starting mid-chunk fills that chunk from its start and slices
/// the head off, at most one partial seed chunk of extra work. A unique column is
/// a window too, through a permutation of its value space.
pub fn generate_series(
    plan: &ColumnPlan,
    n: usize,
    seed: u64,
    row_offset: usize,
) -> Result<Series, String> {
    let head = row_offset % SEED_CHUNK;
    let seed = ColumnSeed {
        base: seed,
        first_chunk: row_offset / SEED_CHUNK,
    };
    let series = if plan.unique {
        crate::unique::generate_unique_series(plan, n + head, seed)?
    } else {
        generate_chunked(plan, n + head, seed)?
    };
    Ok(if head == 0 {
        series
    } else {
        series.slice(head as i64, n)
    })
}

fn generate_chunked(plan: &ColumnPlan, n: usize, seed: ColumnSeed) -> Result<Series, String> {
    Ok(match plan.kind {
        Kind::Int64 => gen_int64_column(plan, n, seed)?.into_series(),
        Kind::Int32 => gen_int32_column(plan, n, seed)?.into_series(),
        Kind::Int16 => gen_int16_column(plan, n, seed)?.into_series(),
        Kind::Int8 => gen_int8_column(plan, n, seed)?.into_series(),
        Kind::UInt64 => gen_uint64_column(plan, n, seed)?.into_series(),
        Kind::Int128 => gen_int128_column(plan, n, seed)?.into_series(),
        Kind::UInt128 => gen_uint128_column(plan, n, seed)?.into_series(),
        Kind::UInt32 => gen_uint32_column(plan, n, seed)?.into_series(),
        Kind::UInt16 => gen_uint16_column(plan, n, seed)?.into_series(),
        Kind::UInt8 => gen_uint8_column(plan, n, seed)?.into_series(),
        Kind::Float64 => gen_float64_column(plan, n, seed)?.into_series(),
        Kind::Float32 => gen_float32_column(plan, n, seed)?.into_series(),
        Kind::Bool => gen_bool_column(plan, n, seed)?.into_series(),
        Kind::String => gen_string_column(plan, n, seed)?.into_series(),
        Kind::Template => gen_template_column(plan, n, seed)?.into_series(),
        Kind::Pattern => gen_pattern_column(plan, n, seed)?.into_series(),
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
        let n = SEED_CHUNK * 2 + 17;
        let a = generate_series(&p, n, 42, 0).unwrap();
        let b = generate_series(&p, n, 42, 0).unwrap();
        assert!(a.equals(&b));
        assert_eq!(a.len(), n);
        let c = generate_series(&p, n, 43, 0).unwrap();
        assert!(!a.equals(&c));
    }

    #[test]
    fn a_shorter_frame_is_a_prefix_of_a_longer_one() {
        // Chunk seeds depend on the chunk index alone, so the first rows of a
        // column never change when more rows are asked for.
        let p = simple("float64");
        let short = generate_series(&p, 1000, 7, 0).unwrap();
        let long = generate_series(&p, SEED_CHUNK + 1000, 7, 0).unwrap();
        assert!(short.equals(&long.slice(0, 1000)));
    }

    #[test]
    fn an_offset_call_is_a_window_onto_the_whole_column() {
        // Every kind, every offset: rows [offset, offset + n) of a call with
        // that offset are the rows the whole column holds there -- on a chunk
        // boundary, mid-chunk, spanning several chunks, and nullable.
        let whole_n = SEED_CHUNK * 3 + 500;
        let template = [("chars".to_string(), vec!["ab".to_string()], 2, 4)];
        for kind in ["int64", "float64", "bool", "string", "template", "index"] {
            let plan = ColumnPlan::build(PlanArgs {
                name: "c".into(),
                kind,
                template: (kind == "template").then_some(&template[..]),
                n_categories: (kind == "index").then_some(7),
                ..Default::default()
            })
            .unwrap();
            let whole = generate_series(&plan, whole_n, 9, 0).unwrap();
            for (offset, n) in [
                (0, 10),
                (SEED_CHUNK, 10),
                (17, 100),
                (SEED_CHUNK - 3, 10),
                (SEED_CHUNK + 1234, SEED_CHUNK * 2 - 2000),
                (whole_n - 5, 5),
            ] {
                let window = generate_series(&plan, n, 9, offset).unwrap();
                assert!(
                    window.equals_missing(&whole.slice(offset as i64, n)),
                    "{kind} at offset {offset} for {n} rows"
                );
            }
        }
        let nullable = ColumnPlan::build(PlanArgs {
            name: "c".into(),
            kind: "int64",
            nullable: true,
            null_probability: 0.3,
            ..Default::default()
        })
        .unwrap();
        let whole = generate_series(&nullable, whole_n, 9, 0).unwrap();
        let window = generate_series(&nullable, 1000, 9, SEED_CHUNK - 400).unwrap();
        assert!(window.equals_missing(&whole.slice((SEED_CHUNK - 400) as i64, 1000)));
        assert_eq!(window.len(), 1000);
    }

    fn in_pool<T: Send>(threads: usize, f: impl FnOnce() -> T + Send) -> T {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(f)
    }

    #[test]
    fn a_seed_gives_one_column_on_any_number_of_threads() {
        // Tasks are sized by the thread count; seed chunks are not. Every
        // kind, nullable and not, across several tasks of every size.
        let n = MAX_TASK_ROWS * 3 + 777;
        let template = [("chars".to_string(), vec!["ab".to_string()], 2, 4)];
        for kind in ["int64", "float64", "bool", "string", "template", "index"] {
            for nullable in [false, true] {
                let plan = ColumnPlan::build(PlanArgs {
                    name: "c".into(),
                    kind,
                    nullable,
                    null_probability: 0.3,
                    template: (kind == "template").then_some(&template[..]),
                    n_categories: (kind == "index").then_some(7),
                    ..Default::default()
                })
                .unwrap();
                let one = in_pool(1, || generate_series(&plan, n, 5, 0).unwrap());
                for threads in [2, 3, 8, 32] {
                    let many = in_pool(threads, || generate_series(&plan, n, 5, 0).unwrap());
                    assert!(
                        one.equals_missing(&many),
                        "{kind} (nullable {nullable}) differs on {threads} threads"
                    );
                }
            }
        }
    }

    #[test]
    fn a_task_is_whole_seed_chunks_and_at_most_the_largest() {
        for n in [0, 1, SEED_CHUNK, 65_536, 1_000_000, 50_000_000] {
            for threads in [1, 4, 32] {
                let rows = in_pool(threads, || task_rows(n));
                assert!(rows.is_multiple_of(SEED_CHUNK), "{n} on {threads}: {rows}");
                assert!((SEED_CHUNK..=MAX_TASK_ROWS).contains(&rows));
            }
        }
        // A fixture-sized frame splits on many threads, so it fills on them.
        assert!(in_pool(32, || task_rows(65_536)) < 65_536);
    }

    #[test]
    fn a_columns_values_stay_where_they_are_when_its_null_rate_moves() {
        // Nulls draw from a stream of their own, so a row present at two null
        // rates holds the same value at both, and the value a column with no
        // nulls holds there.
        let n = SEED_CHUNK * 5 + 31;
        let template = [("chars".to_string(), vec!["abc".to_string()], 1, 6)];
        for kind in ["int64", "float64", "bool", "string", "template", "index"] {
            let with = |nullable: bool, p: f64| {
                let plan = ColumnPlan::build(PlanArgs {
                    name: "c".into(),
                    kind,
                    nullable,
                    null_probability: p,
                    template: (kind == "template").then_some(&template[..]),
                    n_categories: (kind == "index").then_some(9),
                    ..Default::default()
                })
                .unwrap();
                generate_series(&plan, n, 13, 0).unwrap()
            };
            let none = with(false, 0.0);
            for p in [0.1, 0.5] {
                let some = with(true, p);
                assert!(some.null_count() > 0, "{kind} at {p}");
                let present = some.is_not_null();
                assert!(
                    some.filter(&present)
                        .unwrap()
                        .equals(&none.filter(&present).unwrap()),
                    "{kind}: a present value moved at null rate {p}"
                );
            }
        }
    }

    #[test]
    fn a_fixed_width_column_arrives_as_one_chunk() {
        // The whole point of filling into one buffer: nothing downstream --
        // a gather, a cast, a write -- pays for a column split 40 ways.
        let n = SEED_CHUNK * 40 + 3;
        for kind in ["int64", "float32", "bool"] {
            let s = generate_series(&simple(kind), n, 3, 0).unwrap();
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
        assert_eq!(generate_series(&nullable, n, 3, 0).unwrap().n_chunks(), 1);
    }

    #[test]
    fn a_string_column_stays_chunked() {
        // Deliberate, and the opposite of every other kind: merging a view
        // array copies sixteen bytes of view per row, which costs more than
        // the split it removes. Pinned so the merge is not quietly re-added.
        let n = SEED_CHUNK * 40 + 3;
        let s = generate_series(&simple("string"), n, 3, 0).unwrap();
        assert_eq!(s.len(), n);
        assert!(s.n_chunks() > 1, "a long string column is left split");
    }

    #[test]
    fn integer_bounds_hold_at_the_extremes_of_i64_and_u64() {
        let lo = 9_007_199_254_740_990i64; // 2**53 - 2
        let hi = 9_007_199_254_740_999i64;
        let p = bounded("int64", Limit::Int(lo), Limit::Int(hi));
        let s = generate_series(&p, 5000, 1, 0).unwrap();
        let ca = s.i64().unwrap();
        assert_eq!(ca.min().unwrap(), lo);
        assert_eq!(ca.max().unwrap(), hi);

        let lo = u64::MAX - 15;
        let p = bounded("uint64", Limit::UInt(lo), Limit::UInt(u64::MAX));
        let s = generate_series(&p, 5000, 1, 0).unwrap();
        let ca = s.u64().unwrap();
        assert!(ca.min().unwrap() >= lo);
        assert!(
            ca.n_unique().unwrap() > 1,
            "a 16-value range must not collapse"
        );
    }

    #[test]
    fn a_128_bit_column_is_drawn_at_full_width() {
        // Past what 64 bits hold, up to the type's own limits, a window as
        // every column is.
        for (kind, lo, hi) in [
            (
                "int128",
                Limit::Wide(-(1i128 << 100)),
                Limit::Wide(1i128 << 100),
            ),
            ("int128", Limit::Wide(i128::MIN), Limit::Wide(i128::MAX)),
            (
                "uint128",
                Limit::WideU(1u128 << 127),
                Limit::WideU(u128::MAX),
            ),
            ("uint128", Limit::UInt(0), Limit::WideU(u128::MAX)),
        ] {
            let p = bounded(kind, lo, hi);
            let n = SEED_CHUNK * 3 + 11;
            let whole = generate_series(&p, n, 5, 0).unwrap();
            let window = generate_series(&p, 500, 5, SEED_CHUNK + 7).unwrap();
            assert!(
                window.equals(&whole.slice((SEED_CHUNK + 7) as i64, 500)),
                "{kind}"
            );
            if kind == "int128" {
                let ca = whole.i128().unwrap();
                let (min, max) = (ca.min().unwrap(), ca.max().unwrap());
                assert!(lo.as_i128() <= min && max <= hi.as_i128(), "{kind}");
                assert!(max > u64::MAX as i128 && min < i64::MIN as i128, "{kind}");
            } else {
                let ca = whole.u128().unwrap();
                let (min, max) = (ca.min().unwrap(), ca.max().unwrap());
                assert!(lo.as_u128() <= min && max <= hi.as_u128(), "{kind}");
                assert!(max > (1u128 << 126) * 3, "{kind}: {max}");
            }
        }
    }

    #[test]
    fn a_128_bit_range_that_fits_64_bits_draws_what_a_64_bit_one_does() {
        // One 64-bit offset from the low end, as `Uniform<i64>` draws: every
        // range a 64-bit draw could take is unchanged by the wider kind.
        let narrow = generate_series(
            &bounded("int64", Limit::Int(-50), Limit::Int(9_000)),
            5_000,
            3,
            0,
        )
        .unwrap();
        let wide = generate_series(
            &bounded("int128", Limit::Int(-50), Limit::Int(9_000)),
            5_000,
            3,
            0,
        )
        .unwrap();
        let wide = wide.cast(&DataType::Int64).unwrap();
        assert!(narrow.equals(&wide));
    }

    #[test]
    fn a_float_limit_on_an_integer_column_truncates() {
        let p = bounded("int32", Limit::Float(2.9), Limit::Float(2.9));
        let s = generate_series(&p, 10, 1, 0).unwrap();
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
        let s = generate_series(&p, 10_000, 3, 0).unwrap();
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
        let s = generate_series(&p, 1000, 3, 0).unwrap();
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
        let s = generate_series(&p, 20_000, 9, 0).unwrap();
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
        let s = generate_series(&p, 5_000, 4, 0).unwrap();
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
        let s = generate_series(&p, 2000, 5, 0).unwrap();
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
        let s = generate_series(&p, 5_000, 11, 0).unwrap();
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
        let s = generate_series(&p, 100, 1, 0).unwrap();
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
            let s = generate_series(&simple(kind), 0, 1, 0).unwrap();
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
            generate_series(&p, 0, 1, 0).unwrap().dtype(),
            &DataType::UInt32
        );
    }
}

#[cfg(test)]
mod timing {
    //! `cargo test --release -- --ignored --nocapture timing`: one thread,
    //! ns per row, for the string kinds pass 3 of PLAN-2.0.0.md speeds up.
    use super::*;
    use crate::plan::PlanArgs;

    fn ns_per_row(plan: &ColumnPlan, n: usize) -> f64 {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        pool.install(|| {
            let _ = std::hint::black_box(generate_series(plan, n, 1, 0).unwrap());
            let mut best = f64::MAX;
            for _ in 0..7 {
                let t = std::time::Instant::now();
                let _ = std::hint::black_box(generate_series(plan, n, 1, 0).unwrap());
                best = best.min(t.elapsed().as_secs_f64());
            }
            best * 1e9 / n as f64
        })
    }

    #[test]
    #[ignore]
    fn timing_strings() {
        let n = 1_000_000;
        let template = [
            (
                "chars".to_string(),
                vec!["abcdefghijklmnopqrstuvwxyz".to_string()],
                3,
                10,
            ),
            ("lit".to_string(), vec!["@example.".to_string()], 0, 0),
            (
                "one_of".to_string(),
                vec!["com".to_string(), "org".to_string()],
                0,
                0,
            ),
        ];
        for (label, args) in [
            (
                "string 5-15",
                PlanArgs {
                    name: "c".into(),
                    kind: "string",
                    str_min_len: Some(5),
                    str_max_len: Some(15),
                    ..Default::default()
                },
            ),
            (
                "string 5-15, 10% null",
                PlanArgs {
                    name: "c".into(),
                    kind: "string",
                    str_min_len: Some(5),
                    str_max_len: Some(15),
                    nullable: true,
                    null_probability: 0.1,
                    ..Default::default()
                },
            ),
            (
                "string 20-40",
                PlanArgs {
                    name: "c".into(),
                    kind: "string",
                    str_min_len: Some(20),
                    str_max_len: Some(40),
                    ..Default::default()
                },
            ),
            (
                "string 8",
                PlanArgs {
                    name: "c".into(),
                    kind: "string",
                    str_min_len: Some(8),
                    str_max_len: Some(8),
                    ..Default::default()
                },
            ),
            (
                "template email",
                PlanArgs {
                    name: "c".into(),
                    kind: "template",
                    template: Some(&template[..]),
                    ..Default::default()
                },
            ),
            (
                "pattern ORD-[0-9]{6}",
                PlanArgs {
                    name: "c".into(),
                    kind: "pattern",
                    pattern: Some(r"^ORD-[0-9]{6}$"),
                    ..Default::default()
                },
            ),
            (
                "pattern bench",
                PlanArgs {
                    name: "c".into(),
                    kind: "pattern",
                    pattern: Some(r"^[A-Z]{2}-\d{4,6}(-(EU|US|UK))?$"),
                    ..Default::default()
                },
            ),
        ] {
            let plan = ColumnPlan::build(args).unwrap();
            println!("{label:<24} {:>6.2} ns/row", ns_per_row(&plan, n));
        }
    }
}

#[cfg(test)]
mod timing_parts {
    use super::*;
    use std::hint::black_box;

    fn time(label: &str, n: usize, mut f: impl FnMut() -> usize) {
        black_box(f());
        let mut best = f64::MAX;
        for _ in 0..7 {
            let t = std::time::Instant::now();
            black_box(f());
            best = best.min(t.elapsed().as_secs_f64());
        }
        println!("{label:<34} {:>6.2} ns/row", best * 1e9 / n as f64);
    }

    #[test]
    #[ignore]
    fn timing_string_parts() {
        let n = 1_000_000;
        let mut scratch = vec![0u8; 64];
        scratch.resize(n * 10 + crate::chars::SLACK, 0);
        let alphabet = Alphabet::new(CHARSET);
        time("chars only (Alphabet::fill), 10 a row", n, || {
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(1);
            alphabet.fill(&mut rng, &mut scratch, n * 10);
            scratch[n] as usize
        });
        time("next_u64 x2 per row (floor)", n, || {
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(1);
            (0..n)
                .map(|_| (rng.next_u64() ^ rng.next_u64()) as usize & 1)
                .sum()
        });
    }
}

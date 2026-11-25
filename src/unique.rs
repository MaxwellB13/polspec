//! Filling a column whose values must all differ.
//!
//! Every unique column has an enumerable value space -- an integer range, a
//! set of categories, the two booleans, a grid over a float range, the
//! strings a template can make -- and is filled through a keyed permutation
//! of that space (`crate::permute`): the `k`-th value the column holds is the
//! `π(k)`-th value of the space. Distinct values come from distinct `k`, and
//! `k` depends only on the row and the seed, so the column is unique over the
//! whole frame, any window of rows is the whole column's slice -- a batch of
//! `generate_batches` included -- and chunks fill in parallel with nothing
//! held but the output.
//!
//! A string's space is enumerable too: a template's values -- or a pattern's
//! -- decode from an index one mixed-radix digit per part
//! (`format::Template::decode`). A plain string is drawn evenly over its
//! lengths, as a column that is not unique is: `EvenLengths` orders its space
//! so that any prefix of it holds every length alike, and the permutation
//! runs over such a prefix.
//!
//! Nulls are exempt, as they are everywhere else in polspec: a null means "no
//! value", and repeating it is not repeating a value. A null spends no value
//! either: `k` counts the non-null rows before a row, so a column's space has
//! to hold only the values it draws. Counting them for a window that starts
//! deep in a frame cannot mean drawing every row before it, so a nullable
//! unique column draws its nulls a seed chunk at a time: how many, from the
//! binomial a chunk's row-by-row draws would give (`ChunkNulls`), then which
//! rows, evenly. The rows before a window cost one draw a seed chunk.

use polars::prelude::*;
use polars_arrow::array::BooleanArray;
use polars_arrow::array::Utf8ViewArray;
use polars_arrow::bitmap::Bitmap;
use polars_arrow::datatypes::ArrowDataType;
use polars_core::chunked_array::builder::StringChunkedBuilder;
use rand::Rng;
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;

use crate::format::Enumerable;
use crate::permute::Permutation;
use crate::plan::{ColumnPlan, Kind, Limit};
use crate::sample::{
    CHARSET, ColumnSeed, SEED_CHUNK, VALIDITY_BYTES_PER_CHUNK, draws_nulls, null_bernoulli,
    null_rng, task_rows,
};
use crate::views::Views;

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
// Which rows hold a value, and which value
// ---------------------------------------------------------------------------

/// A binomial's tail past this, relative to its mode, is left out of
/// `ChunkNulls`' table: less mass than an `f64` resolves.
const TAIL: f64 = 1e-20;

/// How many of a seed chunk's rows are null: `Binomial(SEED_CHUNK, p)`, the
/// count `SEED_CHUNK` independent draws at `p` give, by inversion of a table.
///
/// The table is built from the ratio of neighbouring probabilities alone --
/// multiplications and divisions, no `exp` or `ln` -- so it is the same on
/// every platform, and so is every count drawn from it.
struct ChunkNulls {
    /// The smallest count the table holds.
    first: usize,
    /// `P(count <= first + i)`, scaled to `2^64`; the last is `u64::MAX`.
    thresholds: Vec<u64>,
}

impl ChunkNulls {
    fn new(p: f64) -> Self {
        let n = SEED_CHUNK;
        let q = 1.0 - p;
        if q <= 0.0 {
            return ChunkNulls {
                first: n,
                thresholds: vec![u64::MAX],
            };
        }
        // Weights relative to the mode's, out to where they vanish.
        let mode = (((n + 1) as f64 * p) as usize).min(n);
        let (up, down) = (p / q, q / p);
        let mut above = Vec::new();
        let (mut w, mut k) = (1.0f64, mode);
        while k < n {
            w *= (n - k) as f64 / (k + 1) as f64 * up;
            if w < TAIL {
                break;
            }
            above.push(w);
            k += 1;
        }
        let mut below = Vec::new();
        let (mut w, mut k) = (1.0f64, mode);
        while k > 0 {
            w *= k as f64 / (n - k + 1) as f64 * down;
            if w < TAIL {
                break;
            }
            below.push(w);
            k -= 1;
        }
        let first = mode - below.len();
        let weights: Vec<f64> = below
            .iter()
            .rev()
            .copied()
            .chain(std::iter::once(1.0))
            .chain(above)
            .collect();
        let total: f64 = weights.iter().sum();
        let mut cumulative = 0.0;
        let mut thresholds: Vec<u64> = weights
            .iter()
            .map(|w| {
                cumulative += w;
                // `as` saturates, so a sum that rounds past 1 is `u64::MAX`.
                (cumulative / total * 18_446_744_073_709_551_616.0) as u64
            })
            .collect();
        *thresholds.last_mut().expect("the mode is in the table") = u64::MAX;
        ChunkNulls { first, thresholds }
    }

    /// One chunk's null count, from its null stream's first draw.
    #[inline]
    fn count(&self, rng: &mut Xoshiro256PlusPlus) -> usize {
        let u = rng.next_u64();
        let i = self.thresholds.partition_point(|t| *t <= u);
        self.first + i.min(self.thresholds.len() - 1)
    }
}

/// One seed chunk's validity bits for its first `rows` rows: `nulls` of its
/// `SEED_CHUNK` rows null, each subset of that size alike likely. Row by row,
/// a row is null with the share of the rows left that the nulls left are --
/// so a chunk cut short is the start of the whole chunk. Returns how many of
/// the `rows` hold a value.
fn place_nulls(bits: &mut [u8], rows: usize, nulls: usize, rng: &mut Xoshiro256PlusPlus) -> usize {
    let mut left = nulls;
    let mut values = 0;
    for row in 0..rows {
        let remaining = (SEED_CHUNK - row) as u128;
        let null = if left == 0 {
            false
        } else if left as u128 == remaining {
            true
        } else {
            ((rng.next_u64() as u128 * remaining) >> 64) < left as u128
        };
        if null {
            left -= 1;
        } else {
            bits[row / 8] |= 1 << (row % 8);
            values += 1;
        }
    }
    values
}

/// The rows a call fills and the value each non-null one takes.
///
/// `first_rank[j]` is the rank -- the position in the permuted space -- of
/// the first value seed chunk `j` of the call holds; each later value in the
/// chunk takes the next. `needed` is one past the last rank, so the space
/// must hold that many values.
struct Ranks {
    validity: Option<Bitmap>,
    first_rank: Vec<u128>,
    needed: u128,
}

impl Ranks {
    fn new(plan: &ColumnPlan, n: usize, seed: ColumnSeed) -> Result<Self, String> {
        let first_row = seed.first_chunk * SEED_CHUNK;
        let chunks = n.div_ceil(SEED_CHUNK);
        if !draws_nulls(plan) {
            return Ok(Ranks {
                validity: None,
                first_rank: (0..chunks)
                    .map(|j| (first_row + j * SEED_CHUNK) as u128)
                    .collect(),
                needed: (first_row + n) as u128,
            });
        }
        // The same refusal as any nullable column's, for a rate out of range.
        null_bernoulli(plan)?;
        let table = ChunkNulls::new(plan.null_probability);
        // The values every seed chunk before the call holds: one draw each.
        let from_zero = ColumnSeed {
            base: seed.base,
            first_chunk: 0,
        };
        let before: u128 = (0..seed.first_chunk)
            .into_par_iter()
            .map(|c| (SEED_CHUNK - table.count(&mut null_rng(from_zero, c))) as u128)
            .sum();
        let mut bytes = vec![0u8; n.div_ceil(8)];
        let values: Vec<usize> = bytes
            .par_chunks_mut(VALIDITY_BYTES_PER_CHUNK)
            .enumerate()
            .map(|(j, bits)| {
                let rows = SEED_CHUNK.min(n - j * SEED_CHUNK);
                let mut rng = null_rng(seed, j);
                let nulls = table.count(&mut rng);
                place_nulls(bits, rows, nulls, &mut rng)
            })
            .collect();
        let mut rank = before;
        let first_rank = values
            .iter()
            .map(|v| {
                let start = rank;
                rank += *v as u128;
                start
            })
            .collect();
        Ok(Ranks {
            validity: Some(Bitmap::from_u8_vec(bytes, n)),
            first_rank,
            needed: rank,
        })
    }

    /// Calls `f(row, rank)` for each row in `rows` that holds a value, in
    /// order; `rows` starts on a seed chunk.
    #[inline]
    fn for_each(&self, rows: std::ops::Range<usize>, mut f: impl FnMut(usize, u128)) {
        let mut rank = 0;
        for row in rows {
            if row % SEED_CHUNK == 0 {
                rank = self.first_rank[row / SEED_CHUNK];
            }
            if self.validity.as_ref().is_none_or(|v| v.get_bit(row)) {
                f(row, rank);
                rank += 1;
            }
        }
    }
    /// Calls `f(row, π(rank))` for each row in `rows` that holds a value, in
    /// order. A call's ranks run on without a gap, so they go through the
    /// permutation a block at a time (`Permutation::apply_run`).
    fn for_each_value(
        &self,
        rows: std::ops::Range<usize>,
        permutation: &Permutation,
        mut f: impl FnMut(usize, u128),
    ) {
        const BLOCK: usize = 256;
        let mut at = [0usize; BLOCK];
        let mut values = [0u128; BLOCK];
        let (mut m, mut first) = (0usize, 0u128);
        self.for_each(rows, |row, rank| {
            if m == 0 {
                first = rank;
            }
            at[m] = row;
            m += 1;
            if m == BLOCK {
                permutation.apply_run(first, &mut values);
                at.iter().zip(&values).for_each(|(r, v)| f(*r, *v));
                m = 0;
            }
        });
        if m > 0 {
            permutation.apply_run(first, &mut values[..m]);
            at[..m]
                .iter()
                .zip(&values[..m])
                .for_each(|(r, v)| f(*r, *v));
        }
    }
}

// ---------------------------------------------------------------------------
// The permuted kinds
// ---------------------------------------------------------------------------

/// Rows `[first_row, first_row + n)` of a column whose `k`-th value is
/// `decode(π(k))`, and their validity. `first_row` is where `seed`'s first
/// seed chunk starts, so the rows ranked and the nulls drawn are the whole
/// column's. A row's value depends on the row alone, so the tasks are only a
/// way to share the work. A null row's slot holds `N::default()`.
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
    let ranks = Ranks::new(plan, n, seed)?;
    if ranks.needed > domain {
        return Err(too_small(plan, ranks.needed, domain));
    }
    let mut values: Vec<N> = vec![N::default(); n];
    if ranks.needed == 0 {
        return Ok((values, ranks.validity));
    }
    let permutation = Permutation::new(domain, permutation_seed(seed));
    let task = task_rows(n);
    values
        .par_chunks_mut(task)
        .enumerate()
        .for_each(|(t, slots)| {
            let start = t * task;
            ranks.for_each_value(start..start + slots.len(), &permutation, |row, value| {
                slots[row - start] = decode(value);
            });
        });
    Ok((values, ranks.validity))
}

/// The key a column's permutation is built from: the column seed, kept apart
/// from the seeds its null stream draws from.
fn permutation_seed(seed: ColumnSeed) -> u64 {
    seed.base ^ 0x5EED_0F5A_BE11_A5E5
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

/// An integer column's value space, for a type holding `min..=max`: its
/// lowest value and how many values it holds. The bounds are clamped to the
/// type, and taken in either order.
fn int_space(plan: &ColumnPlan, min: i128, max: i128) -> (i128, u128) {
    let clamp = |v: i128| -> i128 { v.clamp(min, max) };
    let lo = plan.min.map_or(min, |l| clamp(l.as_i128()));
    let hi = plan.max.map_or(max, |l| clamp(l.as_i128()));
    let (lo, hi) = if lo <= hi { (lo, hi) } else { (hi, lo) };
    (lo, (hi - lo + 1) as u128)
}

macro_rules! impl_unique_int_column {
    ($fn_name:ident, $polars_type:ident, $native_type:ty) => {
        fn $fn_name(
            plan: &ColumnPlan,
            n: usize,
            seed: ColumnSeed,
        ) -> Result<ChunkedArray<$polars_type>, String> {
            let (lo, domain) = int_space(
                plan,
                <$native_type>::MIN as i128,
                <$native_type>::MAX as i128,
            );
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

/// A unique `Int128` column -- or a `Decimal`'s physical integers -- over any
/// bounds an `i128` holds. The value space of the full range is one larger
/// than a `u128` counts, so it stops one short of `i128::MAX`.
fn unique_int128(
    plan: &ColumnPlan,
    n: usize,
    seed: ColumnSeed,
) -> Result<ChunkedArray<Int128Type>, String> {
    let (lo, domain) = int128_space(plan);
    permuted_column(plan, n, seed, domain, |i| lo.wrapping_add(i as i128))
}

/// An `Int128` column's value space: its lowest value and how many values.
fn int128_space(plan: &ColumnPlan) -> (i128, u128) {
    let lo = plan.min.map_or(i128::MIN, Limit::as_i128);
    let hi = plan.max.map_or(i128::MAX, Limit::as_i128);
    let (lo, hi) = if lo <= hi { (lo, hi) } else { (hi, lo) };
    (lo, (hi.wrapping_sub(lo) as u128).saturating_add(1))
}

/// A unique `UInt128` column, over any bounds a `u128` holds.
fn unique_uint128(
    plan: &ColumnPlan,
    n: usize,
    seed: ColumnSeed,
) -> Result<ChunkedArray<UInt128Type>, String> {
    let (lo, domain) = uint128_space(plan);
    permuted_column(plan, n, seed, domain, |i| lo + i)
}

/// A `UInt128` column's value space: its lowest value and how many values.
fn uint128_space(plan: &ColumnPlan) -> (u128, u128) {
    let lo = plan.min.map_or(0, Limit::as_u128);
    let hi = plan.max.map_or(u128::MAX, Limit::as_u128);
    let (lo, hi) = if lo <= hi { (lo, hi) } else { (hi, lo) };
    (lo, (hi - lo).saturating_add(1))
}

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

/// A float column's range: its bounds, or `±default` where open, in order.
fn float_bounds(plan: &ColumnPlan, default: f64) -> (f64, f64) {
    let lo = plan.min.map_or(-default, |l| l.as_f64());
    let hi = plan.max.map_or(default, |l| l.as_f64());
    if lo <= hi { (lo, hi) } else { (hi, lo) }
}

/// The bound an open float range stops at, either side.
const FLOAT_DEFAULT_BOUND: f64 = 1_000_000.0;

macro_rules! impl_unique_float_column {
    ($fn_name:ident, $polars_type:ident, $native_type:ty, $default_bound:expr) => {
        fn $fn_name(
            plan: &ColumnPlan,
            n: usize,
            seed: ColumnSeed,
        ) -> Result<ChunkedArray<$polars_type>, String> {
            let (lo, hi) = float_bounds(plan, $default_bound);
            let (domain, point) = float_grid(lo, hi, <$native_type>::EPSILON as f64);
            permuted_column(plan, n, seed, domain, |i| point(i) as $native_type)
        }
    };
}

impl_unique_float_column!(unique_float32, Float32Type, f32, FLOAT_DEFAULT_BOUND);
impl_unique_float_column!(unique_float64, Float64Type, f64, FLOAT_DEFAULT_BOUND);

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

/// A plain string space ordered evenly over its lengths stops growing once it
/// holds this many values: more rows than a Polars column holds, so no frame
/// is refused for it, and few enough that the shortest lengths still share.
const EVEN_ROOM: u128 = 1 << 32;

/// The digits a `u128` holds in base 62, 62^21 < 2^128 < 62^22: past these
/// a string's leading characters come from a hash of the rest.
const BASE62_DIGITS: usize = 22;

/// 62^10, the most base-62 digits a `u64` piece is written in at once.
const BASE62_10: u128 = 839_299_365_868_340_224;

/// Part of the even ordering: indices `[start, ...)` take the next members of
/// lengths `first_length..` in turn, from member `member` on.
struct Tier {
    start: u128,
    member: u128,
    first_length: usize,
    lengths: u128,
}

/// The unique plain strings of `min..=max` characters, ordered so that every
/// prefix of the space holds each length alike -- member 0 of every length,
/// then member 1 of every length, and so on, a length dropping out once its
/// members are spent. The permutation runs over the shortest prefix of whole
/// tiers holding `EVEN_ROOM` values (or the whole space, if smaller), so a
/// column draws its lengths evenly, as one that is not unique does, while its
/// shortest length has room; a length too short to share drops out, and
/// lengths too small for the frame leave it to the longer ones.
///
/// A length's members are taken in an order of their own, a permutation of
/// its `62^len` strings, so the members a prefix uses spread over all of them
/// rather than sharing their leading characters.
struct EvenLengths {
    min: usize,
    max: usize,
    tiers: Vec<Tier>,
    domain: u128,
    within: Vec<Permutation>,
}

impl EvenLengths {
    fn new(min: usize, max: usize, seed: u64) -> Self {
        let max = max.max(min);
        let radix = CHARSET.len() as u128;
        let sizes: Vec<u128> = (min..=max)
            .map(|len| radix.saturating_pow(len as u32))
            .collect();
        let mut tiers = Vec::new();
        let (mut start, mut member) = (0u128, 0u128);
        for (j, size) in sizes.iter().enumerate() {
            let lengths = (sizes.len() - j) as u128;
            tiers.push(Tier {
                start,
                member,
                first_length: j,
                lengths,
            });
            start = start.saturating_add((size - member).saturating_mul(lengths));
            member = *size;
            if start >= EVEN_ROOM {
                break;
            }
        }
        let within = sizes
            .iter()
            .enumerate()
            .map(|(j, size)| {
                Permutation::new(*size, seed ^ (j as u64 + 1).wrapping_mul(LENGTH_KEY))
            })
            .collect();
        EvenLengths {
            min,
            max,
            tiers,
            domain: start,
            within,
        }
    }

    /// The length (from `min`) and member at `index`, below the domain.
    #[inline]
    fn locate(&self, index: u128) -> (usize, u128) {
        let tier = self
            .tiers
            .iter()
            .rev()
            .find(|t| t.start <= index)
            .expect("the first tier starts at 0");
        let offset = index - tier.start;
        let (member, length) = if offset <= u64::MAX as u128 {
            let (o, l) = (offset as u64, tier.lengths as u64);
            ((o / l) as u128, (o % l) as usize)
        } else {
            (offset / tier.lengths, (offset % tier.lengths) as usize)
        };
        (tier.first_length + length, tier.member + member)
    }
}

/// Keys each length's permutation apart from the others' and the rows'.
const LENGTH_KEY: u64 = 0xD1B5_4A32_D192_ED03;

impl Enumerable for EvenLengths {
    fn cardinality(&self) -> u128 {
        self.domain
    }

    fn decode(&self, index: u128, out: &mut Vec<u8>) {
        let (j, member) = self.locate(index);
        let mut value = self.within[j].apply(member);
        let width = self.min + j;
        out.clear();
        out.resize(width, CHARSET[0]);
        // The digits the length's space reaches, right to left: ten at a
        // time while the value is wider than a `u64`, then in `u64`
        // arithmetic. A space of 62^22 or more saturates a `u128`, so the
        // characters before its last 22 come from a hash of the value
        // instead; the digits make the string unique.
        let reach = width - width.min(BASE62_DIGITS);
        let mut end = width;
        while value > u64::MAX as u128 {
            write_base62(&mut out[end - 10..end], (value % BASE62_10) as u64);
            value /= BASE62_10;
            end -= 10;
        }
        write_base62(&mut out[reach..end], value as u64);
        let mut h = splitmix(member as u64 ^ ((member >> 64) as u64).rotate_left(29));
        for slot in out[..reach].iter_mut() {
            *slot = CHARSET[(h % 62) as usize];
            h = splitmix(h);
        }
    }

    fn max_bytes(&self) -> usize {
        self.max
    }
}

/// `value`'s low `out.len()` base-62 digits, most significant first.
#[inline]
fn write_base62(out: &mut [u8], mut value: u64) {
    for slot in out.iter_mut().rev() {
        *slot = CHARSET[(value % 62) as usize];
        value /= 62;
    }
}

#[inline]
fn splitmix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// A string column -- plain, a `format=` template or a `pattern` -- through
/// its value space, task by task like `sample::build_strings`.
fn unique_strings(
    plan: &ColumnPlan,
    n: usize,
    seed: ColumnSeed,
    space: &(dyn Enumerable + Sync),
) -> Result<StringChunked, String> {
    let name = PlSmallStr::from(plan.name.as_str());
    let domain = space.cardinality();
    let ranks = Ranks::new(plan, n, seed)?;
    if ranks.needed > domain {
        return Err(too_small(plan, ranks.needed, domain));
    }
    if n == 0 {
        return Ok(StringChunkedBuilder::new(name, 0).finish());
    }
    let permutation = (ranks.needed > 0).then(|| Permutation::new(domain, permutation_seed(seed)));
    let task = task_rows(n);

    let arrays: Vec<Utf8ViewArray> = (0..n.div_ceil(task))
        .into_par_iter()
        .map(|t| {
            let start = t * task;
            let len = (start + task).min(n) - start;
            let mut views = Views::with_capacity(len, ranks.validity.is_some(), 0);
            let mut scratch: Vec<u8> = Vec::with_capacity(space.max_bytes());
            let mut next = start;
            let Some(permutation) = permutation.as_ref() else {
                (start..start + len).for_each(|_| views.push_null());
                return views.finish();
            };
            ranks.for_each_value(start..start + len, permutation, |row, value| {
                for _ in next..row {
                    views.push_null();
                }
                next = row + 1;
                space.decode(value, &mut scratch);
                // SAFETY: a template's parts are ASCII or a `String`, a
                // pattern's units are each a `str`, and a plain string is
                // `CHARSET`, so the buffer is valid UTF-8 -- see
                // `format::Template::compile` and `pattern.rs`.
                views.push(unsafe { std::str::from_utf8_unchecked(&scratch) });
            });
            for _ in next..start + len {
                views.push_null();
            }
            views.finish()
        })
        .collect();

    Ok(StringChunked::from_chunk_iter(name, arrays))
}

/// Rows of a unique column, from where `seed`'s first chunk starts.
pub fn generate_unique_series(
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
        Kind::Int128 => unique_int128(plan, n, seed)?.into_series(),
        Kind::UInt128 => unique_uint128(plan, n, seed)?.into_series(),
        Kind::UInt32 => unique_uint32(plan, n, seed)?.into_series(),
        Kind::UInt16 => unique_uint16(plan, n, seed)?.into_series(),
        Kind::UInt8 => unique_uint8(plan, n, seed)?.into_series(),
        Kind::Float64 => unique_float64(plan, n, seed)?.into_series(),
        Kind::Float32 => unique_float32(plan, n, seed)?.into_series(),
        Kind::Bool => unique_bool(plan, n, seed)?.into_series(),
        Kind::Index => unique_index(plan, n, seed)?.into_series(),
        Kind::String => {
            let space =
                EvenLengths::new(plan.str_min_len, plan.str_max_len, permutation_seed(seed));
            unique_strings(plan, n, seed, &space)?.into_series()
        }
        Kind::Template => {
            let template = plan.template.as_ref().ok_or_else(|| {
                format!("Column '{}' has kind 'template' but no template", plan.name)
            })?;
            unique_strings(plan, n, seed, template)?.into_series()
        }
        Kind::Pattern => {
            let pattern = plan.pattern.as_ref().ok_or_else(|| {
                format!("Column '{}' has kind 'pattern' but no pattern", plan.name)
            })?;
            unique_strings(plan, n, seed, pattern.as_ref())?.into_series()
        }
    })
}

// ---------------------------------------------------------------------------
// A unique column's values at given rows
// ---------------------------------------------------------------------------

/// `decode(π(row))` at each of `rows`: what a unique column with no nulls
/// holds at those rows of the whole frame, wherever they sit -- a parent a
/// child points at, read without drawing the rows between.
fn decoded_at<N, F>(
    plan: &ColumnPlan,
    rows: &[u64],
    seed: u64,
    domain: u128,
    decode: F,
) -> Result<Vec<N>, String>
where
    N: Send,
    F: Fn(u128) -> N + Sync,
{
    let Some(&last) = rows.iter().max() else {
        return Ok(Vec::new());
    };
    let needed = last as u128 + 1;
    if needed > domain {
        return Err(too_small(plan, needed, domain));
    }
    let key = permutation_seed(ColumnSeed {
        base: seed,
        first_chunk: 0,
    });
    let permutation = Permutation::new(domain, key);
    Ok(rows
        .par_iter()
        .map(|&row| decode(permutation.apply(row as u128)))
        .collect())
}

fn strings_at(
    plan: &ColumnPlan,
    rows: &[u64],
    seed: u64,
    space: &(dyn Enumerable + Sync),
) -> Result<Series, String> {
    let name = PlSmallStr::from(plan.name.as_str());
    let domain = space.cardinality();
    let Some(&last) = rows.iter().max() else {
        return Ok(StringChunkedBuilder::new(name, 0).finish().into_series());
    };
    if last as u128 + 1 > domain {
        return Err(too_small(plan, last as u128 + 1, domain));
    }
    let key = permutation_seed(ColumnSeed {
        base: seed,
        first_chunk: 0,
    });
    let permutation = Permutation::new(domain, key);
    // Task by task straight into each task's view array, as `unique_strings`
    // builds a whole column: no `String` per value.
    let arrays: Vec<Utf8ViewArray> = rows
        .par_chunks(task_rows(rows.len()))
        .map(|chunk| {
            let mut views = Views::with_capacity(chunk.len(), false, 0);
            let mut scratch: Vec<u8> = Vec::with_capacity(space.max_bytes());
            for &row in chunk {
                space.decode(permutation.apply(row as u128), &mut scratch);
                // SAFETY: as in `unique_strings`, every space decodes to UTF-8.
                views.push(unsafe { std::str::from_utf8_unchecked(&scratch) });
            }
            views.finish()
        })
        .collect();
    Ok(StringChunked::from_chunk_iter(name, arrays).into_series())
}

/// A unique column's values at `rows` of the whole frame, `seed` its column
/// seed: the column `generate_unique_series` fills, gathered at `rows`, for
/// rows anywhere and in any order. A column that draws nulls is refused: a
/// row's value is `π` of how many rows before it hold one, which only the
/// rows before it tell.
pub fn generate_unique_at(plan: &ColumnPlan, seed: u64, rows: &[u64]) -> Result<Series, String> {
    if draws_nulls(plan) {
        return Err(format!(
            "Column '{}' draws nulls, so its values at given rows depend on the \
             rows before them; only a column with no nulls is read at rows.",
            plan.name
        ));
    }
    let name = PlSmallStr::from(plan.name.as_str());
    macro_rules! ints {
        ($t:ty, $polars_type:ident) => {{
            let (lo, domain) = int_space(plan, <$t>::MIN as i128, <$t>::MAX as i128);
            let values = decoded_at(plan, rows, seed, domain, |i| (lo + i as i128) as $t)?;
            ChunkedArray::<$polars_type>::from_vec(name, values).into_series()
        }};
    }
    macro_rules! floats {
        ($t:ty, $polars_type:ident) => {{
            let (lo, hi) = float_bounds(plan, FLOAT_DEFAULT_BOUND);
            let (domain, point) = float_grid(lo, hi, <$t>::EPSILON as f64);
            let values = decoded_at(plan, rows, seed, domain, |i| point(i) as $t)?;
            ChunkedArray::<$polars_type>::from_vec(name, values).into_series()
        }};
    }
    let column_seed = ColumnSeed {
        base: seed,
        first_chunk: 0,
    };
    Ok(match plan.kind {
        Kind::Int64 => ints!(i64, Int64Type),
        Kind::Int32 => ints!(i32, Int32Type),
        Kind::Int16 => ints!(i16, Int16Type),
        Kind::Int8 => ints!(i8, Int8Type),
        Kind::UInt64 => ints!(u64, UInt64Type),
        Kind::UInt32 => ints!(u32, UInt32Type),
        Kind::UInt16 => ints!(u16, UInt16Type),
        Kind::UInt8 => ints!(u8, UInt8Type),
        Kind::Int128 => {
            let (lo, domain) = int128_space(plan);
            let values = decoded_at(plan, rows, seed, domain, |i| lo.wrapping_add(i as i128))?;
            ChunkedArray::<Int128Type>::from_vec(name, values).into_series()
        }
        Kind::UInt128 => {
            let (lo, domain) = uint128_space(plan);
            let values = decoded_at(plan, rows, seed, domain, |i| lo + i)?;
            ChunkedArray::<UInt128Type>::from_vec(name, values).into_series()
        }
        Kind::Float64 => floats!(f64, Float64Type),
        Kind::Float32 => floats!(f32, Float32Type),
        Kind::Bool => {
            let values = decoded_at(plan, rows, seed, 2, |i| i == 1)?;
            BooleanChunked::from_slice(name, &values).into_series()
        }
        Kind::Index => {
            let domain = plan.n_categories.unwrap_or(0) as u128;
            let values = decoded_at(plan, rows, seed, domain, |i| i as u32)?;
            UInt32Chunked::from_vec(name, values).into_series()
        }
        Kind::String => {
            let space = EvenLengths::new(
                plan.str_min_len,
                plan.str_max_len,
                permutation_seed(column_seed),
            );
            strings_at(plan, rows, seed, &space)?
        }
        Kind::Template => {
            let template = plan.template.as_ref().ok_or_else(|| {
                format!("Column '{}' has kind 'template' but no template", plan.name)
            })?;
            strings_at(plan, rows, seed, template)?
        }
        Kind::Pattern => {
            let pattern = plan.pattern.as_ref().ok_or_else(|| {
                format!("Column '{}' has kind 'pattern' but no pattern", plan.name)
            })?;
            strings_at(plan, rows, seed, pattern.as_ref())?
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::{Limit, PlanArgs};
    use crate::sample::generate_series;
    use rand::SeedableRng;

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
        let whole_n = SEED_CHUNK * 3 + 500;
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
                (SEED_CHUNK - 3, 10),
                (SEED_CHUNK + 1234, SEED_CHUNK * 2 - 2000),
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

    fn template_plan(parts: &[(&str, &[&str], usize, usize)]) -> ColumnPlan {
        let raw: Vec<crate::format::RawPart> = parts
            .iter()
            .map(|(kind, strings, lo, hi)| {
                (
                    kind.to_string(),
                    strings.iter().map(|s| s.to_string()).collect(),
                    *lo,
                    *hi,
                )
            })
            .collect();
        ColumnPlan::build(PlanArgs {
            name: "c".into(),
            kind: "template",
            template: Some(&raw),
            unique: true,
            ..Default::default()
        })
        .expect("valid plan")
    }

    #[test]
    fn a_string_or_template_column_is_a_window_too() {
        let whole_n = SEED_CHUNK * 3 + 300;
        for p in [
            plan("string", None, None, None),
            template_plan(&[
                ("chars", &["abcdefghijklmnopqrstuvwxyz"], 3, 8),
                ("lit", &["@example."], 0, 0),
                ("one_of", &["com", "org", "net"], 0, 0),
            ]),
        ] {
            let whole = column(&p, whole_n, 21).unwrap();
            assert_eq!(distinct_count(&whole), whole_n, "{:?} repeated", p.kind);
            for (offset, n) in [(0, 10), (SEED_CHUNK - 3, 10), (SEED_CHUNK + 77, 5_000)] {
                let window = generate_series(&p, n, 21, offset).unwrap();
                assert!(window.equals_missing(&whole.slice(offset as i64, n)));
            }
        }
    }

    #[test]
    fn a_template_smaller_than_the_frame_names_the_column() {
        // "a" or "b", one or two wide: six strings in all.
        let p = template_plan(&[("chars", &["ab"], 1, 2)]);
        let s = column(&p, 6, 3).unwrap();
        assert_eq!(distinct_count(&s), 6);
        let err = column(&p, 7, 3).unwrap_err();
        assert!(
            err.contains("6 distinct value(s)") && err.contains("7 are needed"),
            "{err}"
        );
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
    fn a_unique_128_bit_column_is_distinct_over_its_full_range() {
        for (kind, lo, hi) in [
            (
                "int128",
                Some(Limit::Wide(i128::MIN)),
                Some(Limit::Wide(i128::MAX)),
            ),
            ("uint128", None, None),
            (
                "uint128",
                Some(Limit::WideU(u128::MAX - 99)),
                Some(Limit::WideU(u128::MAX)),
            ),
        ] {
            let p = plan(kind, lo, hi, None);
            let n = if kind == "uint128" && lo.is_some() {
                100
            } else {
                SEED_CHUNK * 2 + 9
            };
            let s = column(&p, n, 4).unwrap();
            assert_eq!(distinct_count(&s), n, "{kind}");
        }
        // A hundred values hold a hundred rows and no more.
        let p = plan(
            "uint128",
            Some(Limit::WideU(u128::MAX - 99)),
            Some(Limit::WideU(u128::MAX)),
            None,
        );
        assert!(column(&p, 101, 4).is_err());
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
    }

    #[test]
    fn a_null_spends_no_value() {
        // Half the rows null: 100 rows draw about 50 values from a domain of
        // 60, and a frame is refused only once its values outgrow it.
        let p = nullable("int64", Some(Limit::Int(1)), Some(Limit::Int(60)), 0.5);
        let s = column(&p, 100, 9).unwrap();
        let present = s.drop_nulls();
        assert!(present.len() <= 60, "{} values", present.len());
        assert_eq!(distinct_count(&present), present.len());
        let err = column(&p, 1_000, 9).unwrap_err();
        assert!(err.contains("60 distinct value(s)"), "{err}");
        // Every value once, when the values exactly fill the domain.
        let filled = (1..400)
            .map(|n| column(&p, n, 9).unwrap())
            .find(|s| s.len() - s.null_count() == 60)
            .expect("some frame holds exactly 60 values");
        let mut values: Vec<i64> = filled.i64().unwrap().iter().flatten().collect();
        values.sort_unstable();
        assert_eq!(values, (1..=60).collect::<Vec<i64>>());
    }

    #[test]
    fn a_nullable_column_is_unique_across_chunks_and_a_window_is_its_slice() {
        let whole_n = SEED_CHUNK * 5 + 321;
        for p in [
            nullable("int64", None, None, 0.3),
            nullable("uint32", None, None, 0.97),
            nullable("int32", Some(Limit::Int(0)), Some(Limit::Int(9_000)), 0.6),
            ColumnPlan::build(PlanArgs {
                name: "c".into(),
                kind: "string",
                nullable: true,
                null_probability: 0.25,
                str_min_len: Some(1),
                str_max_len: Some(3),
                unique: true,
                ..Default::default()
            })
            .unwrap(),
        ] {
            let whole = column(&p, whole_n, 4).unwrap();
            let present = whole.drop_nulls();
            assert_eq!(distinct_count(&present), present.len(), "{:?}", p.kind);
            for (offset, n) in [
                (0, 10),
                (SEED_CHUNK - 3, 10),
                (SEED_CHUNK * 2, SEED_CHUNK),
                (SEED_CHUNK * 3 + 17, SEED_CHUNK + 100),
                (whole_n - 5, 5),
            ] {
                let window = generate_series(&p, n, 4, offset).unwrap();
                assert!(
                    window.equals_missing(&whole.slice(offset as i64, n)),
                    "{:?} at offset {offset} for {n} rows",
                    p.kind
                );
            }
        }
    }

    #[test]
    fn a_chunks_null_count_is_the_binomial_its_rows_give() {
        for p in [0.0001, 0.02, 0.2, 0.5, 0.93, 1.0] {
            let table = ChunkNulls::new(p);
            let draws = 20_000;
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(1);
            let counts: Vec<f64> = (0..draws).map(|_| table.count(&mut rng) as f64).collect();
            let mean = counts.iter().sum::<f64>() / draws as f64;
            let var = counts.iter().map(|c| (c - mean).powi(2)).sum::<f64>() / draws as f64;
            let n = SEED_CHUNK as f64;
            let (m, v) = (n * p, n * p * (1.0 - p));
            // Five standard errors of the mean, and of the variance.
            assert!(
                (mean - m).abs() <= 5.0 * (v / draws as f64).sqrt() + 1e-9,
                "p {p}: mean {mean}, expected {m}"
            );
            assert!(
                (var - v).abs() <= 5.0 * v * (2.0 / draws as f64).sqrt() + 1e-9,
                "p {p}: variance {var}, expected {v}"
            );
            assert!(counts.iter().all(|c| *c <= n));
        }
    }

    #[test]
    fn nulls_are_placed_evenly_and_a_short_chunk_is_the_start_of_a_whole_one() {
        let mut hits = vec![0u32; SEED_CHUNK];
        for s in 0..400 {
            let mut whole = vec![0u8; VALIDITY_BYTES_PER_CHUNK];
            let values = place_nulls(
                &mut whole,
                SEED_CHUNK,
                1_000,
                &mut Xoshiro256PlusPlus::seed_from_u64(s),
            );
            assert_eq!(values, SEED_CHUNK - 1_000);
            let mut short = vec![0u8; 100];
            place_nulls(
                &mut short,
                800,
                1_000,
                &mut Xoshiro256PlusPlus::seed_from_u64(s),
            );
            assert_eq!(short, whole[..100]);
            for (row, hit) in hits.iter_mut().enumerate() {
                *hit += u32::from(whole[row / 8] >> (row % 8) & 1 == 0);
            }
        }
        // Each eighth of the chunk holds an eighth of the nulls.
        let expected = 400.0 * 1_000.0 / 8.0;
        for eighth in hits.chunks(SEED_CHUNK / 8) {
            let got: u32 = eighth.iter().sum();
            assert!(
                (got as f64 - expected).abs() < 0.03 * expected,
                "{got} nulls in an eighth, expected about {expected}"
            );
        }
    }

    #[test]
    fn a_column_of_nulls_alone_needs_no_values() {
        // One value in the domain, ten rows, all null: nothing is drawn.
        let p = nullable("int64", Some(Limit::Int(5)), Some(Limit::Int(5)), 1.0);
        let s = column(&p, 10, 1).unwrap();
        assert_eq!(s.null_count(), 10);
    }

    fn string_plan(min: usize, max: usize) -> ColumnPlan {
        ColumnPlan::build(PlanArgs {
            name: "c".into(),
            kind: "string",
            str_min_len: Some(min),
            str_max_len: Some(max),
            unique: true,
            ..Default::default()
        })
        .unwrap()
    }

    fn lengths(s: &Series) -> Vec<usize> {
        s.str().unwrap().iter().flatten().map(|v| v.len()).collect()
    }

    #[test]
    fn a_unique_string_draws_its_lengths_evenly() {
        for (min, max) in [(8, 12), (5, 15), (20, 30)] {
            let s = column(&string_plan(min, max), 50_000, 3).unwrap();
            assert_eq!(distinct_count(&s), 50_000);
            let mut counts = vec![0usize; max - min + 1];
            for len in lengths(&s) {
                counts[len - min] += 1;
            }
            let expected = 50_000.0 / counts.len() as f64;
            for (i, c) in counts.iter().enumerate() {
                assert!(
                    (*c as f64 - expected).abs() < 0.08 * expected,
                    "({min}, {max}): {c} strings of {} characters, expected {expected}",
                    min + i
                );
            }
        }
    }

    #[test]
    fn a_small_string_space_is_drawn_whole() {
        // 62 + 62^2 strings of one or two characters: all of them, once.
        let p = string_plan(1, 2);
        let s = column(&p, 62 + 62 * 62, 8).unwrap();
        assert_eq!(distinct_count(&s), 62 + 62 * 62);
        assert!(column(&p, 62 + 62 * 62 + 1, 8).is_err());
        // Zero characters is a length too: the one empty string.
        let s = column(&string_plan(0, 1), 63, 8).unwrap();
        assert_eq!(distinct_count(&s), 63);
    }

    #[test]
    fn a_strings_characters_all_vary() {
        // No position of a long or short string is stuck on one character:
        // each length's members are spread over its whole space.
        for (min, max) in [(12, 12), (8, 12), (30, 40)] {
            let s = column(&string_plan(min, max), 5_000, 6).unwrap();
            let values: Vec<&str> = s.str().unwrap().iter().flatten().collect();
            for position in 0..min {
                let seen: std::collections::HashSet<u8> =
                    values.iter().map(|v| v.as_bytes()[position]).collect();
                assert!(
                    seen.len() > 40,
                    "({min}, {max}): position {position} holds {} characters",
                    seen.len()
                );
            }
        }
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

    #[test]
    fn values_at_rows_are_the_whole_column_gathered() {
        // Rows past the first seed chunk, out of order and repeated.
        let n = SEED_CHUNK * 2 + 77;
        let rows: Vec<u64> = vec![
            (n - 1) as u64,
            0,
            (SEED_CHUNK + 3) as u64,
            5,
            (SEED_CHUNK + 3) as u64,
        ];
        let mut plans = vec![
            plan("int64", None, None, None),
            plan(
                "int16",
                Some(Limit::Int(-10_000)),
                Some(Limit::Int(10_000)),
                None,
            ),
            plan("uint64", Some(Limit::Int(10)), None, None),
            plan("int128", None, None, None),
            plan("uint128", None, None, None),
            plan(
                "float64",
                Some(Limit::Float(0.0)),
                Some(Limit::Float(1.0)),
                None,
            ),
            plan("float32", None, None, None),
            plan("index", None, None, Some(n + 10)),
            string_plan(3, 9),
        ];
        plans.push(
            ColumnPlan::build(PlanArgs {
                name: "c".into(),
                kind: "pattern",
                pattern: Some(r"^ORD-[0-9]{6}$"),
                unique: true,
                ..Default::default()
            })
            .unwrap(),
        );
        for p in &plans {
            let whole = column(p, n, 17).unwrap();
            let at = generate_unique_at(p, 17, &rows).unwrap();
            let indices = IdxCa::from_vec("i".into(), rows.iter().map(|&r| r as IdxSize).collect());
            let want = whole.take(&indices).unwrap();
            assert!(at.equals(&want), "{:?}: {at:?} vs {want:?}", p.kind);
        }
        // The two booleans, at both rows a unique boolean holds.
        let b = plan("bool", None, None, None);
        let whole = column(&b, 2, 4).unwrap();
        let at = generate_unique_at(&b, 4, &[1, 0]).unwrap();
        let want = whole
            .take(&IdxCa::from_vec("i".into(), vec![1, 0]))
            .unwrap();
        assert!(at.equals(&want));
    }

    #[test]
    fn values_at_rows_refuse_a_row_past_the_domain_and_a_nullable_column() {
        let small = plan("int64", Some(Limit::Int(1)), Some(Limit::Int(10)), None);
        let err = generate_unique_at(&small, 1, &[3, 10]).unwrap_err();
        assert!(
            err.contains("10 distinct") && err.contains("11 are needed"),
            "{err}"
        );
        let nullable = nullable("int64", None, None, 0.2);
        let err = generate_unique_at(&nullable, 1, &[0]).unwrap_err();
        assert!(err.contains("draws nulls"), "{err}");
        assert_eq!(generate_unique_at(&small, 1, &[]).unwrap().len(), 0);
    }
}

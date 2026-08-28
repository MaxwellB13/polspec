use std::collections::HashMap;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;
use polars::prelude::*;
use polars_core::chunked_array::builder::{BooleanChunkedBuilder, PrimitiveChunkedBuilder, StringChunkedBuilder};
use rand::distributions::{Distribution, Uniform, WeightedIndex};
use rand::prelude::*;
use rand_distr::{Beta, Exp, Gamma, LogNormal, Normal, Poisson};
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;

const CHARSET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
const DEFAULT_WIDE_INT_BOUND: i64 = 1_000_000;
const DEFAULT_WIDE_UINT_BOUND: u64 = 1_000_000;
const DEFAULT_FLOAT_BOUND: f64 = 1_000_000.0;
const DEFAULT_FLOAT32_BOUND: f32 = 1_000_000.0;
const DEFAULT_STR_MIN_LEN: i64 = 5;
const DEFAULT_STR_MAX_LEN: i64 = 15;
const MIN_CHUNK: usize = 4096;

/// One column's generation instructions, sent over from the Python side.
struct ColumnSpec {
    name: String,
    kind: String,
    nullable: bool,
    null_probability: f64,
    min: Option<f64>,
    max: Option<f64>,
    categories: Option<Vec<String>>,
    weights: Option<Vec<f64>>,
    str_min_len: Option<i64>,
    str_max_len: Option<i64>,
    distribution: Option<String>,
    distribution_params: Option<HashMap<String, f64>>,
}

type ColumnSpecTuple = (
    String,
    String,
    bool,
    f64,
    Option<f64>,
    Option<f64>,
    Option<Vec<String>>,
    Option<Vec<f64>>,
    Option<i64>,
    Option<i64>,
    Option<String>,
    Option<HashMap<String, f64>>,
);

impl From<ColumnSpecTuple> for ColumnSpec {
    fn from(t: ColumnSpecTuple) -> Self {
        ColumnSpec {
            name: t.0,
            kind: t.1,
            nullable: t.2,
            null_probability: t.3,
            min: t.4,
            max: t.5,
            categories: t.6,
            weights: t.7,
            str_min_len: t.8,
            str_max_len: t.9,
            distribution: t.10,
            distribution_params: t.11,
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum DistKind {
    Uniform,
    Normal(Normal<f64>),
    LogNormal(LogNormal<f64>),
    Exp(Exp<f64>),
    Poisson(Poisson<f64>),
    Gamma(Gamma<f64>),
    Beta(Beta<f64>),
}

impl DistKind {
    fn from_spec(spec: &ColumnSpec) -> PyResult<Self> {
        let dist_name = match &spec.distribution {
            Some(d) => d.trim().to_ascii_lowercase(),
            None => return Ok(DistKind::Uniform),
        };
        let params = spec.distribution_params.as_ref();
        let get_param = |keys: &[&str], default: f64| -> f64 {
            if let Some(map) = params {
                for k in keys {
                    if let Some(&v) = map.get(*k) {
                        return v;
                    }
                }
            }
            default
        };

        match dist_name.as_str() {
            "" | "uniform" => Ok(DistKind::Uniform),
            "normal" => {
                let mean = get_param(&["mean", "mu", "loc"], 0.0);
                let std = get_param(&["std", "sigma", "scale"], 1.0);
                if std <= 0.0 {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "Normal distribution standard deviation for column '{}' must be positive, got {std}",
                        spec.name
                    )));
                }
                Normal::new(mean, std).map(DistKind::Normal).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "Invalid normal distribution parameters for column '{}': {e}",
                        spec.name
                    ))
                })
            }
            "lognormal" => {
                let mean = get_param(&["mean", "mu", "meanlog"], 0.0);
                let std = get_param(&["std", "sigma", "sdlog"], 1.0);
                if std <= 0.0 {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "LogNormal distribution standard deviation for column '{}' must be positive, got {std}",
                        spec.name
                    )));
                }
                LogNormal::new(mean, std).map(DistKind::LogNormal).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "Invalid lognormal distribution parameters for column '{}': {e}",
                        spec.name
                    ))
                })
            }
            "exponential" | "exp" => {
                let rate = if let Some(map) = params {
                    if let Some(&scale) = map.get("scale") {
                        if scale <= 0.0 {
                            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                                "Exponential distribution scale for column '{}' must be positive, got {scale}",
                                spec.name
                            )));
                        }
                        1.0 / scale
                    } else {
                        get_param(&["rate", "lambda", "lambda_"], 1.0)
                    }
                } else {
                    1.0
                };
                Exp::new(rate).map(DistKind::Exp).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "Invalid exponential distribution parameters for column '{}': {e}",
                        spec.name
                    ))
                })
            }
            "poisson" => {
                let lambda = get_param(&["lambda", "lambda_", "rate", "mean"], 1.0);
                Poisson::new(lambda).map(DistKind::Poisson).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "Invalid poisson distribution parameters for column '{}': {e}",
                        spec.name
                    ))
                })
            }
            "gamma" => {
                let shape = get_param(&["shape", "alpha", "k"], 1.0);
                let scale = get_param(&["scale", "beta", "theta"], 1.0);
                Gamma::new(shape, scale).map(DistKind::Gamma).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "Invalid gamma distribution parameters for column '{}': {e}",
                        spec.name
                    ))
                })
            }
            "beta" => {
                let alpha = get_param(&["alpha", "a", "shape1"], 1.0);
                let beta = get_param(&["beta", "b", "shape2"], 1.0);
                Beta::new(alpha, beta).map(DistKind::Beta).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "Invalid beta distribution parameters for column '{}': {e}",
                        spec.name
                    ))
                })
            }
            other => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Unsupported distribution '{other}' for column '{}'",
                spec.name
            ))),
        }
    }

    #[inline(always)]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        match self {
            DistKind::Uniform => unreachable!(),
            DistKind::Normal(d) => d.sample(rng),
            DistKind::LogNormal(d) => d.sample(rng),
            DistKind::Exp(d) => d.sample(rng),
            DistKind::Poisson(d) => d.sample(rng),
            DistKind::Gamma(d) => d.sample(rng),
            DistKind::Beta(d) => d.sample(rng),
        }
    }
}

#[derive(Clone)]
enum CategorySampler {
    Uniform(Uniform<usize>),
    Weighted(WeightedIndex<f64>),
}

impl CategorySampler {
    fn new(categories_len: usize, weights: Option<&Vec<f64>>, col_name: &str) -> PyResult<Option<Self>> {
        if categories_len == 0 {
            return Ok(None);
        }
        if let Some(w) = weights {
            if w.len() != categories_len {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Length of weights ({}) must match number of categories ({}) for column '{}'",
                    w.len(),
                    categories_len,
                    col_name
                )));
            }
            let weighted = WeightedIndex::new(w).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Invalid weights for column '{}': {e}",
                    col_name
                ))
            })?;
            Ok(Some(CategorySampler::Weighted(weighted)))
        } else {
            Ok(Some(CategorySampler::Uniform(Uniform::new(0, categories_len))))
        }
    }

    #[inline(always)]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> usize {
        match self {
            CategorySampler::Uniform(u) => u.sample(rng),
            CategorySampler::Weighted(w) => w.sample(rng),
        }
    }
}

/// Splits `n` rows into roughly-equal, reproducibly-seeded chunks so a
/// column can be filled in parallel while staying deterministic for a
/// given top-level seed.
fn chunk_size_for(n: usize) -> usize {
    let threads = rayon::current_num_threads().max(1);
    (n / threads).max(1).max(MIN_CHUNK)
}

fn seed_for_chunk(base_seed: u64, chunk_index: usize) -> u64 {
    base_seed ^ (chunk_index as u64).wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(0x1)
}

macro_rules! impl_gen_int_column {
    ($fn_name:ident, $polars_type:ident, $native_type:ty, $default_min:expr, $default_max:expr) => {
        fn $fn_name(spec: &ColumnSpec, dist_kind: DistKind, n: usize, seed: u64) -> ChunkedArray<$polars_type> {
            let name = PlSmallStr::from(spec.name.as_str());
            if n == 0 {
                return PrimitiveChunkedBuilder::<$polars_type>::new(name, 0).finish();
            }
            let lo = spec.min.map(|v| v as $native_type).unwrap_or($default_min);
            let hi = spec.max.map(|v| v as $native_type).unwrap_or($default_max);
            let (lo, hi) = if lo <= hi { (lo, hi) } else { (hi, lo) };
            let uniform_dist = Uniform::new_inclusive(lo, hi);
            let chunk_size = chunk_size_for(n);
            let n_chunks = n.div_ceil(chunk_size.max(1)).max(1);

            let mut per_chunk: Vec<ChunkedArray<$polars_type>> = (0..n_chunks)
                .into_par_iter()
                .map(|i| {
                    let start = i * chunk_size;
                    let end = (start + chunk_size).min(n);
                    let len = end - start;
                    let mut builder = PrimitiveChunkedBuilder::<$polars_type>::new(name.clone(), len);
                    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed_for_chunk(seed, i));

                    let sample_val = |rng: &mut Xoshiro256PlusPlus| -> $native_type {
                        match dist_kind {
                            DistKind::Uniform => uniform_dist.sample(rng),
                            _ => {
                                let val = dist_kind.sample(rng);
                                let rounded = if val.is_nan() {
                                    lo as f64
                                } else {
                                    val.round()
                                };
                                let clamped = rounded.clamp(lo as f64, hi as f64);
                                clamped as $native_type
                            }
                        }
                    };

                    if spec.nullable {
                        let null_p = spec.null_probability;
                        for _ in 0..len {
                            if rng.gen_bool(null_p) {
                                builder.append_null();
                            } else {
                                builder.append_value(sample_val(&mut rng));
                            }
                        }
                    } else {
                        for _ in 0..len {
                            builder.append_value(sample_val(&mut rng));
                        }
                    }
                    builder.finish()
                })
                .collect();

            let mut result = per_chunk.remove(0);
            for ca in per_chunk {
                result.append_owned(ca).expect("all chunks share the same dtype");
            }
            result
        }
    };
}

impl_gen_int_column!(gen_int8_column, Int8Type, i8, i8::MIN, i8::MAX);
impl_gen_int_column!(gen_int16_column, Int16Type, i16, i16::MIN, i16::MAX);
impl_gen_int_column!(gen_int32_column, Int32Type, i32, i32::MIN, i32::MAX);
impl_gen_int_column!(gen_int64_column, Int64Type, i64, -DEFAULT_WIDE_INT_BOUND, DEFAULT_WIDE_INT_BOUND);
impl_gen_int_column!(gen_uint8_column, UInt8Type, u8, u8::MIN, u8::MAX);
impl_gen_int_column!(gen_uint16_column, UInt16Type, u16, u16::MIN, u16::MAX);
impl_gen_int_column!(gen_uint32_column, UInt32Type, u32, u32::MIN, u32::MAX);
impl_gen_int_column!(gen_uint64_column, UInt64Type, u64, 0, DEFAULT_WIDE_UINT_BOUND);

macro_rules! impl_gen_float_column {
    ($fn_name:ident, $polars_type:ident, $native_type:ty, $default_bound:expr) => {
        fn $fn_name(spec: &ColumnSpec, dist_kind: DistKind, n: usize, seed: u64) -> ChunkedArray<$polars_type> {
            let name = PlSmallStr::from(spec.name.as_str());
            if n == 0 {
                return PrimitiveChunkedBuilder::<$polars_type>::new(name, 0).finish();
            }
            let lo = spec.min.map(|v| v as $native_type).unwrap_or(-$default_bound);
            let hi = spec.max.map(|v| v as $native_type).unwrap_or($default_bound);
            let (lo, hi) = if lo <= hi { (lo, hi) } else { (hi, lo) };
            let uniform_dist = Uniform::new_inclusive(lo, hi);
            let chunk_size = chunk_size_for(n);
            let n_chunks = n.div_ceil(chunk_size.max(1)).max(1);

            let mut per_chunk: Vec<ChunkedArray<$polars_type>> = (0..n_chunks)
                .into_par_iter()
                .map(|i| {
                    let start = i * chunk_size;
                    let end = (start + chunk_size).min(n);
                    let len = end - start;
                    let mut builder = PrimitiveChunkedBuilder::<$polars_type>::new(name.clone(), len);
                    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed_for_chunk(seed, i));

                    let sample_val = |rng: &mut Xoshiro256PlusPlus| -> $native_type {
                        match dist_kind {
                            DistKind::Uniform => uniform_dist.sample(rng),
                            _ => {
                                let val = dist_kind.sample(rng) as $native_type;
                                if val.is_nan() {
                                    lo
                                } else {
                                    val.clamp(lo, hi)
                                }
                            }
                        }
                    };

                    if spec.nullable {
                        let null_p = spec.null_probability;
                        for _ in 0..len {
                            if rng.gen_bool(null_p) {
                                builder.append_null();
                            } else {
                                builder.append_value(sample_val(&mut rng));
                            }
                        }
                    } else {
                        for _ in 0..len {
                            builder.append_value(sample_val(&mut rng));
                        }
                    }
                    builder.finish()
                })
                .collect();

            let mut result = per_chunk.remove(0);
            for ca in per_chunk {
                result.append_owned(ca).expect("all chunks share the same dtype");
            }
            result
        }
    };
}

impl_gen_float_column!(gen_float32_column, Float32Type, f32, DEFAULT_FLOAT32_BOUND);
impl_gen_float_column!(gen_float64_column, Float64Type, f64, DEFAULT_FLOAT_BOUND);

fn gen_bool_column(spec: &ColumnSpec, p_true: f64, n: usize, seed: u64) -> BooleanChunked {
    let name = PlSmallStr::from(spec.name.as_str());
    if n == 0 {
        return BooleanChunkedBuilder::new(name, 0).finish();
    }
    let chunk_size = chunk_size_for(n);
    let n_chunks = n.div_ceil(chunk_size.max(1)).max(1);

    let mut per_chunk: Vec<BooleanChunked> = (0..n_chunks)
        .into_par_iter()
        .map(|i| {
            let start = i * chunk_size;
            let end = (start + chunk_size).min(n);
            let len = end - start;
            let mut builder = BooleanChunkedBuilder::new(name.clone(), len);
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed_for_chunk(seed, i));
            if spec.nullable {
                let null_p = spec.null_probability;
                for _ in 0..len {
                    if rng.gen_bool(null_p) {
                        builder.append_null();
                    } else {
                        builder.append_value(rng.gen_bool(p_true));
                    }
                }
            } else {
                for _ in 0..len {
                    builder.append_value(rng.gen_bool(p_true));
                }
            }
            builder.finish()
        })
        .collect();

    let mut result = per_chunk.remove(0);
    for ca in per_chunk {
        result.append_owned(ca).expect("all chunks share the same dtype");
    }
    result
}

fn gen_string_column(
    spec: &ColumnSpec,
    sampler: Option<CategorySampler>,
    n: usize,
    seed: u64,
) -> StringChunked {
    let name = PlSmallStr::from(spec.name.as_str());
    if n == 0 {
        return StringChunkedBuilder::new(name, 0).finish();
    }
    let categories = spec.categories.clone();
    let min_len = spec.str_min_len.unwrap_or(DEFAULT_STR_MIN_LEN).max(0) as usize;
    let max_len = spec.str_max_len.unwrap_or(DEFAULT_STR_MAX_LEN).max(min_len as i64) as usize;

    let len_dist = (max_len > min_len).then(|| Uniform::new_inclusive(min_len, max_len));

    let chunk_size = chunk_size_for(n);
    let n_chunks = n.div_ceil(chunk_size.max(1)).max(1);

    let mut per_chunk: Vec<StringChunked> = (0..n_chunks)
        .into_par_iter()
        .map(|i| {
            let start = i * chunk_size;
            let end = (start + chunk_size).min(n);
            let len = end - start;
            let mut builder = StringChunkedBuilder::new(name.clone(), len);
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed_for_chunk(seed, i));

            if let Some(cats) = &categories {
                if let Some(s) = &sampler {
                    if spec.nullable {
                        let null_p = spec.null_probability;
                        for _ in 0..len {
                            if rng.gen_bool(null_p) {
                                builder.append_null();
                            } else {
                                builder.append_value(&cats[s.sample(&mut rng)]);
                            }
                        }
                    } else {
                        for _ in 0..len {
                            builder.append_value(&cats[s.sample(&mut rng)]);
                        }
                    }
                } else {
                    for _ in 0..len {
                        builder.append_value("");
                    }
                }
            } else {
                let mut scratch: Vec<u8> = vec![0u8; max_len];
                if spec.nullable {
                    let null_p = spec.null_probability;
                    for _ in 0..len {
                        if rng.gen_bool(null_p) {
                            builder.append_null();
                        } else {
                            let str_len = len_dist.as_ref().map_or(min_len, |dist| dist.sample(&mut rng));
                            let mut rand_val = rng.next_u64();
                            let mut bits_left = 64;
                            for j in 0..str_len {
                                if bits_left < 6 {
                                    rand_val = rng.next_u64();
                                    bits_left = 64;
                                }
                                let mut idx = (rand_val & 0x3F) as usize;
                                rand_val >>= 6;
                                bits_left -= 6;
                                if idx >= 62 {
                                    idx = (rng.next_u32() as usize) % 62;
                                }
                                scratch[j] = CHARSET[idx];
                            }
                            // SAFETY: CHARSET contains only ASCII bytes (A-Z, a-z, 0-9).
                            let s = unsafe { std::str::from_utf8_unchecked(&scratch[..str_len]) };
                            builder.append_value(s);
                        }
                    }
                } else {
                    for _ in 0..len {
                        let str_len = len_dist.as_ref().map_or(min_len, |dist| dist.sample(&mut rng));
                        let mut rand_val = rng.next_u64();
                        let mut bits_left = 64;
                        for j in 0..str_len {
                            if bits_left < 6 {
                                rand_val = rng.next_u64();
                                bits_left = 64;
                            }
                            let mut idx = (rand_val & 0x3F) as usize;
                            rand_val >>= 6;
                            bits_left -= 6;
                            if idx >= 62 {
                                idx = (rng.next_u32() as usize) % 62;
                            }
                            scratch[j] = CHARSET[idx];
                        }
                        // SAFETY: CHARSET contains only ASCII bytes (A-Z, a-z, 0-9).
                        let s = unsafe { std::str::from_utf8_unchecked(&scratch[..str_len]) };
                        builder.append_value(s);
                    }
                }
            }
            builder.finish()
        })
        .collect();

    let mut result = per_chunk.remove(0);
    for ca in per_chunk {
        result.append_owned(ca).expect("all chunks share the same dtype");
    }
    result
}

fn generate_series(spec: &ColumnSpec, n: usize, seed: u64) -> PyResult<Series> {
    let dist_kind = DistKind::from_spec(spec)?;
    let series = match spec.kind.to_ascii_lowercase().as_str() {
        "int" | "int64" | "i64" => gen_int64_column(spec, dist_kind, n, seed).into_series(),
        "int32" | "i32" => gen_int32_column(spec, dist_kind, n, seed).into_series(),
        "int16" | "i16" => gen_int16_column(spec, dist_kind, n, seed).into_series(),
        "int8" | "i8" => gen_int8_column(spec, dist_kind, n, seed).into_series(),
        "uint64" | "u64" => gen_uint64_column(spec, dist_kind, n, seed).into_series(),
        "uint32" | "u32" => gen_uint32_column(spec, dist_kind, n, seed).into_series(),
        "uint16" | "u16" => gen_uint16_column(spec, dist_kind, n, seed).into_series(),
        "uint8" | "u8" => gen_uint8_column(spec, dist_kind, n, seed).into_series(),
        "float" | "float64" | "f64" => gen_float64_column(spec, dist_kind, n, seed).into_series(),
        "float32" | "f32" => gen_float32_column(spec, dist_kind, n, seed).into_series(),
        "bool" | "boolean" => {
            let p_true = if let Some(w) = &spec.weights {
                if w.len() == 2 && (w[0] + w[1]) > 0.0 {
                    w[1] / (w[0] + w[1])
                } else {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "Boolean weights for column '{}' must be a 2-element sequence [p_false, p_true]",
                        spec.name
                    )));
                }
            } else if let Some(params) = &spec.distribution_params {
                *params.get("p").unwrap_or(&0.5)
            } else {
                0.5
            };
            gen_bool_column(spec, p_true, n, seed).into_series()
        }
        "string" | "str" | "utf8" | "enum" | "categorical" => {
            let cat_len = spec.categories.as_ref().map(|c| c.len()).unwrap_or(0);
            let sampler = CategorySampler::new(cat_len, spec.weights.as_ref(), &spec.name)?;
            gen_string_column(spec, sampler, n, seed).into_series()
        }
        other => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Unsupported column kind '{other}' for column '{}'",
                spec.name
            )))
        }
    };
    Ok(series)
}

#[pyfunction]
#[pyo3(signature = (columns, n_rows, seed=None))]
fn generate_dataframe(
    py: Python<'_>,
    columns: Vec<ColumnSpecTuple>,
    n_rows: usize,
    seed: Option<u64>,
) -> PyResult<PyDataFrame> {
    let base_seed = seed.unwrap_or_else(|| {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0)
    });

    let specs: Vec<ColumnSpec> = columns.into_iter().map(ColumnSpec::from).collect();

    let df = py.detach(|| -> PyResult<DataFrame> {
        // Columns are independent, so generate them in parallel too; each
        // column's own seed is derived from its position to stay deterministic.
        let series_result: PyResult<Vec<Series>> = specs
            .par_iter()
            .enumerate()
            .map(|(col_idx, spec)| {
                let col_seed = base_seed ^ (col_idx as u64).wrapping_mul(0xD1B54A32D192ED03);
                generate_series(spec, n_rows, col_seed)
            })
            .collect();

        let cols: Vec<Column> = series_result?.into_iter().map(Column::from).collect();
        DataFrame::new(n_rows, cols)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    })?;

    Ok(PyDataFrame(df))
}

#[pymodule]
fn _polspec(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(generate_dataframe, m)?)?;
    Ok(())
}

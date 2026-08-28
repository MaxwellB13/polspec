use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;
use polars::prelude::*;
use polars_core::chunked_array::builder::StringChunkedBuilder;
use rand::distributions::{Distribution, Uniform};
use rand::prelude::*;
use rand::rngs::SmallRng;
use rayon::prelude::*;

const CHARSET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
const DEFAULT_INT_BOUND: i64 = 1_000_000;
const DEFAULT_FLOAT_BOUND: f64 = 1_000_000.0;
const DEFAULT_STR_MIN_LEN: i64 = 5;
const DEFAULT_STR_MAX_LEN: i64 = 15;
const MIN_CHUNK: usize = 4096;

/// One column's generation instructions, sent over from the Python side.
///
/// `kind` is one of: "int", "float", "bool", "string". Polars-specific
/// dtypes (Int8, Enum, Categorical, ...) are resolved to one of these four
/// physical kinds in Python, and cast to the precise dtype after the
/// DataFrame comes back from Rust.
struct ColumnSpec {
    name: String,
    kind: String,
    nullable: bool,
    null_probability: f64,
    min: Option<f64>,
    max: Option<f64>,
    categories: Option<Vec<String>>,
    str_min_len: Option<i64>,
    str_max_len: Option<i64>,
}

type ColumnSpecTuple = (
    String,
    String,
    bool,
    f64,
    Option<f64>,
    Option<f64>,
    Option<Vec<String>>,
    Option<i64>,
    Option<i64>,
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
            str_min_len: t.7,
            str_max_len: t.8,
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

fn gen_bool_column(spec: &ColumnSpec, n: usize, seed: u64) -> BooleanChunked {
    let mut values: Vec<Option<bool>> = vec![None; n];
    let chunk_size = chunk_size_for(n);
    values
        .par_chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(i, chunk)| {
            let mut rng = SmallRng::seed_from_u64(seed_for_chunk(seed, i));
            for v in chunk.iter_mut() {
                if spec.nullable && rng.gen_bool(spec.null_probability) {
                    *v = None;
                } else {
                    *v = Some(rng.gen_bool(0.5));
                }
            }
        });
    BooleanChunked::from_iter_options(PlSmallStr::from(spec.name.as_str()), values.into_iter())
}

fn gen_int_column(spec: &ColumnSpec, n: usize, seed: u64) -> Int64Chunked {
    let lo = spec.min.unwrap_or(-DEFAULT_INT_BOUND as f64) as i64;
    let hi = spec.max.unwrap_or(DEFAULT_INT_BOUND as f64) as i64;
    let (lo, hi) = if lo <= hi { (lo, hi) } else { (hi, lo) };
    let value_dist = Uniform::new_inclusive(lo, hi);
    let mut values: Vec<Option<i64>> = vec![None; n];
    let chunk_size = chunk_size_for(n);
    values
        .par_chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(i, chunk)| {
            let mut rng = SmallRng::seed_from_u64(seed_for_chunk(seed, i));
            for v in chunk.iter_mut() {
                if spec.nullable && rng.gen_bool(spec.null_probability) {
                    *v = None;
                } else {
                    *v = Some(value_dist.sample(&mut rng));
                }
            }
        });
    Int64Chunked::from_iter_options(PlSmallStr::from(spec.name.as_str()), values.into_iter())
}

fn gen_float_column(spec: &ColumnSpec, n: usize, seed: u64) -> Float64Chunked {
    let lo = spec.min.unwrap_or(-DEFAULT_FLOAT_BOUND);
    let hi = spec.max.unwrap_or(DEFAULT_FLOAT_BOUND);
    let (lo, hi) = if lo <= hi { (lo, hi) } else { (hi, lo) };
    let value_dist = Uniform::new_inclusive(lo, hi);
    let mut values: Vec<Option<f64>> = vec![None; n];
    let chunk_size = chunk_size_for(n);
    values
        .par_chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(i, chunk)| {
            let mut rng = SmallRng::seed_from_u64(seed_for_chunk(seed, i));
            for v in chunk.iter_mut() {
                if spec.nullable && rng.gen_bool(spec.null_probability) {
                    *v = None;
                } else {
                    *v = Some(value_dist.sample(&mut rng));
                }
            }
        });
    Float64Chunked::from_iter_options(PlSmallStr::from(spec.name.as_str()), values.into_iter())
}

fn gen_string_column(spec: &ColumnSpec, n: usize, seed: u64) -> StringChunked {
    let categories = spec.categories.clone();
    let min_len = spec.str_min_len.unwrap_or(DEFAULT_STR_MIN_LEN).max(0) as usize;
    let max_len = spec.str_max_len.unwrap_or(DEFAULT_STR_MAX_LEN).max(min_len as i64) as usize;

    // Cached once per column (not per element/chunk) so the hot per-character
    // loop below never pays Uniform's setup cost more than necessary.
    let charset_dist = Uniform::new(0, CHARSET.len());
    let len_dist = (max_len > min_len).then(|| Uniform::new_inclusive(min_len, max_len));
    let category_dist = categories
        .as_ref()
        .filter(|cats| !cats.is_empty())
        .map(|cats| Uniform::new(0, cats.len()));

    let name = PlSmallStr::from(spec.name.as_str());
    let chunk_size = chunk_size_for(n);
    let n_chunks = n.div_ceil(chunk_size.max(1)).max(1);

    // Building straight into a StringChunkedBuilder (Arrow's StringView
    // layout) instead of a Vec<Option<String>> matters: profiling showed
    // heap allocation, not RNG throughput, dominates string generation --
    // one malloc per row for the intermediate Vec<Option<String>>, on top
    // of a second, single-threaded copy into polars' own buffer. The
    // builder writes directly into polars' contiguous buffer (with inline
    // storage for short strings, no allocation at all below ~12 bytes), and
    // each chunk is built in parallel, then stitched together for free via
    // `append_owned` (an O(1) chunk-list merge, not a data copy).
    let mut per_chunk: Vec<StringChunked> = (0..n_chunks)
        .into_par_iter()
        .map(|i| {
            let start = i * chunk_size;
            let end = (start + chunk_size).min(n);
            let len = end - start;
            let mut builder = StringChunkedBuilder::new(name.clone(), len);
            let mut rng = SmallRng::seed_from_u64(seed_for_chunk(seed, i));
            let mut scratch = String::new();
            for _ in 0..len {
                if spec.nullable && rng.gen_bool(spec.null_probability) {
                    builder.append_null();
                    continue;
                }
                if let Some(cats) = &categories {
                    if let Some(dist) = &category_dist {
                        builder.append_value(&cats[dist.sample(&mut rng)]);
                    } else {
                        builder.append_value("");
                    }
                } else {
                    let str_len = len_dist.map_or(min_len, |dist| dist.sample(&mut rng));
                    scratch.clear();
                    scratch.extend((0..str_len).map(|_| CHARSET[charset_dist.sample(&mut rng)] as char));
                    builder.append_value(&scratch);
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
    let series = match spec.kind.as_str() {
        "int" => gen_int_column(spec, n, seed).into_series(),
        "float" => gen_float_column(spec, n, seed).into_series(),
        "bool" => gen_bool_column(spec, n, seed).into_series(),
        "string" => gen_string_column(spec, n, seed).into_series(),
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

    let columns: Vec<Column> = series_result?.into_iter().map(Column::from).collect();
    let df = DataFrame::new(n_rows, columns)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    Ok(PyDataFrame(df))
}

#[pymodule]
fn _polspec(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(generate_dataframe, m)?)?;
    Ok(())
}

//! The `polspec._polspec` extension module: the glue between Python and the
//! sampling code, and nothing else.
//!
//! - `plan.rs` -- `ColumnPlan`, the typed instructions for one column.
//! - `format.rs` -- `Template`, the shape of a string with a format.
//! - `pattern.rs` -- strings from a `pattern`, walked through its DFA.
//! - `dist.rs` -- the distributions and their canonical parameters.
//! - `samplers.rs` -- drawing from them, the same on every platform
//!   (`ziggurat_tables.rs` holds their constants).
//! - `math.rs` -- `exp`, `ln` and `pow`, the same bits on every platform
//!   (`math_tables.rs` holds their tables).
//! - `sample.rs` -- filling a column, in seeded parallel chunks.
//! - `alias.rs` -- weighted categories, drawn by the alias method.
//! - `unique.rs` -- filling a column whose values must all differ.
//! - `permute.rs` -- the keyed permutation behind both, and behind a unique
//!   foreign key's pick of parent rows.
//! - `assemble.rs` -- a `List` column from its lengths and elements, and a
//!   `Decimal` from its physical integers.

mod alias;
mod assemble;
mod chars;
mod dist;
mod format;
mod math;
mod math_tables;
mod pattern;
mod permute;
mod plan;
mod sample;
mod samplers;
mod unique;
mod views;
mod ziggurat_tables;

use polars::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_polars::{PyDataFrame, PySeries};
use rayon::prelude::*;

use plan::ColumnPlan;

/// Fills every column of a frame in parallel.
///
/// Each column's seed is derived from `seed` and the column's seed name --
/// its own name unless the spec gave it another -- so adding or reordering
/// columns never changes the values of the others, and a renamed column can
/// keep producing the data it did. With no seed, the current time is used.
///
/// `row_offset` asks for rows `[row_offset, row_offset + n_rows)` of the frame
/// `seed` describes, so a batch is a window onto one frame whatever its size.
#[pyfunction]
#[pyo3(signature = (columns, n_rows, seed=None, row_offset=0))]
fn generate_dataframe(
    py: Python<'_>,
    columns: Vec<ColumnPlan>,
    n_rows: usize,
    seed: Option<u64>,
    row_offset: usize,
) -> PyResult<PyDataFrame> {
    let base_seed = seed.unwrap_or_else(|| {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0)
    });

    let df = py
        .detach(|| -> Result<DataFrame, String> {
            let series: Result<Vec<Series>, String> = columns
                .par_iter()
                .map(|plan| {
                    let col_seed = sample::seed_for_column(base_seed, plan.seed_key());
                    sample::generate_series(plan, n_rows, col_seed, row_offset)
                })
                .collect();
            let cols: Vec<Column> = series?.into_iter().map(Column::from).collect();
            DataFrame::new(n_rows, cols).map_err(|e| e.to_string())
        })
        .map_err(PyValueError::new_err)?;

    Ok(PyDataFrame(df))
}

/// Positions `start..start + n` of the keyed permutation of `[0, domain)`,
/// as `UInt64` indices -- the rows of a parent a unique foreign key takes, so
/// a window of the child takes the parent rows the whole frame takes there.
#[pyfunction]
fn permuted_indices(
    py: Python<'_>,
    domain: u64,
    seed: u64,
    start: u64,
    n: u64,
) -> PyResult<PySeries> {
    let indices = py
        .detach(|| permute::window(domain, seed, start, n))
        .map_err(PyValueError::new_err)?;
    let series = UInt64Chunked::from_vec(PlSmallStr::from_static("index"), indices);
    Ok(PySeries(series.into_series()))
}

/// A `List` column from each cell's length and every cell's elements, in
/// order; a null length is a null cell. The elements keep their dtype.
#[pyfunction]
fn list_from_lengths(
    py: Python<'_>,
    name: &str,
    lengths: PySeries,
    elements: PySeries,
) -> PyResult<PySeries> {
    let series = py
        .detach(|| assemble::list_from_lengths(name, &lengths.0, &elements.0))
        .map_err(PyValueError::new_err)?;
    Ok(PySeries(series))
}

/// A `Decimal(precision, scale)` column from its physical integers.
#[pyfunction]
fn decimal_from_physical(
    py: Python<'_>,
    raw: PySeries,
    precision: usize,
    scale: usize,
) -> PyResult<PySeries> {
    let series = py
        .detach(|| assemble::decimal_from_physical(&raw.0, precision, scale))
        .map_err(PyValueError::new_err)?;
    Ok(PySeries(series))
}

/// The canonical parameters of one distribution: `(name, default, must_be_positive)`.
#[pyfunction]
fn distribution_params(name: &str) -> PyResult<Vec<(String, f64, bool)>> {
    let params = dist::params_of(name).ok_or_else(|| {
        PyValueError::new_err(format!(
            "Unsupported distribution '{name}'; expected one of {}",
            dist::names().join(", ")
        ))
    })?;
    Ok(params
        .iter()
        .map(|p| (p.name.to_string(), p.default, p.positive))
        .collect())
}

/// Every distribution the engine can sample, canonical names only.
#[pyfunction]
fn distributions() -> Vec<String> {
    dist::names().into_iter().map(String::from).collect()
}

/// How a `pattern` column is generated: `"generate"` from the pattern,
/// `"keep"` the plain text it drew before 1.4 (every value of which already
/// passes the pattern, `plain_lengths` characters long), or `"unreachable"`
/// with the reason the pattern cannot be generated from.
#[pyfunction]
#[pyo3(signature = (pattern, lengths, plain_lengths, unique=false))]
fn pattern_status(
    py: Python<'_>,
    pattern: &str,
    lengths: Option<(usize, usize)>,
    plain_lengths: (usize, usize),
    unique: bool,
) -> (&'static str, Option<String>) {
    py.detach(|| {
        if pattern::plain_text_always_passes(pattern, plain_lengths.0, plain_lengths.1) {
            return ("keep", None);
        }
        let numbered = |sampler: &pattern::PatternSampler| match unique {
            true => sampler.counts().map(|_| ()),
            false => Ok(()),
        };
        match pattern::sampler(pattern, lengths).and_then(|s| numbered(&s)) {
            Ok(()) => ("generate", None),
            Err(reason) => ("unreachable", Some(reason)),
        }
    })
}

/// Every column kind a `ColumnPlan` accepts.
#[pyfunction]
fn kinds() -> Vec<String> {
    plan::Kind::NAMES.iter().map(|s| s.to_string()).collect()
}

/// The rows in one seed chunk of a column: each is filled from its own seed,
/// so a window that starts on a multiple of this draws no row it does not
/// keep.
#[pyfunction]
fn chunk_size() -> usize {
    sample::SEED_CHUNK
}

#[pymodule]
fn _polspec(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ColumnPlan>()?;
    m.add_function(wrap_pyfunction!(chunk_size, m)?)?;
    m.add_function(wrap_pyfunction!(decimal_from_physical, m)?)?;
    m.add_function(wrap_pyfunction!(list_from_lengths, m)?)?;
    m.add_function(wrap_pyfunction!(generate_dataframe, m)?)?;
    m.add_function(wrap_pyfunction!(distribution_params, m)?)?;
    m.add_function(wrap_pyfunction!(distributions, m)?)?;
    m.add_function(wrap_pyfunction!(kinds, m)?)?;
    m.add_function(wrap_pyfunction!(pattern_status, m)?)?;
    m.add_function(wrap_pyfunction!(permuted_indices, m)?)?;
    Ok(())
}

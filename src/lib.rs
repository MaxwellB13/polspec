//! The `polspec._polspec` extension module: the glue between Python and the
//! sampling code, and nothing else.
//!
//! - `plan.rs` -- `ColumnPlan`, the typed instructions for one column.
//! - `dist.rs` -- the distributions and their canonical parameters.
//! - `sample.rs` -- filling a column, in seeded parallel chunks.

mod dist;
mod plan;
mod sample;

use polars::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;
use rayon::prelude::*;

use plan::ColumnPlan;

/// Fills every column of a frame in parallel.
///
/// Each column's seed is derived from `seed` and the column's name, so adding
/// or reordering columns never changes the values of the others. With no
/// seed, the current time is used.
#[pyfunction]
#[pyo3(signature = (columns, n_rows, seed=None))]
fn generate_dataframe(
    py: Python<'_>,
    columns: Vec<ColumnPlan>,
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

    let df = py
        .detach(|| -> Result<DataFrame, String> {
            let series: Result<Vec<Series>, String> = columns
                .par_iter()
                .map(|plan| {
                    let col_seed = sample::seed_for_column(base_seed, &plan.name);
                    sample::generate_series(plan, n_rows, col_seed)
                })
                .collect();
            let cols: Vec<Column> = series?.into_iter().map(Column::from).collect();
            DataFrame::new(n_rows, cols).map_err(|e| e.to_string())
        })
        .map_err(PyValueError::new_err)?;

    Ok(PyDataFrame(df))
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

/// Every column kind a `ColumnPlan` accepts.
#[pyfunction]
fn kinds() -> Vec<String> {
    plan::Kind::NAMES.iter().map(|s| s.to_string()).collect()
}

#[pymodule]
fn _polspec(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ColumnPlan>()?;
    m.add_function(wrap_pyfunction!(generate_dataframe, m)?)?;
    m.add_function(wrap_pyfunction!(distribution_params, m)?)?;
    m.add_function(wrap_pyfunction!(distributions, m)?)?;
    m.add_function(wrap_pyfunction!(kinds, m)?)?;
    Ok(())
}

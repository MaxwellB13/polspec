//! `ColumnPlan`: one column's generation instructions, typed at the boundary.
//!
//! Python builds one of these per column. Everything the engine could object
//! to -- an unknown kind, a weight vector of the wrong length, a distribution
//! parameter out of range -- is rejected here, at construction, with a message
//! naming the column, so the sampling code never has to.

use std::collections::HashMap;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::dist::Distribution;

/// The physical kind of value a column is filled with.
///
/// Python maps every Polars dtype onto one of these: temporal dtypes cross as
/// the integer they store, and any column with a finite domain (`choices`, an
/// `Enum`, a capacity-limited `Categorical`) crosses as `Index`, receiving
/// indices into that domain back.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Float32,
    Float64,
    Bool,
    String,
    Index,
}

impl Kind {
    pub const NAMES: [&'static str; 13] = [
        "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "float32",
        "float64", "bool", "string", "index",
    ];

    pub fn parse(name: &str) -> Option<Kind> {
        Some(match name {
            "int8" => Kind::Int8,
            "int16" => Kind::Int16,
            "int32" => Kind::Int32,
            "int64" => Kind::Int64,
            "uint8" => Kind::UInt8,
            "uint16" => Kind::UInt16,
            "uint32" => Kind::UInt32,
            "uint64" => Kind::UInt64,
            "float32" => Kind::Float32,
            "float64" => Kind::Float64,
            "bool" => Kind::Bool,
            "string" => Kind::String,
            "index" => Kind::Index,
            _ => return None,
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            Kind::Int8 => "int8",
            Kind::Int16 => "int16",
            Kind::Int32 => "int32",
            Kind::Int64 => "int64",
            Kind::UInt8 => "uint8",
            Kind::UInt16 => "uint16",
            Kind::UInt32 => "uint32",
            Kind::UInt64 => "uint64",
            Kind::Float32 => "float32",
            Kind::Float64 => "float64",
            Kind::Bool => "bool",
            Kind::String => "string",
            Kind::Index => "index",
        }
    }

    pub fn is_numeric(self) -> bool {
        !matches!(self, Kind::Bool | Kind::String | Kind::Index)
    }
}

/// One end of a numeric range, kept in the widest type that holds it exactly.
///
/// A Python `int` extracts as `Int` when it fits an `i64`, as `UInt` when it
/// only fits a `u64`, and a Python `float` as `Float`; so `Int64` and `UInt64`
/// bounds reach the sampler without the rounding an `f64` channel would apply.
#[derive(Clone, Copy, Debug, PartialEq, FromPyObject, IntoPyObject)]
pub enum Limit {
    #[pyo3(transparent)]
    Int(i64),
    #[pyo3(transparent)]
    UInt(u64),
    #[pyo3(transparent)]
    Float(f64),
}

impl Limit {
    pub fn as_f64(self) -> f64 {
        match self {
            Limit::Int(v) => v as f64,
            Limit::UInt(v) => v as f64,
            Limit::Float(v) => v,
        }
    }

    /// The value as an integer, truncating a float toward zero.
    pub fn as_i128(self) -> i128 {
        match self {
            Limit::Int(v) => v as i128,
            Limit::UInt(v) => v as i128,
            Limit::Float(v) => v as i128,
        }
    }

    pub fn is_finite(self) -> bool {
        match self {
            Limit::Float(v) => v.is_finite(),
            _ => true,
        }
    }
}

/// One column's generation instructions.
#[pyclass(frozen, from_py_object, module = "polspec._polspec")]
#[derive(Clone, Debug)]
pub struct ColumnPlan {
    pub name: String,
    pub kind: Kind,
    pub nullable: bool,
    pub null_probability: f64,
    pub min: Option<Limit>,
    pub max: Option<Limit>,
    /// Size of the domain an `Index` column samples from.
    pub n_categories: Option<usize>,
    /// Per-category weights for `Index`, or `[p_false, p_true]` for `Bool`.
    pub weights: Option<Vec<f64>>,
    pub str_min_len: usize,
    pub str_max_len: usize,
    pub distribution: Distribution,
    /// Probability of `true` for a `Bool` column.
    pub p_true: f64,
    /// Whether every non-null value must differ from every other.
    pub unique: bool,
}

pub const DEFAULT_STR_MIN_LEN: usize = 5;
pub const DEFAULT_STR_MAX_LEN: usize = 15;

impl ColumnPlan {
    /// Builds and validates a plan. Every error names the column.
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        name: String,
        kind: &str,
        nullable: bool,
        null_probability: f64,
        min: Option<Limit>,
        max: Option<Limit>,
        n_categories: Option<usize>,
        weights: Option<Vec<f64>>,
        str_min_len: Option<usize>,
        str_max_len: Option<usize>,
        distribution: Option<&str>,
        params: Option<&HashMap<String, f64>>,
        unique: bool,
    ) -> Result<Self, String> {
        let kind = Kind::parse(kind).ok_or_else(|| {
            format!(
                "Unsupported column kind '{kind}' for column '{name}'; expected one of {}",
                Kind::NAMES.join(", ")
            )
        })?;
        if !(0.0..=1.0).contains(&null_probability) || null_probability.is_nan() {
            return Err(format!(
                "null_probability for column '{name}' must be within [0, 1], got {null_probability}"
            ));
        }
        for (label, value) in [("min", min), ("max", max)] {
            if let Some(v) = value
                && !v.is_finite()
            {
                return Err(format!(
                    "Bound {label} for column '{name}' must be finite, got {}",
                    v.as_f64()
                ));
            }
        }
        if let Some(w) = &weights {
            if w.iter().any(|x| !x.is_finite() || *x < 0.0) {
                return Err(format!(
                    "Weights for column '{name}' must be finite and non-negative"
                ));
            }
            if w.iter().sum::<f64>() <= 0.0 {
                return Err(format!("Weights for column '{name}' must not all be zero"));
            }
        }
        match kind {
            Kind::Index => {
                let n = n_categories.ok_or_else(|| {
                    format!("Column '{name}' has kind 'index' but no n_categories")
                })?;
                if n == 0 {
                    return Err(format!(
                        "Column '{name}' has kind 'index' with an empty domain"
                    ));
                }
                if let Some(w) = &weights
                    && w.len() != n
                {
                    return Err(format!(
                        "Length of weights ({}) must match number of categories ({n}) for column '{name}'",
                        w.len()
                    ));
                }
            }
            Kind::Bool => {
                if let Some(w) = &weights
                    && w.len() != 2
                {
                    return Err(format!(
                        "Boolean weights for column '{name}' must be a 2-element sequence [p_false, p_true]"
                    ));
                }
            }
            _ => {
                if weights.is_some() {
                    return Err(format!(
                        "Weights for column '{name}' need a finite domain (kind 'index' or 'bool'), not '{}'",
                        kind.name()
                    ));
                }
            }
        }
        let str_min_len = str_min_len.unwrap_or(DEFAULT_STR_MIN_LEN);
        let str_max_len = str_max_len.unwrap_or(DEFAULT_STR_MAX_LEN).max(str_min_len);

        let distribution = if kind.is_numeric() {
            Distribution::new(distribution, params, &name)?
        } else {
            Distribution::Uniform
        };
        let p_true = match (&weights, params) {
            (Some(w), _) if kind == Kind::Bool => w[1] / (w[0] + w[1]),
            (_, Some(p)) if kind == Kind::Bool => *p.get("p").unwrap_or(&0.5),
            _ => 0.5,
        };
        if !(0.0..=1.0).contains(&p_true) {
            return Err(format!(
                "Boolean probability for column '{name}' must be within [0, 1], got {p_true}"
            ));
        }

        Ok(ColumnPlan {
            name,
            kind,
            nullable,
            null_probability: if nullable { null_probability } else { 0.0 },
            min,
            max,
            n_categories,
            weights,
            str_min_len,
            str_max_len,
            distribution,
            p_true,
            unique,
        })
    }
}

#[pymethods]
impl ColumnPlan {
    #[new]
    #[pyo3(signature = (
        name, kind, *, nullable=false, null_probability=0.0, min=None, max=None,
        n_categories=None, weights=None, str_min_len=None, str_max_len=None,
        distribution=None, params=None, unique=false,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        name: String,
        kind: &str,
        nullable: bool,
        null_probability: f64,
        min: Option<Limit>,
        max: Option<Limit>,
        n_categories: Option<usize>,
        weights: Option<Vec<f64>>,
        str_min_len: Option<usize>,
        str_max_len: Option<usize>,
        distribution: Option<&str>,
        params: Option<HashMap<String, f64>>,
        unique: bool,
    ) -> PyResult<Self> {
        ColumnPlan::build(
            name,
            kind,
            nullable,
            null_probability,
            min,
            max,
            n_categories,
            weights,
            str_min_len,
            str_max_len,
            distribution,
            params.as_ref(),
            unique,
        )
        .map_err(PyValueError::new_err)
    }

    #[getter]
    fn name(&self) -> &str {
        &self.name
    }

    #[getter]
    fn unique(&self) -> bool {
        self.unique
    }

    #[getter]
    fn kind(&self) -> &'static str {
        self.kind.name()
    }

    #[getter]
    fn nullable(&self) -> bool {
        self.nullable
    }

    #[getter]
    fn null_probability(&self) -> f64 {
        self.null_probability
    }

    #[getter]
    fn min(&self) -> Option<Limit> {
        self.min
    }

    #[getter]
    fn max(&self) -> Option<Limit> {
        self.max
    }

    #[getter]
    fn n_categories(&self) -> Option<usize> {
        self.n_categories
    }

    #[getter]
    fn weights(&self) -> Option<Vec<f64>> {
        self.weights.clone()
    }

    #[getter]
    fn str_min_len(&self) -> usize {
        self.str_min_len
    }

    #[getter]
    fn str_max_len(&self) -> usize {
        self.str_max_len
    }

    #[getter]
    fn distribution(&self) -> &'static str {
        self.distribution.name()
    }

    #[getter]
    fn p_true(&self) -> f64 {
        self.p_true
    }

    fn __repr__(&self) -> String {
        format!(
            "ColumnPlan(name={:?}, kind={:?}, nullable={}, null_probability={}, min={:?}, max={:?}, n_categories={:?}, distribution={:?})",
            self.name,
            self.kind.name(),
            self.nullable,
            self.null_probability,
            self.min.map(Limit::as_f64),
            self.max.map(Limit::as_f64),
            self.n_categories,
            self.distribution.name(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan(kind: &str) -> Result<ColumnPlan, String> {
        ColumnPlan::build(
            "c".into(),
            kind,
            false,
            0.0,
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
    }

    #[test]
    fn every_kind_name_parses_back_to_itself() {
        for name in Kind::NAMES {
            assert_eq!(Kind::parse(name).unwrap().name(), name);
        }
        assert!(
            Kind::parse("Int64").is_none(),
            "kinds are exact and lowercase"
        );
    }

    #[test]
    fn unknown_kind_names_the_column() {
        let err = plan("varchar").unwrap_err();
        assert!(err.contains("'varchar'") && err.contains("'c'"), "{err}");
    }

    #[test]
    fn index_needs_a_domain_and_matching_weights() {
        assert!(plan("index").unwrap_err().contains("no n_categories"));
        let bad = ColumnPlan::build(
            "c".into(),
            "index",
            false,
            0.0,
            None,
            None,
            Some(3),
            Some(vec![1.0, 1.0]),
            None,
            None,
            None,
            None,
            false,
        );
        assert!(bad.unwrap_err().contains("must match number of categories"));
    }

    #[test]
    fn limits_keep_integer_precision() {
        let big = Limit::UInt(u64::MAX);
        assert_eq!(big.as_i128(), u64::MAX as i128);
        assert_eq!(Limit::Int(i64::MIN).as_i128(), i64::MIN as i128);
        assert_eq!(Limit::Float(2.9).as_i128(), 2);
        assert!(!Limit::Float(f64::INFINITY).is_finite());
    }

    #[test]
    fn bool_probability_comes_from_weights_then_params() {
        let from_weights = ColumnPlan::build(
            "b".into(),
            "bool",
            false,
            0.0,
            None,
            None,
            None,
            Some(vec![1.0, 3.0]),
            None,
            None,
            None,
            None,
            false,
        )
        .unwrap();
        assert!((from_weights.p_true - 0.75).abs() < 1e-12);
        let mut params = HashMap::new();
        params.insert("p".to_string(), 0.2);
        let from_params = ColumnPlan::build(
            "b".into(),
            "bool",
            false,
            0.0,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            Some(&params),
            false,
        )
        .unwrap();
        assert!((from_params.p_true - 0.2).abs() < 1e-12);
        assert!((plan("bool").unwrap().p_true - 0.5).abs() < 1e-12);
    }

    #[test]
    fn null_probability_is_zeroed_for_non_nullable_columns() {
        let p = ColumnPlan::build(
            "c".into(),
            "int64",
            false,
            0.4,
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
        assert_eq!(p.null_probability, 0.0);
    }
}

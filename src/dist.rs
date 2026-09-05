//! The distributions the engine samples, and their parameters.
//!
//! Python is the source of truth for parameter *aliases*: `polspec/distributions.py`
//! canonicalises `mu`/`loc` to `mean` and so on when a column is declared, so
//! by the time a plan reaches this module every key is canonical and read
//! exactly. `PARAMS` is exported to Python so a test can assert the two sides
//! agree on names, defaults and which parameters must be positive.

use std::collections::HashMap;

use rand::Rng;
use rand::distributions::Distribution as _;
use rand_distr::{Beta, Exp, Gamma, LogNormal, Normal, Poisson};

/// One parameter of a distribution: canonical name, default, positivity.
pub struct ParamSpec {
    pub name: &'static str,
    pub default: f64,
    pub positive: bool,
}

const fn param(name: &'static str, default: f64, positive: bool) -> ParamSpec {
    ParamSpec {
        name,
        default,
        positive,
    }
}

/// Every distribution, with its parameters, in the order Python declares them.
pub const PARAMS: [(&str, &[ParamSpec]); 7] = [
    ("uniform", &[]),
    (
        "normal",
        &[param("mean", 0.0, false), param("std", 1.0, true)],
    ),
    (
        "lognormal",
        &[param("mean", 0.0, false), param("std", 1.0, true)],
    ),
    // Parameterised by *either* a scale or a rate; scale wins when both are given.
    (
        "exponential",
        &[param("scale", 1.0, true), param("rate", 1.0, true)],
    ),
    ("poisson", &[param("lambda", 1.0, true)]),
    (
        "gamma",
        &[param("shape", 1.0, true), param("scale", 1.0, true)],
    ),
    (
        "beta",
        &[param("alpha", 1.0, true), param("beta", 1.0, true)],
    ),
];

pub fn params_of(name: &str) -> Option<&'static [ParamSpec]> {
    PARAMS
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, params)| *params)
}

pub fn names() -> Vec<&'static str> {
    PARAMS.iter().map(|(n, _)| *n).collect()
}

#[derive(Clone, Copy, Debug)]
pub enum Distribution {
    Uniform,
    Normal(Normal<f64>),
    LogNormal(LogNormal<f64>),
    Exp(Exp<f64>),
    Poisson(Poisson<f64>),
    Gamma(Gamma<f64>),
    Beta(Beta<f64>),
}

impl Distribution {
    /// A distribution from its canonical name and canonical parameter keys.
    ///
    /// Unknown keys are ignored (a caller may share one parameter dict across
    /// columns); a parameter that must be positive and is not is an error.
    pub fn new(
        name: Option<&str>,
        params: Option<&HashMap<String, f64>>,
        column: &str,
    ) -> Result<Self, String> {
        let name = match name {
            None => return Ok(Distribution::Uniform),
            Some(n) => n,
        };
        let specs = params_of(name).ok_or_else(|| {
            format!(
                "Unsupported distribution '{name}' for column '{column}'; expected one of {}",
                names().join(", ")
            )
        })?;
        let get = |key: &str| -> Result<f64, String> {
            let spec = specs
                .iter()
                .find(|p| p.name == key)
                .expect("parameter names below match PARAMS");
            let value = params
                .and_then(|m| m.get(key).copied())
                .unwrap_or(spec.default);
            if !value.is_finite() {
                return Err(format!(
                    "{name} distribution {key} for column '{column}' must be finite, got {value}"
                ));
            }
            if spec.positive && value <= 0.0 {
                return Err(format!(
                    "{name} distribution {key} for column '{column}' must be positive, got {value}"
                ));
            }
            Ok(value)
        };
        let invalid = |e: &dyn std::fmt::Display| {
            format!("Invalid {name} distribution parameters for column '{column}': {e}")
        };
        Ok(match name {
            "uniform" => Distribution::Uniform,
            "normal" => Normal::new(get("mean")?, get("std")?)
                .map(Distribution::Normal)
                .map_err(|e| invalid(&e))?,
            "lognormal" => LogNormal::new(get("mean")?, get("std")?)
                .map(Distribution::LogNormal)
                .map_err(|e| invalid(&e))?,
            "exponential" => {
                let has_scale = params.is_some_and(|m| m.contains_key("scale"));
                let rate = if has_scale {
                    1.0 / get("scale")?
                } else {
                    get("rate")?
                };
                Exp::new(rate)
                    .map(Distribution::Exp)
                    .map_err(|e| invalid(&e))?
            }
            "poisson" => Poisson::new(get("lambda")?)
                .map(Distribution::Poisson)
                .map_err(|e| invalid(&e))?,
            "gamma" => Gamma::new(get("shape")?, get("scale")?)
                .map(Distribution::Gamma)
                .map_err(|e| invalid(&e))?,
            "beta" => Beta::new(get("alpha")?, get("beta")?)
                .map(Distribution::Beta)
                .map_err(|e| invalid(&e))?,
            _ => unreachable!("params_of accepted the name"),
        })
    }

    pub fn name(&self) -> &'static str {
        match self {
            Distribution::Uniform => "uniform",
            Distribution::Normal(_) => "normal",
            Distribution::LogNormal(_) => "lognormal",
            Distribution::Exp(_) => "exponential",
            Distribution::Poisson(_) => "poisson",
            Distribution::Gamma(_) => "gamma",
            Distribution::Beta(_) => "beta",
        }
    }

    /// One draw. `Uniform` is sampled by the caller over its own typed range.
    #[inline(always)]
    pub fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        match self {
            Distribution::Uniform => unreachable!("uniform is sampled over the typed range"),
            Distribution::Normal(d) => d.sample(rng),
            Distribution::LogNormal(d) => d.sample(rng),
            Distribution::Exp(d) => d.sample(rng),
            Distribution::Poisson(d) => d.sample(rng),
            Distribution::Gamma(d) => d.sample(rng),
            Distribution::Beta(d) => d.sample(rng),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_xoshiro::rand_core::SeedableRng;

    fn params(pairs: &[(&str, f64)]) -> HashMap<String, f64> {
        pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }

    #[test]
    fn no_name_is_uniform() {
        assert!(matches!(
            Distribution::new(None, None, "c").unwrap(),
            Distribution::Uniform
        ));
    }

    #[test]
    fn defaults_apply_when_parameters_are_omitted() {
        for (name, _) in PARAMS {
            let d = Distribution::new(Some(name), None, "c").unwrap();
            assert_eq!(d.name(), name);
        }
    }

    #[test]
    fn aliases_are_not_read_here() {
        // Python canonicalises `sigma` to `std` before a plan is built; the
        // engine reads exact keys only, so `sigma` falls back to the default.
        let d = Distribution::new(Some("normal"), Some(&params(&[("sigma", 5.0)])), "c").unwrap();
        assert_eq!(d.name(), "normal");
        let err =
            Distribution::new(Some("normal"), Some(&params(&[("std", -1.0)])), "c").unwrap_err();
        assert!(
            err.contains("must be positive") && err.contains("'c'"),
            "{err}"
        );
    }

    #[test]
    fn exponential_scale_wins_over_rate() {
        let d = Distribution::new(
            Some("exponential"),
            Some(&params(&[("scale", 4.0), ("rate", 100.0)])),
            "c",
        )
        .unwrap();
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(1);
        // A scale of 4 means a mean of 4; a rate of 100 a mean of 0.01. One
        // draw cannot prove the mean, but a huge draw can only come from scale.
        let draws: f64 = (0..1000).map(|_| d.sample(&mut rng)).sum::<f64>() / 1000.0;
        assert!(draws > 0.1, "rate must not have been used: mean {draws}");
    }

    #[test]
    fn unknown_distribution_names_the_column_and_the_options() {
        let err = Distribution::new(Some("cauchy"), None, "amount").unwrap_err();
        assert!(err.contains("'cauchy'") && err.contains("'amount'") && err.contains("beta"));
    }
}

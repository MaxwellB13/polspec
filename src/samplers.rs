// Copyright 2018 Developers of the Rand project.
// Copyright 2013 The Rust Project Developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! The non-uniform samplers, with the same result on every platform.
//!
//! Ported from `rand_distr` 0.6.0 (see LICENSE-THIRD-PARTY): the same
//! algorithms, the same constants, and the same random numbers consumed in
//! the same order. One thing differs. `rand_distr` does its float maths
//! through `num-traits`, which Polars builds with `std`, so `exp`, `ln` and
//! `powf` were each platform's libm -- glibc, the MSVC CRT, Apple's -- and
//! those disagree in the last bit on a few inputs. A last-bit difference in
//! an accept/reject test changes how many random numbers a draw consumes,
//! and with it every later value in the chunk. Here those three functions
//! come from `Math`, which in the engine is `math.rs` -- Arm's
//! optimized-routines, the algorithms glibc uses, ported to plain IEEE
//! arithmetic: bit-identical everywhere, and as fast as a platform's own. Everything else a sampler computes (`sqrt`,
//! `floor`, `abs`, `+ - * /`) is exactly rounded by IEEE 754, so it is the
//! same everywhere already.
//!
//! The samplers are generic over `Math` so a test can run them on `std`'s
//! functions instead, and check them draw for draw against `rand_distr`
//! itself: the port is exact, and only the maths it calls changed.

use std::f64::consts::PI;

use rand::distr::Open01;
use rand::{Rng, RngExt};

use crate::ziggurat_tables::{self as zig, ZigTable};

/// The float functions a sampler calls that are not exactly rounded.
pub trait Math {
    fn exp(x: f64) -> f64;
    fn ln(x: f64) -> f64;
    fn pow(x: f64, y: f64) -> f64;
}

/// `math.rs`: Arm's algorithms, the same bits on every platform.
pub enum Portable {}

impl Math for Portable {
    #[inline(always)]
    fn exp(x: f64) -> f64 {
        crate::math::exp(x)
    }
    #[inline(always)]
    fn ln(x: f64) -> f64 {
        crate::math::ln(x)
    }
    #[inline(always)]
    fn pow(x: f64, y: f64) -> f64 {
        crate::math::pow(x, y)
    }
}

/// `bits`, the low 52 of which are a fraction, as a float in
/// `[2^exponent, 2^(exponent+1))` -- `rand`'s `into_float_with_exponent`.
#[inline(always)]
fn float_with_exponent(bits: u64, exponent: i32) -> f64 {
    f64::from_bits(bits | (((1023 + exponent) as u64) << 52))
}

/// The ZIGNOR variant of the ziggurat method (Doornik 2005), as
/// `rand_distr::utils::ziggurat`.
#[inline(always)]
fn ziggurat<R, P, Z>(
    rng: &mut R,
    symmetric: bool,
    x_tab: ZigTable,
    f_tab: ZigTable,
    mut pdf: P,
    mut zero_case: Z,
) -> f64
where
    R: Rng + ?Sized,
    P: FnMut(f64) -> f64,
    Z: FnMut(&mut R, f64) -> f64,
{
    loop {
        let bits = rng.next_u64();
        let i = bits as usize & 0xff;
        let u = if symmetric {
            float_with_exponent(bits >> 12, 1) - 3.0
        } else {
            float_with_exponent(bits >> 12, 0) - (1.0 - f64::EPSILON / 2.0)
        };
        let x = u * x_tab[i];
        let test_x = if symmetric { x.abs() } else { x };
        if test_x < x_tab[i + 1] {
            return x;
        }
        if i == 0 {
            return zero_case(rng, u);
        }
        if f_tab[i + 1] + (f_tab[i] - f_tab[i + 1]) * rng.random::<f64>() < pdf(x) {
            return x;
        }
    }
}

/// A standard normal draw.
pub fn standard_normal<M: Math, R: Rng + ?Sized>(rng: &mut R) -> f64 {
    let pdf = |x: f64| M::exp(-x * x / 2.0);
    let zero_case = |rng: &mut R, u: f64| {
        // The tail, by hand. The initial values make the loop run once.
        let mut x = 1.0f64;
        let mut y = 0.0f64;
        while -2.0 * y < x * x {
            let x_: f64 = rng.sample(Open01);
            let y_: f64 = rng.sample(Open01);
            x = M::ln(x_) / zig::ZIG_NORM_R;
            y = M::ln(y_);
        }
        if u < 0.0 {
            x - zig::ZIG_NORM_R
        } else {
            zig::ZIG_NORM_R - x
        }
    };
    ziggurat(
        rng,
        true,
        &zig::ZIG_NORM_X,
        &zig::ZIG_NORM_F,
        pdf,
        zero_case,
    )
}

/// A standard exponential draw (rate 1).
#[inline]
pub fn standard_exp<M: Math, R: Rng + ?Sized>(rng: &mut R) -> f64 {
    let pdf = |x: f64| M::exp(-x);
    let zero_case = |rng: &mut R, _u: f64| zig::ZIG_EXP_R - M::ln(rng.random::<f64>());
    ziggurat(rng, false, &zig::ZIG_EXP_X, &zig::ZIG_EXP_F, pdf, zero_case)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Normal {
    mean: f64,
    std: f64,
}

impl Normal {
    pub fn new(mean: f64, std: f64) -> Self {
        Normal { mean, std }
    }

    pub fn sample<M: Math, R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        self.mean + self.std * standard_normal::<M, R>(rng)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LogNormal(Normal);

impl LogNormal {
    pub fn new(mean: f64, std: f64) -> Self {
        LogNormal(Normal::new(mean, std))
    }

    #[inline]
    pub fn sample<M: Math, R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        M::exp(self.0.sample::<M, R>(rng))
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Exp {
    lambda_inverse: f64,
}

impl Exp {
    pub fn new(rate: f64) -> Self {
        Exp {
            lambda_inverse: 1.0 / rate,
        }
    }

    pub fn sample<M: Math, R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        standard_exp::<M, R>(rng) * self.lambda_inverse
    }
}

/// Marsaglia and Tsang (2000), for a shape of at least 1.
#[derive(Clone, Copy, Debug, PartialEq)]
struct GammaLargeShape {
    scale: f64,
    c: f64,
    d: f64,
}

impl GammaLargeShape {
    fn new(shape: f64, scale: f64) -> Self {
        let d = shape - 1. / 3.;
        GammaLargeShape {
            scale,
            c: 1.0 / (9. * d).sqrt(),
            d,
        }
    }

    fn sample_unscaled<M: Math, R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        loop {
            let x = standard_normal::<M, R>(rng);
            let v_cbrt = 1.0 + self.c * x;
            if v_cbrt <= 0.0 {
                continue;
            }
            let v = v_cbrt * v_cbrt * v_cbrt;
            let u: f64 = rng.sample(Open01);
            let x_sqr = x * x;
            if u < 1.0 - 0.0331 * x_sqr * x_sqr
                || M::ln(u) < 0.5 * x_sqr + self.d * (1.0 - v + M::ln(v))
            {
                return v;
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum GammaRepr {
    Large(GammaLargeShape),
    One(Exp),
    /// A shape below 1: a draw at shape + 1, scaled by `u^(1/shape)`.
    Small {
        inv_shape: f64,
        large: GammaLargeShape,
    },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gamma(GammaRepr);

impl Gamma {
    /// `shape` and `scale` are positive (the caller checks).
    pub fn new(shape: f64, scale: f64) -> Self {
        Gamma(if shape == f64::INFINITY || scale == f64::INFINITY {
            GammaRepr::One(Exp::new(0.0))
        } else if shape == 1.0 {
            GammaRepr::One(Exp::new(1.0 / scale))
        } else if shape < 1.0 {
            GammaRepr::Small {
                inv_shape: 1.0 / shape,
                large: GammaLargeShape::new(shape + 1.0, scale),
            }
        } else {
            GammaRepr::Large(GammaLargeShape::new(shape, scale))
        })
    }

    pub fn sample<M: Math, R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        match &self.0 {
            GammaRepr::Large(g) => g.sample_unscaled::<M, R>(rng) * (g.d * g.scale),
            GammaRepr::One(e) => e.sample::<M, R>(rng),
            GammaRepr::Small { inv_shape, large } => {
                let u: f64 = rng.sample(Open01);
                let a = large.sample_unscaled::<M, R>(rng);
                let b = M::pow(u, *inv_shape);
                // Scale last, so an overflow cannot make inf * 0.0 = NaN.
                (a * b * large.d) * large.scale
            }
        }
    }
}

/// Cheng (1978): algorithm BB when both parameters exceed 1, BC otherwise.
#[derive(Clone, Copy, Debug, PartialEq)]
enum BetaAlgorithm {
    BB {
        alpha: f64,
        beta: f64,
        gamma: f64,
    },
    BC {
        alpha: f64,
        beta: f64,
        kappa1: f64,
        kappa2: f64,
    },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Beta {
    a: f64,
    b: f64,
    switched_params: bool,
    algorithm: BetaAlgorithm,
    /// `ln 4` and `ln 5`, which the reference computes on every iteration
    /// (`std`'s `ln` folds to a constant; `math::ln` does not): computed once,
    /// with the same function, so every draw is unchanged.
    ln4: f64,
    ln5: f64,
}

impl Beta {
    /// `alpha` and `beta` are positive (the caller checks).
    pub fn new<M: Math>(alpha: f64, beta: f64) -> Self {
        let (ln4, ln5) = (M::ln(4.), M::ln(5.));
        // The reference's notation: `a` the smaller parameter, `b` the larger.
        let (a0, b0) = (alpha, beta);
        let (a, b, switched_params) = if a0 < b0 {
            (a0, b0, false)
        } else {
            (b0, a0, true)
        };
        if a > 1.0 {
            let alpha = a + b;
            let beta = ((alpha - 2.) / (2. * a * b - alpha)).sqrt();
            let gamma = a + 1.0 / beta;
            Beta {
                a,
                b,
                switched_params,
                algorithm: BetaAlgorithm::BB { alpha, beta, gamma },
                ln4,
                ln5,
            }
        } else {
            // Here `a` is the larger parameter.
            let (a, b, switched_params) = (b, a, !switched_params);
            let alpha = a + b;
            let beta = 1.0 / b;
            let delta = 1.0 + a - b;
            let kappa1 = delta * (1. / 18. / 4. + 3. / 18. / 4. * b) / (a * beta - 14. / 18.);
            let kappa2 = 0.25 + (0.5 + 0.25 / delta) * b;
            Beta {
                a,
                b,
                switched_params,
                algorithm: BetaAlgorithm::BC {
                    alpha,
                    beta,
                    kappa1,
                    kappa2,
                },
                ln4,
                ln5,
            }
        }
    }

    // `!(a < b)` is the reference's test, and not `a >= b`: a NaN must
    // accept, as it does there.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    pub fn sample<M: Math, R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        let mut w;
        match self.algorithm {
            BetaAlgorithm::BB { alpha, beta, gamma } => loop {
                let u1: f64 = rng.sample(Open01);
                let u2: f64 = rng.sample(Open01);
                let v = beta * M::ln(u1 / (1.0 - u1));
                w = self.a * M::exp(v);
                let z = u1 * u1 * u2;
                let r = gamma * v - self.ln4;
                let s = self.a + r - w;
                if s + 1.0 + self.ln5 >= 5. * z {
                    break;
                }
                let t = M::ln(z);
                if s >= t {
                    break;
                }
                if !(r + alpha * M::ln(alpha / (self.b + w)) < t) {
                    break;
                }
            },
            BetaAlgorithm::BC {
                alpha,
                beta,
                kappa1,
                kappa2,
            } => loop {
                let z;
                let u1: f64 = rng.sample(Open01);
                let u2: f64 = rng.sample(Open01);
                if u1 < 0.5 {
                    let y = u1 * u2;
                    z = u1 * y;
                    if 0.25 * u2 + z - y >= kappa1 {
                        continue;
                    }
                } else {
                    z = u1 * u1 * u2;
                    if z <= 0.25 {
                        let v = beta * M::ln(u1 / (1.0 - u1));
                        w = self.a * M::exp(v);
                        break;
                    }
                    if z >= kappa2 {
                        continue;
                    }
                }
                let v = beta * M::ln(u1 / (1.0 - u1));
                w = self.a * M::exp(v);
                if !(alpha * (M::ln(alpha / (self.b + w)) + v) - self.ln4 < M::ln(z)) {
                    break;
                }
            },
        };
        if !self.switched_params {
            if w == f64::INFINITY {
                // For a large `w`, with `b` finite.
                return 1.0;
            }
            w / (self.b + w)
        } else {
            self.b / (self.b + w)
        }
    }
}

/// Ahrens and Dieter's PTRS-style rejection method (1982), for a large λ.
#[derive(Clone, Copy, Debug, PartialEq)]
struct PoissonRejection {
    lambda: f64,
    s: f64,
    d: f64,
    l: f64,
    c: f64,
    c0: f64,
    c1: f64,
    c2: f64,
    c3: f64,
    omega: f64,
}

impl PoissonRejection {
    fn new(lambda: f64) -> Self {
        let b1 = 1.0 / 24.0 / lambda;
        let b2 = 0.3 * b1 * b1;
        let c3 = 1.0 / 7.0 * b1 * b2;
        let c2 = b2 - 15. * c3;
        let c1 = b1 - 6. * b2 + 45. * c3;
        let c0 = 1.0 - b1 + 3. * b2 - 15. * c3;
        PoissonRejection {
            lambda,
            s: lambda.sqrt(),
            d: 6.0 * lambda.powi(2),
            l: (lambda - 1.1484).floor(),
            c: 0.1069 / lambda,
            c0,
            c1,
            c2,
            c3,
            omega: 1.0 / (2. * PI).sqrt() / lambda.sqrt(),
        }
    }

    /// Step F: the log-probability terms at `k`.
    fn f<M: Math>(&self, k: f64) -> (f64, f64, f64, f64) {
        // Factorials of 0..10, and the coefficients of Table 1.
        const FACT: [f64; 10] = [
            1.0, 1.0, 2.0, 6.0, 24.0, 120.0, 720.0, 5040.0, 40320.0, 362880.0,
        ];
        const A: [f64; 10] = [
            -0.5000000002,
            0.3333333343,
            -0.2499998565,
            0.1999997049,
            -0.1666848753,
            0.1428833286,
            -0.1241963125,
            0.1101687109,
            -0.1142650302,
            0.1055093006,
        ];
        let (px, py) = if k < 10.0 {
            let px = -self.lambda;
            let py = M::pow(self.lambda, k) / FACT[k as usize];
            (px, py)
        } else {
            let delta = (12.0 * k).recip();
            let delta = delta - 4.8 * delta.powi(3);
            let v = (self.lambda - k) / k;
            let px = if v.abs() <= 0.25 {
                k * v.powi(2) * A.iter().rev().fold(0.0, |acc, &a| acc * v + a) - delta
            } else {
                k * M::ln(1.0 + v) - (self.lambda - k) - delta
            };
            let py = 1.0 / (2.0 * PI).sqrt() / k.sqrt();
            (px, py)
        };
        let x = (k - self.lambda + 0.5) / self.s;
        let fx = -0.5 * x * x;
        let fy = self.omega * (((self.c3 * x * x + self.c2) * x * x + self.c1) * x * x + self.c0);
        (px, py, fx, fy)
    }

    fn sample<M: Math, R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        // Step N
        let g = Normal::new(self.lambda, self.s).sample::<M, R>(rng);
        if g >= 0.0 {
            let k1 = g.floor();
            // Step I
            if k1 >= self.l {
                return k1;
            }
            // Step S
            let u: f64 = rng.random();
            if self.d * u >= (self.lambda - k1).powi(3) {
                return k1;
            }
            let (px, py, fx, fy) = self.f::<M>(k1);
            if fy * (1.0 - u) <= py * M::exp(px - fx) {
                return k1;
            }
        }
        loop {
            // Step E
            let e = standard_exp::<M, R>(rng);
            let u: f64 = rng.random::<f64>() * 2.0 - 1.0;
            let t = 1.8 + e * u.signum();
            if t > -0.6744 {
                let k2 = (self.lambda + self.s * t).floor();
                let (px, py, fx, fy) = self.f::<M>(k2);
                // Step H
                if self.c * u.abs() <= py * M::exp(px + e) - fy * M::exp(fx + e) {
                    return k2;
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum PoissonMethod {
    /// Knuth's product of uniforms, for λ below 12.
    Knuth {
        exp_lambda: f64,
    },
    Rejection(PoissonRejection),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Poisson(PoissonMethod);

impl Poisson {
    /// The largest λ the rejection method handles.
    pub const MAX_LAMBDA: f64 = 1.844e19;

    /// `lambda` is positive and finite (the caller checks); `None` above
    /// `MAX_LAMBDA`.
    pub fn new<M: Math>(lambda: f64) -> Option<Self> {
        if lambda < 12.0 {
            Some(Poisson(PoissonMethod::Knuth {
                exp_lambda: M::exp(-lambda),
            }))
        } else if lambda > Self::MAX_LAMBDA {
            None
        } else {
            Some(Poisson(PoissonMethod::Rejection(PoissonRejection::new(
                lambda,
            ))))
        }
    }

    #[inline]
    pub fn sample<M: Math, R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        match &self.0 {
            PoissonMethod::Knuth { exp_lambda } => {
                let mut result = 1.0;
                let mut p = rng.random::<f64>();
                while p > *exp_lambda {
                    p *= rng.random::<f64>();
                    result += 1.0;
                }
                result - 1.0
            }
            PoissonMethod::Rejection(r) => r.sample::<M, R>(rng),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::distr::Distribution as _;
    use rand_xoshiro::Xoshiro256PlusPlus;

    /// `std`'s functions: what `rand_distr` calls, here, so a draw can be
    /// compared with `rand_distr`'s bit for bit.
    enum Std {}

    impl Math for Std {
        fn exp(x: f64) -> f64 {
            x.exp()
        }
        fn ln(x: f64) -> f64 {
            x.ln()
        }
        fn pow(x: f64, y: f64) -> f64 {
            x.powf(y)
        }
    }

    const DRAWS: usize = 200_000;

    type Draw = Box<dyn Fn(&mut Xoshiro256PlusPlus) -> f64>;

    fn rng(seed: u64) -> Xoshiro256PlusPlus {
        Xoshiro256PlusPlus::seed_from_u64(seed)
    }

    /// Ours on `std`, against `rand_distr`'s, from the same seed: every draw
    /// identical, and the generators left in the same state.
    fn agrees(
        name: &str,
        ours: impl Fn(&mut Xoshiro256PlusPlus) -> f64,
        theirs: impl Fn(&mut Xoshiro256PlusPlus) -> f64,
    ) {
        let (mut a, mut b) = (rng(7), rng(7));
        for i in 0..DRAWS {
            let (x, y) = (ours(&mut a), theirs(&mut b));
            assert!(
                x.to_bits() == y.to_bits(),
                "{name}: draw {i} differs: {x:e} vs {y:e}"
            );
        }
        assert_eq!(a.next_u64(), b.next_u64(), "{name}: the streams diverged");
    }

    #[test]
    fn the_port_is_rand_distrs_algorithms_draw_for_draw() {
        for (mean, std) in [(0.0, 1.0), (4.0, 1.0), (-3.5, 0.01), (1e6, 250.0)] {
            let d = Normal::new(mean, std);
            let r = rand_distr::Normal::new(mean, std).unwrap();
            agrees("normal", |g| d.sample::<Std, _>(g), |g| r.sample(g));
            let d = LogNormal::new(mean.min(5.0), std.min(2.0));
            let r = rand_distr::LogNormal::new(mean.min(5.0), std.min(2.0)).unwrap();
            agrees("lognormal", |g| d.sample::<Std, _>(g), |g| r.sample(g));
        }
        for rate in [1.0, 0.25, 40.0] {
            let d = Exp::new(rate);
            let r = rand_distr::Exp::new(rate).unwrap();
            agrees("exponential", |g| d.sample::<Std, _>(g), |g| r.sample(g));
        }
        // Below 1, exactly 1, and above: the three representations.
        for (shape, scale) in [(0.3, 2.0), (1.0, 1.5), (2.5, 1.0), (40.0, 0.1)] {
            let d = Gamma::new(shape, scale);
            let r = rand_distr::Gamma::new(shape, scale).unwrap();
            agrees("gamma", |g| d.sample::<Std, _>(g), |g| r.sample(g));
        }
        // BB (both above 1) and BC, each way round.
        for (alpha, beta) in [(2.0, 5.0), (5.0, 2.0), (0.5, 0.5), (0.4, 3.0), (3.0, 0.4)] {
            let d = Beta::new::<Std>(alpha, beta);
            let r = rand_distr::Beta::new(alpha, beta).unwrap();
            agrees("beta", |g| d.sample::<Std, _>(g), |g| r.sample(g));
        }
        // Knuth below 12, rejection from 12.
        for lambda in [0.5, 3.0, 11.9, 12.0, 50.0, 1e4] {
            let d = Poisson::new::<Std>(lambda).unwrap();
            let r = rand_distr::Poisson::new(lambda).unwrap();
            agrees("poisson", |g| d.sample::<Std, _>(g), |g| r.sample(g));
        }
    }

    /// Every case the engine draws, by name, on `M`.
    fn cases<M: Math>() -> Vec<(&'static str, Draw)> {
        vec![
            (
                "normal",
                Box::new(|g| Normal::new(0.0, 1.0).sample::<M, _>(g)),
            ),
            (
                "lognormal",
                Box::new(|g| LogNormal::new(0.0, 1.0).sample::<M, _>(g)),
            ),
            ("exponential", Box::new(|g| Exp::new(1.0).sample::<M, _>(g))),
            (
                "gamma, shape < 1",
                Box::new(|g| Gamma::new(0.3, 2.0).sample::<M, _>(g)),
            ),
            (
                "gamma",
                Box::new(|g| Gamma::new(2.5, 1.0).sample::<M, _>(g)),
            ),
            (
                "beta, BB",
                Box::new(|g| Beta::new::<M>(2.0, 5.0).sample::<M, _>(g)),
            ),
            (
                "beta, BC",
                Box::new(|g| Beta::new::<M>(0.5, 0.5).sample::<M, _>(g)),
            ),
            (
                "poisson, Knuth",
                Box::new(|g| Poisson::new::<M>(3.0).unwrap().sample::<M, _>(g)),
            ),
            (
                "poisson, rejection",
                Box::new(|g| Poisson::new::<M>(50.0).unwrap().sample::<M, _>(g)),
            ),
        ]
    }

    /// A digest of 100,000 draws from one seed: every bit of every draw.
    fn digest(draw: &dyn Fn(&mut Xoshiro256PlusPlus) -> f64) -> u64 {
        let mut g = rng(42);
        (0..100_000).fold(0xcbf2_9ce4_8422_2325, |h: u64, _| {
            (h ^ draw(&mut g).to_bits()).wrapping_mul(0x0100_0000_01b3)
        })
    }

    /// The engine's draws, pinned: a digest of 100,000 per case, recorded on
    /// Windows and checked by CI on Linux, so the samplers draw the same
    /// values on both. A change to a sampler -- or to `math.rs` -- is a red test
    /// here rather than a quiet change of seeded output.
    #[test]
    fn the_engines_draws_are_pinned() {
        let pinned = [
            ("normal", 0x7ea1_b296_6218_770c),
            ("lognormal", 0x58f7_65bb_0543_a0d1),
            ("exponential", 0x33a4_2765_43c1_7358),
            ("gamma, shape < 1", 0x0b8d_d184_e503_60b8),
            ("gamma", 0xc364_0d86_b7b8_bb6b),
            ("beta, BB", 0x4c71_6bd9_a3d6_659d),
            ("beta, BC", 0x975e_281f_bb23_1f8a),
            ("poisson, Knuth", 0x756f_4018_df3e_37a5),
            ("poisson, rejection", 0x12a6_c018_df3e_37a5),
        ];
        let drawn = cases::<Portable>();
        assert_eq!(drawn.len(), pinned.len());
        let mut changed = Vec::new();
        for ((name, draw), (pinned_name, pin)) in drawn.iter().zip(pinned) {
            assert_eq!(*name, pinned_name);
            let now = digest(draw.as_ref());
            if now != pin {
                changed.push(format!("(\"{name}\", 0x{now:016x})"));
            }
        }
        // On a deliberate change, the message is the new table.
        assert!(
            changed.is_empty(),
            "the draws changed:
{}",
            changed.join(
                ",
"
            )
        );
    }

    /// Mean and variance over 10^6 draws, within a few standard errors:
    /// the samplers draw from the distributions they name.
    #[test]
    fn the_moments_are_the_distributions() {
        let moments = |draw: &dyn Fn(&mut Xoshiro256PlusPlus) -> f64| {
            let mut g = rng(3);
            let n = 1_000_000.0;
            let (mut sum, mut sum_sq) = (0.0, 0.0);
            for _ in 0..1_000_000 {
                let x = draw(&mut g);
                sum += x;
                sum_sq += x * x;
            }
            let mean = sum / n;
            (mean, sum_sq / n - mean * mean)
        };
        // (name, draw, mean, variance)
        let expected: [(&str, Draw, f64, f64); 7] = [
            (
                "normal",
                Box::new(|g| Normal::new(2.0, 3.0).sample::<Portable, _>(g)),
                2.0,
                9.0,
            ),
            (
                "lognormal",
                Box::new(|g| LogNormal::new(0.0, 0.5).sample::<Portable, _>(g)),
                (0.125f64).exp(),
                ((0.25f64).exp() - 1.0) * (0.25f64).exp(),
            ),
            (
                "exponential",
                Box::new(|g| Exp::new(4.0).sample::<Portable, _>(g)),
                0.25,
                0.0625,
            ),
            (
                "gamma, shape < 1",
                Box::new(|g| Gamma::new(0.3, 2.0).sample::<Portable, _>(g)),
                0.6,
                1.2,
            ),
            (
                "gamma",
                Box::new(|g| Gamma::new(2.5, 1.5).sample::<Portable, _>(g)),
                3.75,
                5.625,
            ),
            (
                "beta",
                Box::new(|g| Beta::new::<Portable>(2.0, 5.0).sample::<Portable, _>(g)),
                2.0 / 7.0,
                10.0 / (49.0 * 8.0),
            ),
            (
                "poisson",
                Box::new(|g| {
                    Poisson::new::<Portable>(50.0)
                        .unwrap()
                        .sample::<Portable, _>(g)
                }),
                50.0,
                50.0,
            ),
        ];
        for (name, draw, mean, var) in &expected {
            let (m, v) = moments(draw.as_ref());
            assert!(
                (m - mean).abs() < 0.01 * mean.abs().max(1.0),
                "{name}: mean {m}, want {mean}"
            );
            assert!(
                (v - var).abs() < 0.03 * var,
                "{name}: variance {v}, want {var}"
            );
        }
    }
}

// Copyright (c) 2018-2025, Arm Limited.
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

//! `exp`, `ln` and `pow`, fast and the same on every platform.
//!
//! Ported from Arm optimized-routines (math/exp.c, log.c, pow.c; see
//! LICENSE-THIRD-PARTY) -- the algorithms glibc uses -- with their tables in
//! `math_tables.rs`. They are table-driven and need only `+ - * /`, bit
//! manipulation and comparisons, all exactly rounded by IEEE 754, so they
//! return the same bits on any platform. Two choices keep it that way:
//!
//! - the reference's non-FMA path everywhere: a fused multiply-add rounds
//!   once where `a * b + c` rounds twice, so using one only where the
//!   hardware has it would change the bits on that hardware (Rust never
//!   contracts `a * b + c` into one on its own);
//! - rounding to an integer by adding a large constant (`SHIFT`), which is
//!   plain arithmetic, rather than the instruction the reference uses on
//!   AArch64 only.
//!
//! The reference's `errno` and floating-point exception signalling has no
//! effect on the value and is left out; every result, special cases
//! included, is the reference's in round-to-nearest. A signalling NaN is
//! not distinguished from a quiet one.

use crate::math_tables::{
    EXP_POLY, EXP_TAB, INV_LN2_N, LN2_HI, LN2_LO, LOG_POLY, LOG_POLY1, LOG_TAB, LOG_TAB2,
    NEG_LN2_HI_N, NEG_LN2_LO_N, POW_LOG_POLY, POW_LOG_TAB, SHIFT,
};

const EXP_TABLE_BITS: u32 = 7;
const N_EXP: u64 = 1 << EXP_TABLE_BITS;
const N_LOG: u64 = 1 << 7;
/// pow's sign bias: added to `ki`, it sets the sign bit of the scale.
const SIGN_BIAS: u64 = 0x800 << EXP_TABLE_BITS;

const P1009: f64 = f64::from_bits((1023 + 1009) << 52); // 0x1p1009
const PM1022: f64 = f64::from_bits(1 << 52); // 0x1p-1022
const P52: f64 = f64::from_bits((1023 + 52) << 52); // 0x1p52
const P27: f64 = f64::from_bits((1023 + 27) << 52); // 0x1p27
const P769: f64 = f64::from_bits((1023 + 769) << 52); // 0x1p769
const PM767: f64 = f64::from_bits((1023 - 767) << 52); // 0x1p-767

/// Sign and exponent: the top 12 bits.
#[inline(always)]
fn top12(x: f64) -> u32 {
    (x.to_bits() >> 52) as u32
}

/// An overflow to infinity (`__math_oflow`), negative when `sign`.
#[inline(never)]
fn overflow(sign: bool) -> f64 {
    (if sign { -P769 } else { P769 }) * P769
}

/// An underflow to zero (`__math_uflow`), negative when `sign`.
#[inline(never)]
fn underflow(sign: bool) -> f64 {
    (if sign { -PM767 } else { PM767 }) * PM767
}

/// A NaN (`__math_invalid`). The reference computes `(x - x) / (x - x)`,
/// whose sign bit differs between x86 and AArch64; a constant does not.
#[inline(never)]
fn invalid(_x: f64) -> f64 {
    f64::NAN
}

/// exp's result near overflow or underflow, computed as `scale * (1 + tmp)`
/// without an intermediate rounding (exp.c's `specialcase`).
// The reference's statements, as written, so the port reads against it.
#[allow(clippy::assign_op_pattern)]
#[cold]
fn exp_special(tmp: f64, sbits: u64, ki: u64) -> f64 {
    if ki & 0x8000_0000 == 0 {
        // k > 0: the exponent of scale may have overflowed by <= 460.
        let scale = f64::from_bits(sbits.wrapping_sub(1009 << 52));
        return P1009 * (scale + scale * tmp);
    }
    // k < 0: the subnormal range.
    let scale = f64::from_bits(sbits.wrapping_add(1022 << 52));
    let mut y = scale + scale * tmp;
    if y < 1.0 {
        // Round to the right precision before scaling, to avoid a double
        // rounding.
        let mut lo = scale - y + scale * tmp;
        let hi = 1.0 + y;
        lo = 1.0 - hi + y + lo;
        y = (hi + lo) - 1.0;
        if y == 0.0 {
            y = 0.0; // not -0.0
        }
    }
    PM1022 * y
}

/// pow's version of `exp_special`, signed (pow.c's `specialcase`).
// The reference's statements, as written, so the port reads against it.
#[allow(clippy::assign_op_pattern)]
#[cold]
fn pow_exp_special(tmp: f64, sbits: u64, ki: u64) -> f64 {
    if ki & 0x8000_0000 == 0 {
        let scale = f64::from_bits(sbits.wrapping_sub(1009 << 52));
        let y = scale + scale * tmp;
        return y * P1009;
    }
    let sbits = sbits.wrapping_add(1022 << 52);
    let scale = f64::from_bits(sbits);
    let mut y = scale + scale * tmp;
    if y.abs() < 1.0 {
        let one = if y < 0.0 { -1.0 } else { 1.0 };
        let mut lo = scale - y + scale * tmp;
        let hi = one + y;
        lo = one - hi + y + lo;
        y = (hi + lo) - one;
        if y == 0.0 {
            y = f64::from_bits(sbits & 0x8000_0000_0000_0000);
        }
    }
    PM1022 * y
}

/// The reduction and polynomial both `exp` and `pow` end in:
/// `exp(x + xtail) = 2^(k/N) * exp(r)`, returned as `(tmp, sbits, ki)` with
/// `scale = asdouble(sbits)` and the result `scale + scale * tmp`.
#[inline(always)]
fn exp_core<const TAIL: bool>(x: f64, xtail: f64, sign_bias: u64) -> (f64, u64, u64) {
    let z = INV_LN2_N * x;
    let mut kd = z + SHIFT;
    let ki = kd.to_bits();
    kd -= SHIFT;
    let mut r = x + kd * NEG_LN2_HI_N + kd * NEG_LN2_LO_N;
    if TAIL {
        r += xtail;
    }
    let idx = (2 * (ki % N_EXP)) as usize;
    let top = ki.wrapping_add(sign_bias) << (52 - EXP_TABLE_BITS);
    let tail = f64::from_bits(EXP_TAB[idx]);
    let sbits = EXP_TAB[idx + 1].wrapping_add(top);
    let r2 = r * r;
    let [c2, c3, c4, c5] = EXP_POLY;
    let tmp = tail + r + r2 * (c2 + r * c3) + r2 * r2 * (c4 + r * c5);
    (tmp, sbits, ki)
}

/// `e^x`.
// A call, not inlined: inlined at every call site, its tables and
// polynomials bloat the column loop that inlines every sampler.
#[inline(never)]
pub fn exp(x: f64) -> f64 {
    let mut abstop = top12(x) & 0x7ff;
    let tiny = top12(f64::from_bits((1023 - 54) << 52)); // 0x1p-54
    if abstop.wrapping_sub(tiny) >= top12(512.0).wrapping_sub(tiny) {
        if abstop.wrapping_sub(tiny) >= 0x8000_0000 {
            // |x| < 2^-54, and 0.
            return 1.0 + x;
        }
        if abstop >= top12(1024.0) {
            if x.to_bits() == f64::NEG_INFINITY.to_bits() {
                return 0.0;
            }
            if abstop >= top12(f64::INFINITY) {
                return 1.0 + x;
            }
            return if x.to_bits() >> 63 != 0 {
                underflow(false)
            } else {
                overflow(false)
            };
        }
        // A large x: special-cased below.
        abstop = 0;
    }
    let (tmp, sbits, ki) = exp_core::<false>(x, 0.0, 0);
    if abstop == 0 {
        return exp_special(tmp, sbits, ki);
    }
    let scale = f64::from_bits(sbits);
    scale + scale * tmp
}

/// `ln x`.
// A call, not inlined: inlined at every call site, its tables and
// polynomials bloat the column loop that inlines every sampler.
#[inline(never)]
pub fn ln(x: f64) -> f64 {
    const OFF: u64 = 0x3fe6_0000_0000_0000;
    const LO: u64 = (1.0f64 - 0.0625).to_bits(); // 1 - 0x1p-4
    const HI: u64 = 1.064_697_265_625f64.to_bits(); // 1 + 0x1.09p-4
    let mut ix = x.to_bits();
    let top = (ix >> 48) as u32;

    if ix.wrapping_sub(LO) < HI - LO {
        // Close to 1: a polynomial in x - 1.
        if ix == 1.0f64.to_bits() {
            return 0.0;
        }
        let b = LOG_POLY1;
        let r = x - 1.0;
        let r2 = r * r;
        let r3 = r * r2;
        let mut y = r3
            * (b[1]
                + r * b[2]
                + r2 * b[3]
                + r3 * (b[4]
                    + r * b[5]
                    + r2 * b[6]
                    + r3 * (b[7] + r * b[8] + r2 * b[9] + r3 * b[10])));
        // Worst-case error is around 0.507 ULP.
        let mut w = r * P27;
        let rhi = r + w - w;
        let rlo = r - rhi;
        w = rhi * rhi * b[0]; // b[0] == -0.5
        let hi = r + w;
        let mut lo = r - hi + w;
        lo += b[0] * rlo * (rhi + r);
        y += lo;
        y += hi;
        return y;
    }
    if top.wrapping_sub(0x0010) >= 0x7ff0 - 0x0010 {
        // x < 0x1p-1022, or inf or nan.
        if ix.wrapping_mul(2) == 0 {
            return -1.0 / 0.0; // ln 0 = -inf
        }
        if ix == f64::INFINITY.to_bits() {
            return x;
        }
        if (top & 0x8000) != 0 || (top & 0x7ff0) == 0x7ff0 {
            return invalid(x);
        }
        // Subnormal: normalize.
        ix = (x * P52).to_bits();
        ix = ix.wrapping_sub(52 << 52);
    }

    // x = 2^k z, with z in [OFF, 2*OFF) and in the ith of N subintervals.
    let tmp = ix.wrapping_sub(OFF);
    let i = ((tmp >> (52 - 7)) % N_LOG) as usize;
    let k = (tmp as i64) >> 52; // arithmetic shift
    let iz = ix.wrapping_sub(tmp & (0xfff << 52));
    let (invc, logc) = LOG_TAB[i];
    let z = f64::from_bits(iz);

    // ln x = log1p(z/c - 1) + ln c + k*ln 2, r ~= z/c - 1.
    let (chi, clo) = LOG_TAB2[i];
    let r = (z - chi - clo) * invc;
    let kd = k as f64;
    let w = kd * LN2_HI + logc;
    let hi = w + r;
    let lo = w - hi + r + kd * LN2_LO;
    let r2 = r * r;
    let a = LOG_POLY;
    lo + r2 * a[0] + r * r2 * (a[1] + r * a[2] + r2 * (a[3] + r * a[4])) + hi
}

/// `ln x` as `y + tail`, `tail` holding about 15 more bits (pow.c's
/// `log_inline`). `ix` is `x`'s bits, a subnormal normalized.
#[inline(always)]
fn pow_log(ix: u64) -> (f64, f64) {
    const OFF: u64 = 0x3fe6_9555_0000_0000;
    let tmp = ix.wrapping_sub(OFF);
    let i = ((tmp >> (52 - 7)) % N_LOG) as usize;
    let k = (tmp as i64) >> 52;
    let iz = ix.wrapping_sub(tmp & (0xfff << 52));
    let z = f64::from_bits(iz);
    let kd = k as f64;
    let (invc, logc, logctail) = POW_LOG_TAB[i];

    // Split z so rhi, rlo and rhi*rhi are exact and |rlo| <= |r|.
    let zhi = f64::from_bits(iz.wrapping_add(1 << 31) & (u64::MAX << 32));
    let zlo = z - zhi;
    let rhi = zhi * invc - 1.0;
    let rlo = zlo * invc;
    let r = rhi + rlo;

    // k*ln2 + ln c + r.
    let t1 = kd * LN2_HI + logc;
    let t2 = t1 + r;
    let lo1 = kd * LN2_LO + logctail;
    let lo2 = t1 - t2 + r;

    let a = POW_LOG_POLY;
    let ar = a[0] * r; // a[0] = -0.5
    let ar2 = r * ar;
    let ar3 = r * ar2;
    let arhi = a[0] * rhi;
    let arhi2 = rhi * arhi;
    let hi = t2 + arhi2;
    let lo3 = rlo * (ar + arhi);
    let lo4 = t2 - hi + arhi2;
    let p = ar3 * (a[1] + r * a[2] + ar2 * (a[3] + r * a[4] + ar2 * (a[5] + r * a[6])));
    let lo = lo1 + lo2 + lo3 + lo4 + p;
    let y = hi + lo;
    (y, hi - y + lo)
}

/// `exp(x + xtail)`, negative when `sign_bias` is set (pow.c's
/// `exp_inline`).
#[inline(always)]
fn pow_exp(x: f64, xtail: f64, sign_bias: u64) -> f64 {
    let mut abstop = top12(x) & 0x7ff;
    let tiny = top12(f64::from_bits((1023 - 54) << 52));
    if abstop.wrapping_sub(tiny) >= top12(512.0).wrapping_sub(tiny) {
        if abstop.wrapping_sub(tiny) >= 0x8000_0000 {
            let one = 1.0 + x;
            return if sign_bias != 0 { -one } else { one };
        }
        if abstop >= top12(1024.0) {
            // inf and nan are already handled.
            return if x.to_bits() >> 63 != 0 {
                underflow(sign_bias != 0)
            } else {
                overflow(sign_bias != 0)
            };
        }
        abstop = 0;
    }
    let (tmp, sbits, ki) = exp_core::<true>(x, xtail, sign_bias);
    if abstop == 0 {
        return pow_exp_special(tmp, sbits, ki);
    }
    let scale = f64::from_bits(sbits);
    scale + scale * tmp
}

/// 0 when `iy` is not an integer, 1 when odd, 2 when even.
#[inline(always)]
fn checkint(iy: u64) -> u32 {
    let e = (iy >> 52 & 0x7ff) as u32;
    if e < 0x3ff {
        return 0;
    }
    if e > 0x3ff + 52 {
        return 2;
    }
    if iy & ((1 << (0x3ff + 52 - e)) - 1) != 0 {
        return 0;
    }
    if iy & (1 << (0x3ff + 52 - e)) != 0 {
        return 1;
    }
    2
}

/// Whether `i` is the bits of 0, an infinity or a NaN.
#[inline(always)]
fn zeroinfnan(i: u64) -> bool {
    i.wrapping_mul(2).wrapping_sub(1) >= f64::INFINITY.to_bits().wrapping_mul(2).wrapping_sub(1)
}

/// `x^y`.
// A call, not inlined: inlined at every call site, its tables and
// polynomials bloat the column loop that inlines every sampler.
#[inline(never)]
pub fn pow(x: f64, y: f64) -> f64 {
    let mut sign_bias = 0;
    let mut ix = x.to_bits();
    let iy = y.to_bits();
    let mut topx = top12(x);
    let topy = top12(y);
    if topx.wrapping_sub(0x001) >= 0x7ff - 0x001
        || (topy & 0x7ff).wrapping_sub(0x3be) >= 0x43e - 0x3be
    {
        // x < 0x1p-126 or inf or nan, or |y| < 0x1p-65 or |y| >= 0x1p63 or nan.
        if zeroinfnan(iy) {
            if iy.wrapping_mul(2) == 0 {
                return 1.0;
            }
            if ix == 1.0f64.to_bits() {
                return 1.0;
            }
            let inf2 = f64::INFINITY.to_bits().wrapping_mul(2);
            if ix.wrapping_mul(2) > inf2 || iy.wrapping_mul(2) > inf2 {
                return x + y;
            }
            if ix.wrapping_mul(2) == 1.0f64.to_bits().wrapping_mul(2) {
                return 1.0;
            }
            if (ix.wrapping_mul(2) < 1.0f64.to_bits().wrapping_mul(2)) == (iy >> 63 == 0) {
                return 0.0; // |x| < 1 and y == inf, or |x| > 1 and y == -inf
            }
            return y * y;
        }
        if zeroinfnan(ix) {
            let mut x2 = x * x;
            if ix >> 63 != 0 && checkint(iy) == 1 {
                x2 = -x2;
            }
            return if iy >> 63 != 0 { 1.0 / x2 } else { x2 };
        }
        // x and y are non-zero finite.
        if ix >> 63 != 0 {
            // x < 0.
            match checkint(iy) {
                0 => return invalid(x),
                1 => sign_bias = SIGN_BIAS,
                _ => {}
            }
            ix &= 0x7fff_ffff_ffff_ffff;
            topx &= 0x7ff;
        }
        if (topy & 0x7ff).wrapping_sub(0x3be) >= 0x43e - 0x3be {
            // sign_bias is 0 here: y is not odd.
            if ix == 1.0f64.to_bits() {
                return 1.0;
            }
            if (topy & 0x7ff) < 0x3be {
                // |y| < 2^-65: x^y ~= 1 + y*ln x.
                return if ix > 1.0f64.to_bits() {
                    1.0 + y
                } else {
                    1.0 - y
                };
            }
            return if (ix > 1.0f64.to_bits()) == (topy < 0x800) {
                overflow(false)
            } else {
                underflow(false)
            };
        }
        if topx == 0 {
            // Subnormal x: normalize so the exponent becomes negative.
            ix = (x * P52).to_bits();
            ix &= 0x7fff_ffff_ffff_ffff;
            ix = ix.wrapping_sub(52 << 52);
        }
    }

    let (hi, lo) = pow_log(ix);
    let yhi = f64::from_bits(iy & (u64::MAX << 27));
    let ylo = y - yhi;
    let lhi = f64::from_bits(hi.to_bits() & (u64::MAX << 27));
    let llo = hi - lhi + lo;
    let ehi = yhi * lhi;
    let elo = ylo * lhi + y * llo; // |elo| < |ehi| * 2^-25
    pow_exp(ehi, elo, sign_bias)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{RngExt, SeedableRng};
    use rand_xoshiro::Xoshiro256PlusPlus;

    /// The distance between two finite floats, in units in the last place.
    fn ulps(a: f64, b: f64) -> u64 {
        let key = |x: f64| {
            let i = x.to_bits() as i64;
            if i < 0 { i64::MIN - i } else { i }
        };
        key(a).abs_diff(key(b))
    }

    fn same(a: f64, b: f64) -> bool {
        a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
    }

    /// Within one ulp of the platform's function -- itself within an ulp
    /// of the true value -- over the ranges the samplers use and beyond.
    #[test]
    fn exp_ln_and_pow_are_accurate() {
        let mut g = Xoshiro256PlusPlus::seed_from_u64(5);
        for _ in 0..1_000_000 {
            let x = g.random::<f64>() * 1400.0 - 700.0;
            assert!(ulps(exp(x), x.exp()) <= 1, "exp({x:e})");
            let u = g.random::<f64>();
            let any = f64::from_bits(g.random::<u64>() >> 1); // any positive
            for v in [u, any, 1.0 + (u - 0.5) / 8.0] {
                if v > 0.0 && v.is_finite() {
                    assert!(ulps(ln(v), v.ln()) <= 1, "ln({v:e})");
                }
            }
            let p = g.random::<f64>() * 20.0 + 0.05;
            let q = g.random::<f64>() * 60.0;
            assert!(ulps(pow(u, p), u.powf(p)) <= 1, "pow({u:e}, {p})");
            assert!(
                ulps(pow(q, (p * 0.5).floor()), q.powf((p * 0.5).floor())) <= 1,
                "pow({q}, k)"
            );
        }
    }

    #[test]
    fn the_edges_are_ieee_754s() {
        for (x, want) in [
            (0.0, 1.0),
            (-0.0, 1.0),
            (f64::INFINITY, f64::INFINITY),
            (f64::NEG_INFINITY, 0.0),
            (710.0, f64::INFINITY),
            (-746.0, 0.0),
            (f64::NAN, f64::NAN),
        ] {
            assert!(same(exp(x), want), "exp({x})");
        }
        // The subnormal range, and just above it.
        for x in [-708.5, -720.0, -744.0, 709.7] {
            assert!(ulps(exp(x), x.exp()) <= 1, "exp({x})");
        }
        for (x, want) in [
            (1.0, 0.0),
            (0.0, f64::NEG_INFINITY),
            (-0.0, f64::NEG_INFINITY),
            (f64::INFINITY, f64::INFINITY),
            (-1.0, f64::NAN),
            (f64::NAN, f64::NAN),
        ] {
            assert!(same(ln(x), want), "ln({x})");
        }
        let tiny = f64::from_bits(1); // the smallest subnormal
        assert!(ulps(ln(tiny), tiny.ln()) <= 1);
        for (x, y, want) in [
            (0.5, 0.0, 1.0),
            (1.0, f64::NAN, 1.0),
            (2.0, f64::INFINITY, f64::INFINITY),
            (0.5, f64::INFINITY, 0.0),
            (-2.0, 3.0, -8.0),
            (-2.0, 0.5, f64::NAN),
            (0.0, -1.0, f64::INFINITY),
            (-0.0, -1.0, f64::NEG_INFINITY),
            (10.0, 400.0, f64::INFINITY),
            (10.0, -400.0, 0.0),
        ] {
            assert!(same(pow(x, y), want), "pow({x}, {y})");
        }
        assert!(ulps(pow(tiny, 0.5), tiny.powf(0.5)) <= 1);
    }
}

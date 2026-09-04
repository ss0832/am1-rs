// SPDX-License-Identifier: GPL-3.0-or-later

//! A `Scalar` abstraction over `f64` and a forward-mode dual number carrying three spatial
//! partial derivatives. The integral kernels ([`crate::integrals`], [`crate::overlap`],
//! [`crate::repulsion`]) are written generically over `Scalar`, so instantiating them at
//! `f64` gives the (validated) energy path and at [`Dual`] gives **exact closed-form
//! derivatives** of the same expressions with respect to an interatomic displacement — the
//! basis of the fully analytic gradient (no finite differences).

use std::ops::{Add, Div, Mul, Neg, Sub};

/// Numeric type usable by the integral kernels: `f64` (values) or [`Dual`] (value + gradient).
pub trait Scalar:
    Copy
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + Add<f64, Output = Self>
    + Sub<f64, Output = Self>
    + Mul<f64, Output = Self>
    + Div<f64, Output = Self>
{
    fn cst(x: f64) -> Self;
    fn val(&self) -> f64;
    fn sqrt(self) -> Self;
    fn exp(self) -> Self;
    fn recip(self) -> Self;
    fn powi(self, n: i32) -> Self;
    fn powf(self, x: f64) -> Self;
    fn abs(self) -> Self;
    fn ln(self) -> Self;
    /// Error function. `d/dx erf(x) = 2/√π · e^{−x²}`.
    fn erf(self) -> Self;
    /// Complementary error function `1 − erf(x)`, evaluated without cancellation for large
    /// `x`. `d/dx erfc(x) = −2/√π · e^{−x²}`.
    fn erfc(self) -> Self;
    /// **Scaled** complementary error function `erfcx(x) = e^{x²} erfc(x)`.
    ///
    /// Not a convenience wrapper: `e^{x²}` overflows above `x ≈ 26.6` and `erfc(x)` underflows
    /// at about the same place, so the product has to be formed as one function rather than as
    /// two. The two-dimensional (Parry) Ewald sum needs exactly this — its reciprocal term is
    /// `e^{hz} erfc(h/2α + αz)`, where both factors leave double precision long before the
    /// product does. See [`crate::pbc::ewald2d`].
    ///
    /// `erfcx` decays as `1/(x√π)`, so it stays representable everywhere `erfc` does not.
    fn erfcx(self) -> Self;
    /// Cosine. Needed by the two-dimensional Ewald sum, which is differentiated **through**
    /// its `cos(h·ρ)` phase rather than applying the chain rule to it by hand the way the
    /// three-dimensional sum does.
    fn cos(self) -> Self;
}

/// `2/√π`, the derivative prefactor shared by `erf` and `erfc`.
pub(crate) const TWO_OVER_SQRT_PI: f64 = std::f64::consts::FRAC_2_SQRT_PI;

/// First and second derivatives of `erf` at `x`.
///
/// `erfc` differs only in sign, and both are needed by the periodic Ewald real-space term
/// (`erfc(αr)/r`) and by the Boys function, which is why they live on the AD trait rather
/// than being computed inline at `f64`.
#[inline]
pub(crate) fn erf_derivatives(x: f64) -> (f64, f64) {
    let g = TWO_OVER_SQRT_PI * (-x * x).exp();
    (g, -2.0 * x * g)
}

/// Threshold above which [`erfcx_value`] switches to its asymptotic expansion.
///
/// At `x = 25`, `e^{x²} ≈ 2.6 × 10²⁷¹` and `erfc(x) ≈ 8.3 × 10⁻²⁷⁴` are both still normal
/// doubles, so the direct product is exact there; above `x ≈ 26.6` it is `∞ × 0`. The
/// asymptotic series is good to `~10⁻¹³` relative at the switch, so the two overlap.
const ERFCX_ASYMPTOTIC: f64 = 25.0;

/// `erfcx(x) = e^{x²} erfc(x)`, computed so that neither factor has to be representable.
pub(crate) fn erfcx_value(x: f64) -> f64 {
    if x < 0.0 {
        // erfc(−x) = 2 − erfc(x), so erfcx(−x) = 2e^{x²} − erfcx(x). This genuinely does grow
        // without bound on the negative axis; callers that would reach it (the Parry sum) are
        // written to take the `2 − erfc` route themselves, where the growth cancels.
        return 2.0 * (x * x).exp() - erfcx_value(-x);
    }
    if x < ERFCX_ASYMPTOTIC {
        return (x * x).exp() * libm::erfc(x);
    }
    // erfcx(x) ~ 1/(x√π) · Σ_n (−1)ⁿ (2n−1)!! / (2x²)ⁿ, in Horner form.
    let t = 1.0 / (2.0 * x * x);
    let series = 1.0 - t * (1.0 - 3.0 * t * (1.0 - 5.0 * t * (1.0 - 7.0 * t)));
    series / (x * std::f64::consts::PI.sqrt())
}

/// First and second derivatives of [`erfcx_value`] at `x`.
///
/// From `erfcx = e^{x²} erfc(x)` directly:
///
/// ```text
/// erfcx'(x)  = 2x·erfcx(x) − 2/√π
/// erfcx''(x) = (2 + 4x²)·erfcx(x) − 4x/√π
/// ```
///
/// Both are exact recurrences on the value, so the derivatives inherit whatever accuracy the
/// value has and never reintroduce the overflow the scaled form exists to avoid.
#[inline]
pub(crate) fn erfcx_derivatives(x: f64) -> (f64, f64, f64) {
    let e = erfcx_value(x);
    let d1 = 2.0 * x * e - TWO_OVER_SQRT_PI;
    let d2 = (2.0 + 4.0 * x * x) * e - 2.0 * TWO_OVER_SQRT_PI * x;
    (e, d1, d2)
}

impl Scalar for f64 {
    #[inline]
    fn cst(x: f64) -> Self {
        x
    }
    #[inline]
    fn val(&self) -> f64 {
        *self
    }
    #[inline]
    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }
    #[inline]
    fn exp(self) -> Self {
        f64::exp(self)
    }
    #[inline]
    fn recip(self) -> Self {
        1.0 / self
    }
    #[inline]
    fn powi(self, n: i32) -> Self {
        f64::powi(self, n)
    }
    #[inline]
    fn powf(self, x: f64) -> Self {
        f64::powf(self, x)
    }
    #[inline]
    fn abs(self) -> Self {
        f64::abs(self)
    }
    #[inline]
    fn ln(self) -> Self {
        f64::ln(self)
    }
    #[inline]
    fn erf(self) -> Self {
        libm::erf(self)
    }
    #[inline]
    fn erfc(self) -> Self {
        libm::erfc(self)
    }
    #[inline]
    fn erfcx(self) -> Self {
        erfcx_value(self)
    }
    #[inline]
    fn cos(self) -> Self {
        f64::cos(self)
    }
}

/// Forward-mode dual number: value plus three partial derivatives (∂/∂x, ∂/∂y, ∂/∂z).
#[derive(Clone, Copy, Debug)]
pub struct Dual {
    pub v: f64,
    pub d: [f64; 3],
}

impl Dual {
    #[inline]
    pub fn constant(x: f64) -> Self {
        Self { v: x, d: [0.0; 3] }
    }
    /// A variable whose derivative is the unit vector along axis `i`.
    #[inline]
    pub fn var(x: f64, i: usize) -> Self {
        let mut d = [0.0; 3];
        d[i] = 1.0;
        Self { v: x, d }
    }
    #[inline]
    fn map(self, v: f64, factor: f64) -> Self {
        // chain rule: new value `v`, derivative = factor · self.d
        Self {
            v,
            d: [self.d[0] * factor, self.d[1] * factor, self.d[2] * factor],
        }
    }
}

impl Add for Dual {
    type Output = Self;
    #[inline]
    fn add(self, o: Self) -> Self {
        Self {
            v: self.v + o.v,
            d: [self.d[0] + o.d[0], self.d[1] + o.d[1], self.d[2] + o.d[2]],
        }
    }
}
impl Sub for Dual {
    type Output = Self;
    #[inline]
    fn sub(self, o: Self) -> Self {
        Self {
            v: self.v - o.v,
            d: [self.d[0] - o.d[0], self.d[1] - o.d[1], self.d[2] - o.d[2]],
        }
    }
}
impl Mul for Dual {
    type Output = Self;
    #[inline]
    fn mul(self, o: Self) -> Self {
        Self {
            v: self.v * o.v,
            d: [
                self.d[0] * o.v + self.v * o.d[0],
                self.d[1] * o.v + self.v * o.d[1],
                self.d[2] * o.v + self.v * o.d[2],
            ],
        }
    }
}
impl Div for Dual {
    type Output = Self;
    #[inline]
    fn div(self, o: Self) -> Self {
        let inv = 1.0 / o.v;
        let inv2 = inv * inv;
        Self {
            v: self.v * inv,
            d: [
                (self.d[0] * o.v - self.v * o.d[0]) * inv2,
                (self.d[1] * o.v - self.v * o.d[1]) * inv2,
                (self.d[2] * o.v - self.v * o.d[2]) * inv2,
            ],
        }
    }
}
impl Neg for Dual {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self {
            v: -self.v,
            d: [-self.d[0], -self.d[1], -self.d[2]],
        }
    }
}
// Mixed ops with f64 (constant), for readable kernel code.
impl Add<f64> for Dual {
    type Output = Self;
    #[inline]
    fn add(self, o: f64) -> Self {
        Self {
            v: self.v + o,
            d: self.d,
        }
    }
}
impl Sub<f64> for Dual {
    type Output = Self;
    #[inline]
    fn sub(self, o: f64) -> Self {
        Self {
            v: self.v - o,
            d: self.d,
        }
    }
}
impl Mul<f64> for Dual {
    type Output = Self;
    #[inline]
    fn mul(self, o: f64) -> Self {
        Self {
            v: self.v * o,
            d: [self.d[0] * o, self.d[1] * o, self.d[2] * o],
        }
    }
}
impl Div<f64> for Dual {
    type Output = Self;
    #[inline]
    fn div(self, o: f64) -> Self {
        self * (1.0 / o)
    }
}

impl Scalar for Dual {
    #[inline]
    fn cst(x: f64) -> Self {
        Dual::constant(x)
    }
    #[inline]
    fn val(&self) -> f64 {
        self.v
    }
    #[inline]
    fn sqrt(self) -> Self {
        let s = self.v.sqrt();
        self.map(s, if s > 0.0 { 0.5 / s } else { 0.0 })
    }
    #[inline]
    fn exp(self) -> Self {
        let e = self.v.exp();
        self.map(e, e)
    }
    #[inline]
    fn recip(self) -> Self {
        let r = 1.0 / self.v;
        self.map(r, -r * r)
    }
    #[inline]
    fn powi(self, n: i32) -> Self {
        let p = self.v.powi(n);
        let dfac = if self.v != 0.0 {
            n as f64 * self.v.powi(n - 1)
        } else {
            0.0
        };
        self.map(p, dfac)
    }
    #[inline]
    fn powf(self, x: f64) -> Self {
        let p = self.v.powf(x);
        let dfac = if self.v > 0.0 {
            x * self.v.powf(x - 1.0)
        } else {
            0.0
        };
        self.map(p, dfac)
    }
    #[inline]
    fn abs(self) -> Self {
        if self.v >= 0.0 {
            self
        } else {
            -self
        }
    }
    #[inline]
    fn ln(self) -> Self {
        let l = self.v.ln();
        self.map(l, if self.v != 0.0 { 1.0 / self.v } else { 0.0 })
    }
    #[inline]
    fn erf(self) -> Self {
        let (d1, _) = erf_derivatives(self.v);
        self.map(libm::erf(self.v), d1)
    }
    #[inline]
    fn erfc(self) -> Self {
        let (d1, _) = erf_derivatives(self.v);
        self.map(libm::erfc(self.v), -d1)
    }
    #[inline]
    fn erfcx(self) -> Self {
        let (v, d1, _) = erfcx_derivatives(self.v);
        self.map(v, d1)
    }
    #[inline]
    fn cos(self) -> Self {
        let (s, c) = self.v.sin_cos();
        self.map(c, -s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn erfcx_stays_finite_where_its_two_factors_do_not() {
        // The whole reason `erfcx` exists as its own function. Beyond `x ≈ 26.6` the naive
        // product is `∞ × 0`; the scaled form decays as `1/(x√π)` and stays exact.
        for &x in &[0.0_f64, 0.5, 1.0, 5.0, 20.0, 24.9, 25.1, 40.0, 200.0, 1.0e6] {
            let naive = (x * x).exp() * libm::erfc(x);
            let scaled = erfcx_value(x);
            assert!(
                scaled.is_finite() && scaled > 0.0,
                "erfcx({x}) = {scaled}, which is not a usable value"
            );
            // Where the naive form still works, the two must agree.
            if naive.is_finite() && naive > 0.0 {
                let rel = (naive - scaled).abs() / scaled;
                assert!(
                    rel < 1.0e-12,
                    "erfcx({x}): scaled {scaled:e} vs naive {naive:e}"
                );
            }
            // The asymptote, which is what makes the large-argument branch checkable at all.
            if x > 50.0 {
                let asymptote = 1.0 / (x * std::f64::consts::PI.sqrt());
                let rel = (scaled - asymptote).abs() / asymptote;
                assert!(
                    rel < 1.0e-3,
                    "erfcx({x}) = {scaled:e}, asymptote {asymptote:e}"
                );
            }
        }
        // Known value: erfcx(0) = 1 exactly.
        assert!((erfcx_value(0.0) - 1.0).abs() < 1.0e-15);
        eprintln!(
            "erfcx: naive overflows at x>26.6; scaled gives erfcx(40) = {:.6e}, erfcx(1e6) = {:.6e}",
            erfcx_value(40.0),
            erfcx_value(1.0e6)
        );
    }

    #[test]
    fn erfcx_derivatives_match_finite_differences() {
        // The derivatives come from the exact recurrence `erfcx' = 2x·erfcx − 2/√π` rather than
        // from differentiating the product, so they need checking on their own terms.
        let h = 1.0e-5;
        let mut worst1 = 0.0_f64;
        let mut worst2 = 0.0_f64;
        for i in 0..=120 {
            let x = -3.0 + 0.05 * i as f64;
            let (v, d1, d2) = erfcx_derivatives(x);
            let fd1 = (erfcx_value(x + h) - erfcx_value(x - h)) / (2.0 * h);
            let fd2 = (erfcx_value(x + h) - 2.0 * v + erfcx_value(x - h)) / (h * h);
            worst1 = worst1.max((d1 - fd1).abs() / fd1.abs().max(1.0));
            worst2 = worst2.max((d2 - fd2).abs() / fd2.abs().max(1.0));
        }
        eprintln!("erfcx derivatives vs FD: first {worst1:.2e}, second {worst2:.2e}");
        assert!(worst1 < 1.0e-8, "erfcx' mismatch {worst1:.3e}");
        assert!(worst2 < 1.0e-4, "erfcx'' mismatch {worst2:.3e}");
    }

    #[test]
    fn dual_matches_analytic_derivative() {
        // f(x) = exp(-x^2) / sqrt(x^2 + 4);  check f'(x) at x=1.3.
        let x0 = 1.3;
        let x = Dual::var(x0, 0);
        let f = (-(x * x)).exp() / (x * x + 4.0).sqrt();
        // analytic derivative
        let g = |x: f64| (-(x * x)).exp() / (x * x + 4.0).sqrt();
        let h = 1e-6;
        let fd = (g(x0 + h) - g(x0 - h)) / (2.0 * h);
        assert!((f.d[0] - fd).abs() < 1e-8, "{} vs {}", f.d[0], fd);
    }
}

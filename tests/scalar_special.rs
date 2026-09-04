// SPDX-License-Identifier: GPL-3.0-or-later

//! `erf`, `erfc` and `ln` on the AD scalar trait.
//!
//! These are prerequisites rather than features: the periodic Ewald real-space term is
//! `erfc(αr)/r`, and the Boys function underlying SAM1's Gaussian two-electron integrals is
//! built on `erf`. Both need to differentiate exactly through `Dual` and `Dual2`, because the
//! periodic gradient, stress, Hessian and DFPT all run through the same kernels.
//!
//! Values are checked against independently known references; derivatives against Richardson
//! finite differences of the value.

use am1_rs::dual::{Dual, Scalar};
use am1_rs::dual2::Dual2;

fn richardson_first(f: impl Fn(f64) -> f64, x: f64, h: f64) -> f64 {
    let d = |hh: f64| (f(x + hh) - f(x - hh)) / (2.0 * hh);
    (4.0 * d(h / 2.0) - d(h)) / 3.0
}

#[test]
fn erf_matches_known_values() {
    // Reference values, correctly rounded to double precision.
    let cases: [(f64, f64); 7] = [
        (0.0, 0.0),
        (0.5, 0.520_499_877_813_046_5),
        (1.0, 0.842_700_792_949_714_9),
        (1.5, 0.966_105_146_475_310_7),
        (2.0, 0.995_322_265_018_952_7),
        (3.0, 0.999_977_909_503_001_4),
        (-1.0, -0.842_700_792_949_714_9),
    ];
    let mut worst = 0.0_f64;
    for (x, want) in cases {
        worst = worst.max((Scalar::erf(x) - want).abs());
    }
    eprintln!("    max |erf - reference| = {worst:.3e}");
    assert!(worst < 1.0e-15, "erf off by {worst:.3e}");
}

#[test]
fn erfc_stays_accurate_where_erf_would_cancel() {
    // The point of a separate erfc: 1 - erf(x) loses every significant digit once erf(x)
    // approaches 1. At x = 6, erfc is ~2e-17 while 1 - erf(6) rounds to exactly zero.
    let x = 6.0_f64;
    let direct = Scalar::erfc(x);
    let cancelled = 1.0 - Scalar::erf(x);
    eprintln!("    erfc(6) = {direct:.6e}, 1 - erf(6) = {cancelled:.6e}");
    assert!(direct > 0.0, "erfc(6) underflowed to {direct}");
    assert!(
        (direct - 2.151_973_671_249_650_4e-17).abs() < 1.0e-25,
        "erfc(6) = {direct:.17e}"
    );
    // The identity still holds where it is numerically meaningful.
    for x in [0.0, 0.25, 0.5, 1.0, 1.5] {
        let s: f64 = Scalar::erf(x) + Scalar::erfc(x);
        assert!((s - 1.0).abs() < 1.0e-15, "erf + erfc != 1 at {x}");
    }
}

#[test]
fn erf_erfc_ln_first_derivatives_match_finite_differences() {
    let mut worst = 0.0_f64;
    for &x in &[0.05_f64, 0.25, 0.5, 1.0, 1.7, 2.5, 4.0, -0.75, -2.2] {
        let d_erf = Scalar::erf(Dual::var(x, 0)).d[0];
        let d_erfc = Scalar::erfc(Dual::var(x, 0)).d[0];
        let fd_erf = richardson_first(Scalar::erf, x, 1.0e-3);
        let fd_erfc = richardson_first(Scalar::erfc, x, 1.0e-3);
        worst = worst.max((d_erf - fd_erf).abs());
        worst = worst.max((d_erfc - fd_erfc).abs());
        // erfc' = -erf' exactly.
        assert!((d_erf + d_erfc).abs() < 1.0e-15, "erfc' != -erf' at {x}");

        if x > 0.0 {
            let d_ln = Scalar::ln(Dual::var(x, 0)).d[0];
            let fd_ln = richardson_first(Scalar::ln, x, 1.0e-3 * x);
            worst = worst.max((d_ln - fd_ln).abs());
        }
    }
    eprintln!("    max |analytic - Richardson| first derivative = {worst:.3e}");
    assert!(worst < 1.0e-10, "first derivative off by {worst:.3e}");
}

#[test]
fn erf_erfc_ln_second_derivatives_match_finite_differences() {
    // Differentiate the analytic first derivative, so the reference is not degraded by
    // second-differencing the value.
    let mut worst = 0.0_f64;
    for &x in &[0.05_f64, 0.25, 0.5, 1.0, 1.7, 2.5, 4.0, -0.75, -2.2] {
        let h2_erf = Scalar::erf(Dual2::var(x, 0)).h[0][0];
        let h2_erfc = Scalar::erfc(Dual2::var(x, 0)).h[0][0];
        let fd_erf = richardson_first(|t| Scalar::erf(Dual::var(t, 0)).d[0], x, 1.0e-3);
        let fd_erfc = richardson_first(|t| Scalar::erfc(Dual::var(t, 0)).d[0], x, 1.0e-3);
        worst = worst.max((h2_erf - fd_erf).abs());
        worst = worst.max((h2_erfc - fd_erfc).abs());
        assert!(
            (h2_erf + h2_erfc).abs() < 1.0e-15,
            "erfc'' != -erf'' at {x}"
        );

        if x > 0.0 {
            let h2_ln = Scalar::ln(Dual2::var(x, 0)).h[0][0];
            let fd_ln = richardson_first(|t| Scalar::ln(Dual::var(t, 0)).d[0], x, 1.0e-3 * x);
            worst = worst.max((h2_ln - fd_ln).abs());
        }
    }
    eprintln!("    max |analytic - Richardson| second derivative = {worst:.3e}");
    assert!(worst < 1.0e-9, "second derivative off by {worst:.3e}");
}

#[test]
fn the_ewald_real_space_term_differentiates() {
    // The shape the periodic code will actually use: erfc(alpha*r)/r as a function of a
    // three-dimensional displacement, so the sqrt chain is exercised alongside erfc.
    let alpha = 0.35_f64;
    let f = |v: [f64; 3]| {
        let r = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        Scalar::erfc(alpha * r) / r
    };
    let d = [2.4_f64, -1.1, 0.7];

    let dv = [
        Dual2::var(d[0], 0),
        Dual2::var(d[1], 1),
        Dual2::var(d[2], 2),
    ];
    let r = (dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2]).sqrt();
    let ad = Scalar::erfc(r * alpha) / r;

    assert!((ad.v - f(d)).abs() < 1.0e-15, "value mismatch");

    let h = 1.0e-4;
    let mut worst_g = 0.0_f64;
    let mut worst_h = 0.0_f64;
    for a in 0..3 {
        let (mut p, mut m) = (d, d);
        p[a] += h;
        m[a] -= h;
        worst_g = worst_g.max((ad.g[a] - (f(p) - f(m)) / (2.0 * h)).abs());

        // Second derivatives against a finite difference of the analytic gradient.
        let grad = |v: [f64; 3]| {
            let dv = [Dual::var(v[0], 0), Dual::var(v[1], 1), Dual::var(v[2], 2)];
            let r = (dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2]).sqrt();
            (Scalar::erfc(r * alpha) / r).d
        };
        let (gp, gm) = (grad(p), grad(m));
        for b in 0..3 {
            worst_h = worst_h.max((ad.h[a][b] - (gp[b] - gm[b]) / (2.0 * h)).abs());
        }
    }
    eprintln!("    erfc(ar)/r : max gradient error {worst_g:.3e}, max Hessian error {worst_h:.3e}");
    assert!(worst_g < 1.0e-9, "gradient off by {worst_g:.3e}");
    assert!(worst_h < 1.0e-7, "Hessian off by {worst_h:.3e}");
}

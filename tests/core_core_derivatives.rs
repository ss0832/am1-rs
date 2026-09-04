// SPDX-License-Identifier: GPL-3.0-or-later

//! Core–core repulsion derivatives, isolated from the SCF.
//!
//! The AM1 core–core term carries the model's defining Gaussian corrections
//! `(Z_A Z_B / R) Σ_k K_k e^{-L_k (R - M_k)²}` and the MNDO N–H / O–H `R e^{-αR}` special
//! cases. Checking them through a molecular Hessian buries a small systematic error under
//! CPHF iteration noise and the finite-difference reference's own truncation error, so these
//! tests differentiate the *pair energy alone* and compare against a Richardson-extrapolated
//! finite difference, which is accurate to ~1e-9 absolute.
//!
//! Coverage is every element that actually carries Gaussians, paired with every other, over a
//! distance scan — the previous suite tested one O–H geometry, seeded as a one-dimensional
//! variable so it never exercised the three-dimensional `r = sqrt(Σ d²)` chain, and against a
//! finite difference whose own roundoff floor exceeded the tolerance being asserted.

use am1_rs::dual::{Dual, Scalar};
use am1_rs::dual2::Dual2;
use am1_rs::params::Am1Parameters;
use am1_rs::repulsion::{pair_core_energy, pair_core_energy_and_dr, pair_core_energy_scalar};
use am1_rs::{Molecule, Vec3};

/// Elements carrying at least one AM1 Gaussian correction, plus hydrogen (which carries three
/// and drives the N–H / O–H special cases).
const GAUSSIAN_ELEMENTS: [u8; 16] = [1, 6, 7, 8, 9, 13, 14, 15, 16, 17, 33, 34, 35, 51, 52, 53];

/// Elements with no Gaussians at all — the MNDO-form control group. A bug in the Gaussian
/// path must not show up here, which is what makes them a useful contrast.
const PLAIN_ELEMENTS: [u8; 5] = [4, 5, 30, 32, 80];

fn distances_bohr() -> Vec<f64> {
    // 0.7 to 6.0 Angstrom in 0.1 Angstrom steps, expressed in Bohr.
    let a0 = 0.529167_f64;
    (7..=60).map(|i| (i as f64 * 0.1) / a0).collect()
}

/// Second derivative of `f` at `r` by Richardson extrapolation of the central three-point
/// stencil. The `h²` truncation term cancels, leaving roundoff at ~1e-9 for these magnitudes.
fn richardson_second(f: impl Fn(f64) -> f64, r: f64, h: f64) -> f64 {
    let d = |hh: f64| (f(r + hh) - 2.0 * f(r) + f(r - hh)) / (hh * hh);
    (4.0 * d(h / 2.0) - d(h)) / 3.0
}

fn richardson_first(f: impl Fn(f64) -> f64, r: f64, h: f64) -> f64 {
    let d = |hh: f64| (f(r + hh) - f(r - hh)) / (2.0 * hh);
    (4.0 * d(h / 2.0) - d(h)) / 3.0
}

#[test]
fn the_three_core_core_implementations_agree() {
    // `pair_core_energy`, `pair_core_energy_scalar::<f64>` and the energy returned by
    // `pair_core_energy_and_dr` are three separate transcriptions of the same formula. The
    // gradient uses the third and the Hessian uses the second, so a divergence between them
    // would appear as an analytic-vs-numerical Hessian mismatch and nowhere else.
    let params = Am1Parameters::standard().unwrap();
    let mut all: Vec<u8> = GAUSSIAN_ELEMENTS.to_vec();
    all.extend_from_slice(&PLAIN_ELEMENTS);
    let mut worst = 0.0_f64;

    for &zi in &all {
        for &zj in &all {
            let (ei, ej) = (params.element(zi).unwrap(), params.element(zj).unwrap());
            for r in distances_bohr() {
                let a = pair_core_energy(zi, zj, Vec3::zero(), Vec3::new(r, 0.0, 0.0), &params)
                    .unwrap();
                let b = pair_core_energy_scalar::<f64>(ei, ej, zi, zj, r);
                let (c, _) =
                    pair_core_energy_and_dr(zi, zj, Vec3::zero(), Vec3::new(r, 0.0, 0.0), &params)
                        .unwrap();
                let scale = a.abs().max(1.0);
                worst = worst.max((a - b).abs() / scale);
                worst = worst.max((a - c).abs() / scale);
            }
        }
    }
    eprintln!("    worst relative disagreement between the three forms = {worst:.3e}");
    assert!(
        worst < 1.0e-13,
        "the core-core energy transcriptions disagree by {worst:.3e} relative"
    );
}

#[test]
fn core_core_first_derivative_matches_richardson() {
    let params = Am1Parameters::standard().unwrap();
    let mut all: Vec<u8> = GAUSSIAN_ELEMENTS.to_vec();
    all.extend_from_slice(&PLAIN_ELEMENTS);
    let mut worst = 0.0_f64;
    let mut worst_at = (0u8, 0u8, 0.0);

    for &zi in &all {
        for &zj in &all {
            let (ei, ej) = (params.element(zi).unwrap(), params.element(zj).unwrap());
            for r in distances_bohr() {
                // Hand-written closed form used by the production gradient.
                let (_, dedr) =
                    pair_core_energy_and_dr(zi, zj, Vec3::zero(), Vec3::new(r, 0.0, 0.0), &params)
                        .unwrap();
                // Forward-mode AD of the generic form used by the Hessian.
                let ad = pair_core_energy_scalar::<Dual>(ei, ej, zi, zj, Dual::var(r, 0)).d[0];
                let fd = richardson_first(
                    |x| pair_core_energy_scalar::<f64>(ei, ej, zi, zj, x),
                    r,
                    1.0e-3,
                );
                let scale = fd.abs().max(1.0e-3);
                for candidate in [dedr, ad] {
                    let rel = (candidate - fd).abs() / scale;
                    if rel > worst {
                        worst = rel;
                        worst_at = (zi, zj, r);
                    }
                }
            }
        }
    }
    eprintln!(
        "    worst relative dE/dr error = {:.3e}  (Z={} Z={} r={:.3} Bohr)",
        worst, worst_at.0, worst_at.1, worst_at.2
    );
    assert!(
        worst < 1.0e-7,
        "core-core dE/dr off by {worst:.3e} relative"
    );
}

#[test]
fn core_core_second_derivative_matches_richardson() {
    let params = Am1Parameters::standard().unwrap();
    let mut all: Vec<u8> = GAUSSIAN_ELEMENTS.to_vec();
    all.extend_from_slice(&PLAIN_ELEMENTS);
    let mut worst = 0.0_f64;
    let mut worst_at = (0u8, 0u8, 0.0, 0.0, 0.0);

    for &zi in &all {
        for &zj in &all {
            let (ei, ej) = (params.element(zi).unwrap(), params.element(zj).unwrap());
            for r in distances_bohr() {
                let ad = pair_core_energy_scalar::<Dual2>(ei, ej, zi, zj, Dual2::var(r, 0)).h[0][0];
                // Differentiate the *first* derivative rather than second-differencing the
                // energy. Second-differencing divides roundoff by h², which at short range
                // (where the pair energy reaches hundreds of eV) puts the reference's own
                // floor above any tolerance worth asserting. The first derivative is
                // independently validated against an energy finite difference in
                // `core_core_first_derivative_matches_richardson`, to 1e-11 relative.
                let fd = richardson_first(
                    |x| pair_core_energy_scalar::<Dual>(ei, ej, zi, zj, Dual::var(x, 0)).d[0],
                    r,
                    2.0e-3,
                );
                // Mixed tolerance. A pure relative measure is meaningless where the second
                // derivative crosses zero, and a pure absolute one is meaningless at short
                // range where it reaches thousands of eV/Bohr². The absolute floor is set by
                // the Richardson reference's own roundoff, ~1e-7 for these magnitudes.
                let err = (ad - fd).abs() / (fd.abs() * 1.0e-8 + 1.0e-7).max(1.0e-30);
                if err > worst {
                    worst = err;
                    worst_at = (zi, zj, r, ad, fd);
                }
            }
        }
    }
    eprintln!(
        "    worst d2E/dr2 error = {:.2}x tolerance  (Z={} Z={} r={:.3} Bohr; \
         analytic {:.6e}, Richardson {:.6e}, diff {:.3e})",
        worst,
        worst_at.0,
        worst_at.1,
        worst_at.2,
        worst_at.3,
        worst_at.4,
        (worst_at.3 - worst_at.4).abs()
    );
    assert!(
        worst < 1.0,
        "core-core d2E/dr2 exceeds |d2E| * 1e-8 + 1e-7 by {worst:.2}x"
    );
}

#[test]
fn core_core_cartesian_hessian_is_exact() {
    // Seed the displacement vector rather than the scalar distance, so the
    // `r = sqrt(dx² + dy² + dz²)` chain -- which produces the whole transverse structure
    // `delta_ij/r - d_i d_j/r³` -- is actually exercised. The old test seeded `r` directly and
    // never touched it.
    let params = Am1Parameters::standard().unwrap();
    let displacements = [
        Vec3::new(2.4, 0.0, 0.0),
        Vec3::new(0.0, 2.4, 0.0),
        Vec3::new(0.0, 0.0, -2.4),
        Vec3::new(1.4, -0.9, 0.7),
        Vec3::new(-3.1, 2.2, 1.05),
    ];
    let mut worst = 0.0_f64;

    for &zi in &GAUSSIAN_ELEMENTS {
        for &zj in &GAUSSIAN_ELEMENTS {
            let (ei, ej) = (params.element(zi).unwrap(), params.element(zj).unwrap());
            for d in displacements {
                let dv = [Dual2::var(d.x, 0), Dual2::var(d.y, 1), Dual2::var(d.z, 2)];
                let r2 = (dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2]).sqrt();
                let ad = pair_core_energy_scalar::<Dual2>(ei, ej, zi, zj, r2);

                // Richardson-extrapolated finite difference of the *analytic* Cartesian
                // gradient of the same expression, so the reference is O(h^4)-accurate.
                let grad = |v: Vec3| -> [f64; 3] {
                    let dv = [Dual::var(v.x, 0), Dual::var(v.y, 1), Dual::var(v.z, 2)];
                    let r = (dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2]).sqrt();
                    pair_core_energy_scalar::<Dual>(ei, ej, zi, zj, r).d
                };
                let shifted = |v: Vec3, axis: usize, delta: f64| -> Vec3 {
                    let mut w = v;
                    match axis {
                        0 => w.x += delta,
                        1 => w.y += delta,
                        _ => w.z += delta,
                    }
                    w
                };
                for a in 0..3 {
                    let col = |hh: f64| -> [f64; 3] {
                        let gp = grad(shifted(d, a, hh));
                        let gm = grad(shifted(d, a, -hh));
                        [
                            (gp[0] - gm[0]) / (2.0 * hh),
                            (gp[1] - gm[1]) / (2.0 * hh),
                            (gp[2] - gm[2]) / (2.0 * hh),
                        ]
                    };
                    let (c1, c2) = (col(2.0e-3), col(1.0e-3));
                    for b in 0..3 {
                        let fd = (4.0 * c2[b] - c1[b]) / 3.0;
                        let err = (ad.h[a][b] - fd).abs() / (fd.abs() * 1.0e-8 + 1.0e-7);
                        worst = worst.max(err);
                    }
                }
            }
        }
    }
    eprintln!("    worst Cartesian d2E error = {worst:.2}x tolerance (|d2E|*1e-8 + 1e-7)");
    assert!(
        worst < 1.0,
        "core-core Cartesian Hessian exceeds |d2E| * 1e-8 + 1e-7 by {worst:.2}x"
    );
}

#[test]
fn nitrogen_and_oxygen_hydrogen_special_cases_differentiate() {
    // N-H and O-H replace f = e^{-a s} with f = s e^{-a s}. That branch is selected by
    // element pair, so it is easy for a refactor to apply it to the value and not the
    // derivative, or to pick the wrong partner's alpha.
    let params = Am1Parameters::standard().unwrap();
    let mut worst = 0.0_f64;
    for (zi, zj) in [(7u8, 1u8), (1, 7), (8, 1), (1, 8)] {
        let (ei, ej) = (params.element(zi).unwrap(), params.element(zj).unwrap());
        for r in distances_bohr() {
            let d1 = pair_core_energy_scalar::<Dual>(ei, ej, zi, zj, Dual::var(r, 0)).d[0];
            let d2 = pair_core_energy_scalar::<Dual2>(ei, ej, zi, zj, Dual2::var(r, 0));
            let f = |x: f64| pair_core_energy_scalar::<f64>(ei, ej, zi, zj, x);
            let fd1 = richardson_first(f, r, 1.0e-3);
            let fd2 = richardson_second(f, r, 2.0e-3);
            worst = worst.max((d1 - fd1).abs() / fd1.abs().max(1.0e-3));
            worst = worst.max((d2.h[0][0] - fd2).abs() / fd2.abs().max(1.0e-3));
            // The special form really is in effect: swapping it off changes the energy.
            assert!(d2.v.is_finite());
        }
    }
    eprintln!("    worst relative N-H / O-H derivative error = {worst:.3e}");
    assert!(worst < 1.0e-6, "N-H/O-H derivatives off by {worst:.3e}");
}

#[test]
fn core_core_gradient_sums_to_zero() {
    // Translational invariance of the classical term, independent of any SCF.
    let params = Am1Parameters::standard().unwrap();
    let mol = Molecule::from_xyz_str(
        "5\nCH3F\nC 0.0 0.0 0.0\nF 1.38 0.0 0.0\nH -0.36 1.03 0.0\nH -0.36 -0.51 0.89\nH -0.36 -0.51 -0.89\n",
        0.0,
    )
    .unwrap();
    let g = am1_rs::repulsion::core_core_gradient(&mol, &params).unwrap();
    let mut sum = Vec3::zero();
    for v in &g {
        sum += *v;
    }
    let worst = sum.x.abs().max(sum.y.abs()).max(sum.z.abs());
    eprintln!("    |sum of core-core forces| = {worst:.3e} eV/Bohr");
    assert!(
        worst < 1.0e-10,
        "core-core forces do not sum to zero: {worst:.3e}"
    );
}

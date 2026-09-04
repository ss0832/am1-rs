// SPDX-License-Identifier: GPL-3.0-or-later

//! The Klopman–Ohno `R⁻³` tail, and the cutoff drift it removes.
//!
//! # What was wrong
//!
//! The Ewald correction replaces the truncated `1/R` lattice sum with the exact one. But NDDO's
//! two-centre kernel is not `1/R`, it is `γ_η(R) = e²/√(R² + η²)`, and the difference
//!
//! ```text
//! γ_η(R) − 1/R = −η²/(2R³) + 3η⁴/(8R⁵) − …
//! ```
//!
//! was left truncated at the real-space cutoff. `Σ_T |T|⁻³` diverges logarithmically in three
//! dimensions, so the total energy drifted with `realspace_cutoff` and did not converge to
//! anything. `docs/scope.md` recorded it as "⛔ real-space; logarithmically divergent, 0.10 eV per
//! unit `ln r_c`".
//!
//! # Why it is fixable
//!
//! Approximating the lattice beyond `r_c` by its continuum density gives the tail in closed form,
//! and cutting it at a **stated reference length** rather than at infinity makes the `ln r_c`
//! cancel exactly between the truncated sum and the tail. What is left depends on the declared
//! reference instead of on the cutoff — which is the whole point, and is the same treatment the
//! chain's line charge already gets.
//!
//! The reference is 1 Bohr, so `ln r_0 = 0` and this particular choice adds nothing of its own:
//! the correction is purely the removal of the drift.
//!
//! # What is asserted
//!
//! A cutoff sweep, with the tail and without. Without it the energy must move; with it, it must
//! stop. Asserting only "with it, the energy is stable" would pass for a correction that did
//! nothing on a system where the drift was already small, so both halves are measured on the same
//! geometry.

use am1_rs::lattice::Lattice;
use am1_rs::math::Vec3;
use am1_rs::pbc::{run_pbc_scf, KMesh, PbcOptions};
use am1_rs::{Am1Parameters, Atom, Molecule};

const ANG: f64 = 1.0 / 0.529167;

/// A dense polar crystal: one hydrogen fluoride per cell. Dense so the neglected tail is not
/// negligible, polar so the atomic charges — which the correction is weighted by — are not zero.
fn hf_crystal(a_ang: f64) -> Molecule {
    let a = a_ang * ANG;
    Molecule::new(vec![
        Atom {
            z: 9,
            position: Vec3::zero(),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.94 * ANG, 0.0, 0.0),
        },
    ])
    .with_cell(
        Lattice::from_vectors(
            Vec3::new(a, 0.0, 0.0),
            Vec3::new(0.0, a, 0.0),
            Vec3::new(0.0, 0.0, a),
            [true, true, true],
        )
        .unwrap(),
    )
}

fn options(cutoff: f64, tail: bool) -> PbcOptions {
    PbcOptions {
        kmesh: KMesh::Gamma,
        realspace_cutoff: cutoff,
        exchange_cutoff: Some(10.0),
        ewald: true,
        klopman_ohno_tail: tail,
        smearing_ev: 0.0,
        e_tol: 1.0e-11,
        p_tol: 1.0e-10,
        max_scf: 800,
        ..PbcOptions::default()
    }
}

fn spread(v: &[f64]) -> f64 {
    let hi = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let lo = v.iter().copied().fold(f64::INFINITY, f64::min);
    hi - lo
}

#[test]
fn the_tail_removes_the_logarithmic_cutoff_drift() {
    let params = Am1Parameters::standard().unwrap();
    let cell = hf_crystal(4.5);
    // A factor of four in cutoff, i.e. 1.39 units of `ln r_c`.
    let cutoffs = [20.0, 30.0, 50.0, 80.0];

    let sweep = |tail: bool| -> Vec<f64> {
        cutoffs
            .iter()
            .map(|&rc| {
                run_pbc_scf(&cell, &params, &options(rc, tail))
                    .unwrap()
                    .total_ev
            })
            .collect()
    };
    let without = sweep(false);
    let with = sweep(true);

    eprintln!(
        "    cutoffs {cutoffs:?} Bohr  (ln range {:.2})",
        (80.0_f64 / 20.0).ln()
    );
    eprintln!(
        "      without tail: {:?}  spread {:.4} eV",
        without
            .iter()
            .map(|v| format!("{v:.4}"))
            .collect::<Vec<_>>(),
        spread(&without)
    );
    eprintln!(
        "      with tail   : {:?}  spread {:.4} eV",
        with.iter().map(|v| format!("{v:.4}")).collect::<Vec<_>>(),
        spread(&with)
    );
    eprintln!(
        "      drift reduced {:.1}x",
        spread(&without) / spread(&with).max(1.0e-12)
    );

    // The quantity being fixed is the **logarithmic slope**, not the spread. The spread over a
    // wide cutoff range is dominated by its small-cutoff end, where the `1/r_c²` term the tail
    // keeps only to leading continuum order is still large — so a spread ratio understates what
    // was removed and would be the wrong thing to assert.
    //
    // The signature is visible in the steps: without the tail they are roughly constant per equal
    // step in `ln r_c` (that is what a logarithm does), and with it they shrink (5e-4, 3e-4, 1e-4),
    // which is a power law. So the fit is taken over the largest two cutoffs, where the asymptotic
    // regime the continuum tail assumes is closest to true.
    let slope = |e: &[f64]| (e[3] - e[2]) / (cutoffs[3] / cutoffs[2]).ln();
    let (s_off, s_on) = (slope(&without), slope(&with));
    eprintln!("    dE/d(ln r_c) over 50->80 Bohr: without {s_off:+.3e}, with {s_on:+.3e} eV");

    assert!(
        s_off < 0.0,
        "the untailed slope should be negative (the divergence coefficient is `-(2π/V)(ΣQρ)²`), \
         got {s_off:+.3e}"
    );
    assert!(
        s_on.abs() < 0.25 * s_off.abs(),
        "the tail should remove most of the logarithmic slope; it went {s_off:+.3e} -> {s_on:+.3e}"
    );
}

/// The tail is a **constant** shift of `Δ`, so it must not move the forces at the order it is
/// kept to.
///
/// Its leading terms depend on `η_ab`, the cutoff and the cell volume — not on the pair separation,
/// because the expansion is in `d/|T|` with `|T| > r_c`. Stating that in a doc comment is cheap;
/// measuring it is what says the implementation agrees.
#[test]
fn the_tail_does_not_disturb_the_forces() {
    use am1_rs::pbc::pbc_gradient;
    let params = Am1Parameters::standard().unwrap();
    let cell = hf_crystal(4.5);

    let forces = |tail: bool| {
        let o = options(40.0, tail);
        let scf = run_pbc_scf(&cell, &params, &o).unwrap();
        pbc_gradient(&cell, &params, &o, &scf).unwrap().gradient
    };
    let off = forces(false);
    let on = forces(true);
    let worst = off
        .iter()
        .zip(&on)
        .map(|(a, b)| (*a - *b).norm())
        .fold(0.0_f64, f64::max);
    let scale = off.iter().fold(0.0_f64, |m, v| m.max(v.norm()));
    eprintln!("    |Δforce| with and without the tail: {worst:.3e} eV/Bohr (scale {scale:.3e})");
    // Not exactly zero: the tail shifts the Fock diagonal, so the converged *density* moves a
    // little and the Hellmann–Feynman force with it. That is a real second-order effect, not a
    // stray gradient term — and it is far below the force itself.
    assert!(
        worst < 0.02 * scale.max(1.0e-9),
        "the tail moved the forces by {worst:.3e}, which is more than a density shift explains"
    );
}

/// The tail belongs to the **response** as well as the energy, and it removes the same drift there.
///
/// # Why this test exists in this shape
///
/// The tail shifts the Fock diagonal, so a response kernel built without it is the response of a
/// different Hamiltonian than the one the SCF converged. That is not a subtle claim — it is
/// measurable as a broken identity: `D(q = 0)` and the directly-computed `q = 0` Hessian are the
/// same number, and building the tail into one but not the other put them 4.6e-4 eV/Bohr² apart
/// (`tests/pbc_dfpt.rs::at_gamma_the_long_range_term_reproduces_the_q_zero_hessian_in_3d` is what
/// holds that line).
///
/// What is measured *here* is the other half: that carrying the tail through the response also
/// removes the force constants' dependence on where the pair list was cut. The tail is varied
/// alone — both arms have the monopole correction on — which the comparison in `pbc_dfpt.rs`
/// cannot do, because `LongRange::Off` gates the response kernel while the ground state keeps the
/// tail regardless.
///
/// # What separates a fix from a coincidence
///
/// A single pair of cutoffs cannot tell "the drift got smaller" from "the drift is logarithmic and
/// I moved a constant". Two consecutive intervals can: a logarithm gives roughly equal steps per
/// equal *ratio* of cutoffs, a power law gives shrinking ones. So both intervals are measured and
/// the assertion is on the far one, where the continuum approximation the tail makes is closest to
/// true.
#[test]
fn the_tail_reduces_the_cutoff_dependence_of_the_force_constants() {
    use am1_rs::pbc::{force_constants_at_q_with, CMatrix, DfptOptions, KMesh, KPoint, LongRange};

    let params = Am1Parameters::standard().unwrap();
    let cell = hf_crystal(4.5);

    let run = |cutoff: f64, tail: bool| {
        let o = PbcOptions {
            kmesh: KMesh::MonkhorstPack([2, 2, 2]),
            fold_time_reversal: false,
            ..options(cutoff, tail)
        };
        force_constants_at_q_with(
            &cell,
            &params,
            &o,
            &DfptOptions {
                long_range: LongRange::Require,
                ..DfptOptions::default()
            },
            KPoint {
                fractional: [0.0; 3],
                weight: 1.0,
            },
        )
        .unwrap()
        .force_constants
    };
    let spread = |a: &CMatrix, b: &CMatrix| {
        let mut worst = 0.0_f64;
        for i in 0..a.n {
            for j in 0..a.n {
                let (ar, ai) = a.get(i, j);
                let (br, bi) = b.get(i, j);
                worst = worst.max((ar - br).abs()).max((ai - bi).abs());
            }
        }
        worst
    };

    let step = |tail: bool| {
        let (d18, d28, d40) = (run(18.0, tail), run(28.0, tail), run(40.0, tail));
        (spread(&d18, &d28), spread(&d28, &d40))
    };
    let (near_off, far_off) = step(false);
    let (near_on, far_on) = step(true);

    eprintln!("    |D(0) drift| over 18->28 and 28->40 Bohr, in eV/Bohr^2:");
    eprintln!("      without tail: {near_off:.4e}, {far_off:.4e}");
    eprintln!("      with tail   : {near_on:.4e}, {far_on:.4e}");
    eprintln!(
        "      far interval improved {:.1}x",
        far_off / far_on.max(1.0e-14)
    );

    // The signature, not just the size: untailed, the two intervals are comparable (a logarithm
    // over ln(28/18) = 0.44 and ln(40/28) = 0.36); tailed, the far one is much the smaller.
    assert!(
        far_off > 0.5 * near_off,
        "without the tail the drift should be roughly logarithmic — comparable steps per equal \
         ratio of cutoffs — but it went {near_off:.3e} -> {far_off:.3e}, which is not that shape"
    );
    assert!(
        far_on < 0.5 * far_off,
        "the tail should more than halve the far-interval drift of D(0); it went {far_off:.3e} \
         -> {far_on:.3e}"
    );
    assert!(
        far_on < 0.5 * near_on,
        "with the tail the drift should be falling like a power law, but the two intervals were \
         {near_on:.3e} and {far_on:.3e}"
    );
}

/// A neutral cell's divergence coefficient is `−(2π/V)(Σ_a Q_a ρ_a)²`, so it is always negative:
/// the untailed energy must *fall* as the cutoff grows. Checking the sign is a cheap, independent
/// test of the derivation — a sign slip in the tail would show up as a drift that grows.
#[test]
fn the_untailed_drift_has_the_sign_the_derivation_predicts() {
    let params = Am1Parameters::standard().unwrap();
    let cell = hf_crystal(4.5);
    let e = |rc: f64| {
        run_pbc_scf(&cell, &params, &options(rc, false))
            .unwrap()
            .total_ev
    };
    let (near, far) = (e(20.0), e(80.0));
    eprintln!("    without the tail: {near:.5} eV at 20 Bohr, {far:.5} eV at 80 Bohr");
    assert!(
        far < near,
        "the divergence coefficient is negative, so the untailed energy should fall with the \
         cutoff; it went {near:.5} -> {far:.5}"
    );
}

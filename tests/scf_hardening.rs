// SPDX-License-Identifier: GPL-3.0-or-later

//! **Why a dense crystal's SCF stalls, and what the solver does about it.**
//!
//! The frozen-phonon path runs the *molecular* SCF over a supercell, so every phonon calculation
//! on a solid goes through this solver. On a dense ionic crystal it did not converge, and the
//! error it produced said `error=NaN`, which reads as an overflow and is not one — the field was
//! a hardcoded placeholder.
//!
//! What actually happens, measured on zincblende AlP (`AM1_SCF_DEBUG=1`):
//!
//! ```text
//!   scf    2  E  -31780.39  dE 2.503e2  dP 5.434e-3  |[F,P]| 8.185e0  gap 182.97 eV  diis w 1.00
//!   scf  300  E  -31792.79  dE 3.364e-1 dP 2.554e-3  |[F,P]| 3.769e0  gap 184.07 eV  diis w 1.00
//! ```
//!
//! * the gap is **183 eV** — the occupied set cannot change identity, so this is not a band
//!   crossing;
//! * the DIIS weight is pinned at **1.00** — A-DIIS coefficients live on the simplex, so this is
//!   not a runaway extrapolation either;
//! * `|[F,P]|` plateaus at **3.8** and stays there.
//!
//! The obvious next suspect is the accelerator — A-DIIS interpolates among past Focks and cannot
//! escape a plateau, and `|[F,P]|` never falls below `adiis_switch`, so the hybrid never hands
//! over to CDIIS. [`the_accelerator_that_stalls_and_the_one_that_does_not`] tests that directly,
//! and rules it out: **plain iteration, CDIIS and the hybrid all fail identically.** Damping the
//! density does not rescue it either, down to a mixing of 0.05.
//!
//! So it is not the accelerator and not the mixing: the aufbau fixed point is not reachable for
//! this system by any of them. With `max_image_overlap = 0.20` against NDDO's assumption of zero,
//! the reading is that the model is outside its domain. `docs/pbc.md` records which route does
//! work for such a system (DFPT, on a converged k-mesh).

use am1_rs::lattice::Lattice;
use am1_rs::math::Vec3;
use am1_rs::scf::{run_am1, Am1Options, ScfAccelerator};
use am1_rs::{Am1Parameters, Atom, Molecule};

const ANG: f64 = 1.0 / 0.529167;

/// Zincblende AlP in its two-atom fcc primitive cell — a dense, gapped, entirely ordinary
/// semiconductor, and one the molecular solver could not converge.
fn alp() -> Molecule {
    let a = 5.463 * ANG;
    Molecule::new(vec![
        Atom {
            z: 13,
            position: Vec3::zero(),
        },
        Atom {
            z: 15,
            position: Vec3::new(a / 4.0, a / 4.0, a / 4.0),
        },
    ])
    .with_cell(
        Lattice::from_vectors(
            Vec3::new(0.0, a / 2.0, a / 2.0),
            Vec3::new(a / 2.0, 0.0, a / 2.0),
            Vec3::new(a / 2.0, a / 2.0, 0.0),
            [true, true, true],
        )
        .unwrap(),
    )
}

fn options(accel: ScfAccelerator) -> Am1Options {
    Am1Options {
        realspace_cutoff: 40.0,
        exchange_cutoff: Some(20.0),
        max_scf: 400,
        accelerator: accel,
        ..Am1Options::default()
    }
}

/// The measurement the module documentation quotes: which accelerator gets there.
#[test]
fn the_accelerator_that_stalls_and_the_one_that_does_not() {
    let params = Am1Parameters::standard().unwrap();
    let mol = alp();
    let mut outcomes = Vec::new();
    for (name, accel) in [
        ("none", ScfAccelerator::None),
        ("cdiis", ScfAccelerator::Cdiis),
        ("adiis->cdiis", ScfAccelerator::AdiisCdiis),
    ] {
        let r = run_am1(&mol, &params, &options(accel));
        match &r {
            Ok(res) => eprintln!(
                "    {name:14} converged in {:4} iterations, E = {:.6} eV",
                res.iterations, res.total_ev
            ),
            Err(e) => eprintln!("    {name:14} FAILED: {e}"),
        }
        outcomes.push((name, r.is_ok()));
    }
    // The point of this test is the *pattern*: no Fock-space accelerator reaches the fixed point
    // here. If one of them ever does, the diagnosis in this file's header is wrong and should be
    // revisited rather than quietly outlived.
    assert!(
        outcomes.iter().all(|(_, ok)| !ok),
        "an accelerator now converges zincblende AlP where none did. That is good news, and it \
         means the conclusion recorded in this file's header -- that the failure is the model's \
         domain rather than the solver -- needs re-deriving: {outcomes:?}"
    );
}

/// A failure has to *say* what it was.
///
/// The message used to be `error=NaN` for every non-convergence, from a hardcoded `f64::NAN` in
/// the error's `error` field — so the one number a user had to go on was a placeholder. It now
/// carries the last commutator norm, which is a measured quantity and the one the convergence
/// test is actually about.
#[test]
fn a_non_convergence_reports_the_residual_it_reached() {
    let params = Am1Parameters::standard().unwrap();
    let opts = Am1Options {
        // Far too few iterations to converge anything: the point is the message, not the physics.
        max_scf: 3,
        ..options(ScfAccelerator::AdiisCdiis)
    };
    let err = run_am1(&alp(), &params, &opts).unwrap_err();
    let text = format!("{err}");
    eprintln!("    {text}");
    assert!(
        !text.contains("NaN"),
        "the reported residual is still a placeholder: {text}"
    );
    match err {
        am1_rs::Am1Error::ScfNotConverged { error, .. } => assert!(
            error.is_finite() && error > 0.0,
            "the reported residual should be the measured one, got {error}"
        ),
        other => panic!("expected a non-convergence, got {other}"),
    }
}

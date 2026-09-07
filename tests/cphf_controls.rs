// SPDX-License-Identifier: GPL-3.0-or-later

//! The response solver's limits are settable, and they do what they say.
//!
//! `cphf_max_iter` and `cphf_tol` were private constants through 0.2.2 — 100 and `1e-9` for the
//! molecular CPHF, 200 and `1e-8` for the periodic one. A system whose response needed more passes
//! had no way to ask for them from Rust or from Python short of editing the crate. They are fields
//! on [`Am1Options`] and [`PbcOptions`] now, which is what both APIs already carry.
//!
//! Two things have to hold and they fail independently: the value has to *reach* the solver, and
//! the default has to be what it was, so that nothing silently changed accuracy when the constant
//! became a field.

use am1_rs::pbc::PbcOptions;
use am1_rs::{Am1Options, Am1Parameters, Atom, Molecule, Vec3};

const ANG: f64 = am1_rs::constants::ANGSTROM_TO_BOHR;

fn water() -> Molecule {
    Molecule::new(vec![
        Atom {
            z: 8,
            position: Vec3::zero(),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.9584 * ANG, 0.0, 0.0),
        },
        Atom {
            z: 1,
            position: Vec3::new(-0.24 * ANG, 0.9278 * ANG, 0.0),
        },
    ])
}

/// The defaults are the constants they replaced.
///
/// Turning a constant into an option is a chance to change a number by accident, and the change
/// would show up as a slightly different Hessian rather than as a failure.
#[test]
fn the_defaults_are_the_constants_they_replaced() {
    let m = Am1Options::default();
    assert_eq!(m.cphf_max_iter, 100);
    assert!((m.cphf_tol - 1.0e-9).abs() < 1.0e-20);
    let p = PbcOptions::default();
    assert_eq!(p.cphf_max_iter, 200);
    assert!((p.cphf_tol - 1.0e-8).abs() < 1.0e-20);
}

/// A limit of one iteration must fail, and say so with that number.
///
/// This is the check that the value reaches the solver at all. A field that is plumbed to nowhere
/// looks exactly like one that is plumbed correctly on a system that converges in three passes.
#[test]
fn a_tiny_iteration_cap_is_honoured_and_reported() {
    let params = Am1Parameters::standard().unwrap();
    let options = Am1Options {
        cphf_max_iter: 1,
        ..Default::default()
    };
    let err = am1_rs::hessian::analytic_hessian(&water(), &params, &options, 1.0e-3)
        .expect_err("a one-iteration CPHF cannot have converged");
    let text = err.to_string();
    assert!(
        text.contains("after 1 iterations"),
        "the error does not report the cap it was given: {text}"
    );
}

/// A loose tolerance converges where a tight one would take longer, and gives the same answer to
/// about that tolerance.
///
/// The point is that `cphf_tol` is a knob on the *path*, not on the answer: the CPHF has one
/// solution, and stopping nearer or further from it changes the Hessian by about the residual.
#[test]
fn the_tolerance_trades_iterations_for_accuracy_and_nothing_else() {
    let params = Am1Parameters::standard().unwrap();
    let molecule = water();
    let tight = am1_rs::hessian::analytic_hessian(
        &molecule,
        &params,
        &Am1Options {
            cphf_tol: 1.0e-11,
            cphf_max_iter: 400,
            ..Default::default()
        },
        1.0e-3,
    )
    .unwrap();
    let loose = am1_rs::hessian::analytic_hessian(
        &molecule,
        &params,
        &Am1Options {
            cphf_tol: 1.0e-6,
            ..Default::default()
        },
        1.0e-3,
    )
    .unwrap();
    let mut worst = 0.0_f64;
    let mut scale = 0.0_f64;
    for i in 0..tight.rows {
        for j in 0..tight.cols {
            worst = worst.max((tight[(i, j)] - loose[(i, j)]).abs());
            scale = scale.max(tight[(i, j)].abs());
        }
    }
    eprintln!("Hessian scale {scale:.3e}, tight against loose differ by {worst:.3e}");
    // Loose by five orders and the Hessian moves by less than a part in ten thousand of its own
    // scale: the tolerance is buying iterations, not answers.
    assert!(
        worst < 1.0e-4 * scale,
        "a 1e-6 tolerance changed the Hessian by {worst:e} on a scale of {scale:e}"
    );
}

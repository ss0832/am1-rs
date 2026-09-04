// SPDX-License-Identifier: GPL-3.0-or-later

//! A uniform external electric field: energy, analytic gradient, analytic Hessian.
//!
//! The field enters through the dipole operator (`src/dipole.rs`), so what has to be checked is
//! not that a number comes out but that three separately-written derivatives agree with the
//! energy they claim to differentiate. Two of the checks below are finite differences of a full
//! SCF, which is the only reference that shares no code with the analytic path.
//!
//! `Σ_a F_a = qF` is *also* asserted, but deliberately not as the main test: it follows
//! constructively from `Σ_a Q_a = q`, so it detects miswiring and nothing subtler.

use am1_rs::gradient::{closed_form_gradient, numerical_gradient};
use am1_rs::hessian::{analytic_hessian, numerical_hessian};
use am1_rs::math::Vec3;
use am1_rs::{run_am1, Am1Options, Am1Parameters, Molecule};

/// eV per (e·Bohr) — the crate's field unit. 0.005 Hartree/(e·Bohr) is a strong but tractable
/// field: large enough that the response is well above the finite-difference noise, small enough
/// that the SCF is not driven anywhere pathological.
const FIELD_STRENGTH: f64 = 0.005 * 27.21;

fn water() -> Molecule {
    Molecule::from_xyz_str(
        "3\nwater\nO 0.0000 0.0000 0.0000\nH 0.9584 0.0000 0.0000\nH -0.2400 0.9278 0.0000\n",
        0.0,
    )
    .unwrap()
}

fn options(field: Option<Vec3>, charge: f64, multiplicity: usize) -> Am1Options {
    Am1Options {
        charge,
        multiplicity,
        electric_field: field,
        // Tight, because two of these tests finite-difference the result: an SCF converged only
        // to 1e-8 leaves its own convergence error in a 1e-4 step, and it does not cancel between
        // the displaced points because they converge differently.
        e_tol: 1.0e-12,
        p_tol: 1.0e-11,
        max_scf: 500,
        ..Am1Options::default()
    }
}

/// The field with no field is the field-free calculation, bit for bit.
#[test]
fn a_zero_field_changes_nothing() {
    let params = Am1Parameters::standard().unwrap();
    let mol = water();
    let bare = run_am1(&mol, &params, &options(None, 0.0, 1)).unwrap();
    let zero = run_am1(&mol, &params, &options(Some(Vec3::zero()), 0.0, 1)).unwrap();
    assert!((bare.total_ev - zero.total_ev).abs() < 1.0e-12);
    assert_eq!(zero.field_nuclear_ev, 0.0);
}

/// `∂E/∂F = −μ`. This is the definition of the field coupling, and checking it as a finite
/// difference of the *energy* ties the SCF's reported dipole to the energy expression the field
/// term adds — the two are computed by entirely separate code.
#[test]
fn the_energy_derivative_with_respect_to_the_field_is_minus_the_dipole() {
    let params = Am1Parameters::standard().unwrap();
    let mol = water();
    let h = 1.0e-4 * 27.21;

    let zero = run_am1(&mol, &params, &options(Some(Vec3::zero()), 0.0, 1)).unwrap();
    // e·Bohr, the unit the field is conjugate to.
    let mu = zero.dipole_debye / am1_rs::constants::AU_DIPOLE_TO_DEBYE;

    for axis in 0..3 {
        let unit = match axis {
            0 => Vec3::new(1.0, 0.0, 0.0),
            1 => Vec3::new(0.0, 1.0, 0.0),
            _ => Vec3::new(0.0, 0.0, 1.0),
        };
        let plus = run_am1(&mol, &params, &options(Some(unit * h), 0.0, 1)).unwrap();
        let minus = run_am1(&mol, &params, &options(Some(unit * -h), 0.0, 1)).unwrap();
        let numeric = -(plus.total_ev - minus.total_ev) / (2.0 * h);
        let analytic = mu.get(axis);
        eprintln!("    axis {axis}: -dE/dF = {numeric:+.10}  mu = {analytic:+.10} e·Bohr");
        assert!(
            (numeric - analytic).abs() < 1.0e-6,
            "axis {axis}: -dE/dF = {numeric} but the reported dipole is {analytic}"
        );
    }
}

/// The analytic gradient under a field, against a full-SCF finite difference.
///
/// This is the test of the gradient term. It shares no code with `add_external_field_force`:
/// `numerical_gradient` re-converges the SCF at every displaced geometry, field and all.
#[test]
fn the_analytic_gradient_matches_finite_differences_under_a_field() {
    let params = Am1Parameters::standard().unwrap();
    let mol = water();
    let field = Vec3::new(0.6, -0.5, 0.62).normalized() * FIELD_STRENGTH;
    let opts = options(Some(field), 0.0, 1);

    let analytic = closed_form_gradient(&mol, &params, &opts).unwrap();
    let numeric = numerical_gradient(&mol, &params, &opts, 5.0e-4).unwrap();

    let mut worst = 0.0_f64;
    for (a, n) in analytic.gradient.iter().zip(&numeric.gradient) {
        worst = worst.max((*a - *n).norm());
    }
    eprintln!("    max |analytic - numerical| = {worst:.3e} eV/Bohr under a field");
    assert!(
        worst < 2.0e-5,
        "the analytic gradient disagrees with finite differences by {worst:.3e} eV/Bohr"
    );

    // And the field really is doing something: without it the gradient is different.
    let bare = closed_form_gradient(&mol, &params, &options(None, 0.0, 1)).unwrap();
    let mut change = 0.0_f64;
    for (a, b) in analytic.gradient.iter().zip(&bare.gradient) {
        change = change.max((*a - *b).norm());
    }
    eprintln!("    the field moves the gradient by {change:.3e} eV/Bohr");
    assert!(
        change > 1.0e-3,
        "the field barely changed the gradient ({change:.3e}); this test would pass on a no-op"
    );
}

/// The analytic (CPHF) Hessian under a field, against a finite difference of the analytic
/// gradient.
///
/// The field operator is *linear* in the nuclear positions, so it contributes nothing to the
/// skeleton second derivative and reaches the Hessian only through the CPHF response. That makes
/// this the only test that can see the response term: if it were omitted the Hessian would still
/// be symmetric, still have sensible eigenvalues, and still be wrong.
#[test]
fn the_analytic_hessian_matches_finite_differences_under_a_field() {
    let params = Am1Parameters::standard().unwrap();
    let mol = water();
    let field = Vec3::new(0.0, 0.0, 1.0) * FIELD_STRENGTH;
    let opts = options(Some(field), 0.0, 1);

    let analytic = analytic_hessian(&mol, &params, &opts, 1.0e-3).unwrap();
    let numeric = numerical_hessian(&mol, &params, &opts, 1.0e-3).unwrap();

    let mut worst = 0.0_f64;
    let mut scale = 0.0_f64;
    for i in 0..analytic.rows {
        for j in 0..analytic.cols {
            worst = worst.max((analytic[(i, j)] - numeric[(i, j)]).abs());
            scale = scale.max(analytic[(i, j)].abs());
        }
    }
    eprintln!(
        "    max |analytic - numerical| = {worst:.3e} of a Hessian whose largest element is \
         {scale:.3e}  ({:.2e} relative)",
        worst / scale
    );
    assert!(
        worst < 1.0e-5 * scale.max(1.0),
        "the analytic Hessian disagrees with finite differences by {worst:.3e} eV/Bohr^2"
    );
}

/// Translational invariance under a field: the net force is `qF`, zero for a neutral molecule.
///
/// Kept as a guard rather than as the test — it follows from `Σ_a Q_a = q` by construction, so it
/// catches a wiring mistake and nothing more subtle. The finite-difference tests above are what
/// check the physics.
#[test]
fn the_net_force_is_the_charge_times_the_field() {
    let params = Am1Parameters::standard().unwrap();
    let field = Vec3::new(0.3, 0.9, -0.31).normalized() * FIELD_STRENGTH;

    for (charge, multiplicity) in [(0.0, 1), (1.0, 2)] {
        let mol = water().with_charge(charge);
        let opts = options(Some(field), charge, multiplicity);
        let g = closed_form_gradient(&mol, &params, &opts).unwrap();
        let net: Vec3 = g.forces.iter().fold(Vec3::zero(), |a, f| a + *f);
        let expected = field * charge;
        eprintln!(
            "    charge {charge:+.1}: net force ({:+.3e}, {:+.3e}, {:+.3e}), expected \
             ({:+.3e}, {:+.3e}, {:+.3e})",
            net.x, net.y, net.z, expected.x, expected.y, expected.z
        );
        assert!(
            (net - expected).norm() < 1.0e-6,
            "net force {net:?} != qF {expected:?}"
        );
    }
}

/// A field is refused under periodic boundary conditions **when it points along a periodic
/// direction**, rather than being silently applied.
///
/// Narrowed in 0.2.2. This used to assert a blanket refusal for any cell, matching the code, and
/// the code was too strict: `F·R` shifts by `F·T` under translation, so the perturbation is
/// lattice-periodic exactly when `F·T = 0` for every lattice vector. A cubic cell is periodic in
/// all three directions, so **this** field is still an error — but the reason is now the direction
/// and the message says so. `tests/pbc_external_field.rs` covers the cases that became legal.
#[test]
fn a_periodic_cell_with_a_field_is_an_error() {
    use am1_rs::lattice::Lattice;
    let params = Am1Parameters::standard().unwrap();
    let mol = water().with_cell(Lattice::cubic(20.0).unwrap());
    let err = run_am1(
        &mol,
        &params,
        &options(Some(Vec3::new(0.0, 0.0, FIELD_STRENGTH)), 0.0, 1),
    )
    .unwrap_err();
    let text = err.to_string();
    eprintln!("    {text}");
    assert!(
        text.contains("along a periodic direction"),
        "expected a refusal naming the direction, got: {text}"
    );
}
/// The open-shell path carries the field too.
#[test]
fn the_unrestricted_gradient_matches_finite_differences_under_a_field() {
    let params = Am1Parameters::standard().unwrap();
    let mol = Molecule::from_xyz_str(
        "4\nmethyl\nC 0.0 0.0 0.0\nH 1.079 0.0 0.0\nH -0.5395 0.9344 0.0\nH -0.5395 -0.9344 0.0\n",
        0.0,
    )
    .unwrap();
    let field = Vec3::new(0.0, 0.0, 1.0) * FIELD_STRENGTH;
    let opts = options(Some(field), 0.0, 2);

    let analytic = closed_form_gradient(&mol, &params, &opts).unwrap();
    assert!(analytic.scf.unrestricted, "expected the UHF path");
    let numeric = numerical_gradient(&mol, &params, &opts, 5.0e-4).unwrap();
    let mut worst = 0.0_f64;
    for (a, n) in analytic.gradient.iter().zip(&numeric.gradient) {
        worst = worst.max((*a - *n).norm());
    }
    eprintln!("    UHF: max |analytic - numerical| = {worst:.3e} eV/Bohr under a field");
    assert!(worst < 5.0e-5, "UHF field gradient off by {worst:.3e}");
}

// SPDX-License-Identifier: GPL-3.0-or-later

//! A uniform external electric field under periodic boundary conditions.
//!
//! # What was refused, and what was actually wrong with it
//!
//! Through 0.2.1 a field plus a cell was an error, on the grounds that "`F·R` is unbounded along a
//! periodic direction, so it is not a lattice-periodic perturbation". That reason is correct and
//! the refusal was too broad: `F·R` shifts by `F·T` under translation by `T`, so the perturbation
//! repeats with the lattice exactly when **`F·T = 0` for every lattice vector**. A slab in a field
//! along its normal, and a chain in a transverse field, satisfy that and are ordinary
//! calculations. Only the component *along* a periodic direction is ill-defined.
//!
//! So the check is now on the direction, not on the presence of a cell, and it names the offending
//! component when it fires.
//!
//! # What is asserted
//!
//! * The allowed direction gives an energy, a force and a dipole response that agree with finite
//!   differences -- the same checks the molecular field gets.
//! * The forbidden direction is still an error, and the error names the component.
//! * A field along a non-periodic axis of a **large** cell reproduces the isolated molecule's
//!   answer. That is the one that says the periodic machinery has not quietly changed what the
//!   field means: the same molecule, the same field, two code paths, one number.

use am1_rs::lattice::Lattice;
use am1_rs::math::Vec3;
use am1_rs::pbc::{pbc_energy_and_gradient, run_pbc_scf, KMesh, PbcOptions};
use am1_rs::scf::{run_am1, Am1Options};
use am1_rs::{Am1Parameters, Atom, Molecule};

const ANG: f64 = 1.0 / 0.529167;

fn water_atoms() -> Vec<Atom> {
    [
        (8u8, [0.0, 0.0, 0.0]),
        (1, [0.9614, 0.0, 0.0]),
        (1, [-0.2246, 0.9348, 0.0]),
    ]
    .iter()
    .map(|(z, r)| Atom {
        z: *z,
        position: Vec3::new(r[0], r[1], r[2]) * ANG,
    })
    .collect()
}

/// A chain along `x`, so `y` and `z` are free directions a field may point along.
fn water_chain(n: usize, spacing_ang: f64) -> Molecule {
    let step = spacing_ang * ANG;
    let mut atoms = Vec::new();
    for cell in 0..n {
        for a in water_atoms() {
            atoms.push(Atom {
                z: a.z,
                position: a.position + Vec3::new(step * cell as f64, 0.0, 0.0),
            });
        }
    }
    Molecule::new(atoms).with_cell(
        Lattice::from_vectors(
            Vec3::new(step * n as f64, 0.0, 0.0),
            Vec3::new(0.0, 30.0, 0.0),
            Vec3::new(0.0, 0.0, 30.0),
            [true, false, false],
        )
        .unwrap(),
    )
}

fn options(field: Option<Vec3>) -> PbcOptions {
    PbcOptions {
        kmesh: KMesh::MonkhorstPack([3, 1, 1]),
        fold_time_reversal: false,
        realspace_cutoff: 30.0,
        exchange_cutoff: Some(12.0),
        smearing_ev: 0.0,
        e_tol: 1.0e-11,
        p_tol: 1.0e-10,
        max_scf: 800,
        mixing: 0.3,
        electric_field: field,
        ..PbcOptions::default()
    }
}

/// A field **along** the chain is still refused, and the message names the component.
///
/// Asserting the text, not just that it fails: the whole point of the change is that the two cases
/// are now distinguished, and an error that said "fields are not supported under a cell" would be
/// indistinguishable from the old blanket refusal.
#[test]
fn a_field_along_a_periodic_direction_is_refused_by_name() {
    let params = Am1Parameters::standard().unwrap();
    let cell = water_chain(2, 3.2);
    let err = run_pbc_scf(&cell, &params, &options(Some(Vec3::new(0.002, 0.0, 0.0))))
        .unwrap_err()
        .to_string();
    eprintln!("    {err}");
    assert!(
        err.contains("along a periodic direction"),
        "the error should say which direction is the problem: {err}"
    );
    assert!(
        err.contains("0.0020"),
        "the error should name the offending component: {err}"
    );
}

/// A field with **both** an allowed and a forbidden component is refused: the forbidden part does
/// not become acceptable by being accompanied.
#[test]
fn a_mixed_field_is_refused_for_its_periodic_part() {
    let params = Am1Parameters::standard().unwrap();
    let cell = water_chain(2, 3.2);
    let err = run_pbc_scf(&cell, &params, &options(Some(Vec3::new(0.001, 0.002, 0.0))))
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("along a periodic direction"),
        "a field with a periodic component must be refused: {err}"
    );
}

/// A transverse field is accepted, and moves the energy the way a field should.
#[test]
fn a_transverse_field_lowers_the_energy_and_polarizes_the_cell() {
    let params = Am1Parameters::standard().unwrap();
    let cell = water_chain(2, 3.2);

    let zero = run_pbc_scf(&cell, &params, &options(None)).unwrap();
    let f = 0.004;
    let plus = run_pbc_scf(&cell, &params, &options(Some(Vec3::new(0.0, f, 0.0)))).unwrap();
    let minus = run_pbc_scf(&cell, &params, &options(Some(Vec3::new(0.0, -f, 0.0)))).unwrap();

    eprintln!(
        "    E(0) = {:.9},  E(+F) = {:.9},  E(-F) = {:.9} eV",
        zero.total_ev, plus.total_ev, minus.total_ev
    );
    // `E(F) = E0 - mu*F - (1/2) alpha F^2 + ...`. It is the **average** of the two signs that has
    // to sit below `E(0)`, not each of them: this chain has a dipole along `y`, so the linear term
    // dominates at this field strength and one sign necessarily rises. What the average isolates
    // is the quadratic term, whose coefficient is the polarizability and is positive for any
    // stable system -- so a sign error in the field Hamiltonian would show up here as a *raised*
    // average, which no physical system produces.
    let mean = 0.5 * (plus.total_ev + minus.total_ev);
    let alpha = -2.0 * (mean - zero.total_ev) / (f * f);
    eprintln!(
        "    mean(E(+F), E(-F)) - E(0) = {:.3e} eV,  alpha_yy = {alpha:.4}",
        mean - zero.total_ev
    );
    assert!(
        mean < zero.total_ev,
        "the second-order response must lower the energy: mean {mean:.9} against {:.9}",
        zero.total_ev
    );
    assert!(
        alpha > 0.0,
        "the polarizability came out negative ({alpha:.4}), which no stable system has"
    );

    // `-dE/dF` is the cell dipole along y, and the central difference gives it directly.
    let mu_y = -(plus.total_ev - minus.total_ev) / (2.0 * f);
    // The same quantity from the converged charges, which is a different route through the code.
    let mut from_charges = 0.0;
    for (atom, q) in cell.atoms.iter().zip(&zero.charges) {
        from_charges += atom.position.y * q;
    }
    eprintln!("    mu_y: -dE/dF = {mu_y:.6} e*Bohr, from the point charges {from_charges:.6}");
    // Not equal: the finite difference sees the *relaxed* dipole, which includes the electronic
    // polarization, while the point-charge sum is the unrelaxed monopole part and misses the
    // on-site sp hybridization moment entirely. Same sign and same order is the claim.
    assert!(
        mu_y * from_charges > 0.0,
        "the two routes to the dipole disagree in sign: {mu_y:.6} against {from_charges:.6}"
    );
}

/// The field's force matches a finite difference of the field's own energy.
///
/// The field enters `H_core`, so the energy carries it automatically; the **force** needs its
/// nuclear half added by hand, and forgetting that is invisible at zero field and silent at finite
/// field -- the dynamics simply stops conserving.
#[test]
fn the_field_force_matches_a_finite_difference() {
    let params = Am1Parameters::standard().unwrap();
    let cell = water_chain(2, 3.2);
    let o = options(Some(Vec3::new(0.0, 0.003, 0.0)));

    let (_, g) = pbc_energy_and_gradient(&cell, &params, &o).unwrap();

    let h = 1.0e-4;
    let mut worst = 0.0_f64;
    let mut scale = 0.0_f64;
    for atom in 0..cell.atoms.len() {
        for axis in 0..3 {
            let shifted = |d: f64| {
                let mut m = cell.clone();
                let p = &mut m.atoms[atom].position;
                match axis {
                    0 => p.x += d,
                    1 => p.y += d,
                    _ => p.z += d,
                }
                run_pbc_scf(&m, &params, &o).unwrap().total_ev
            };
            let fd = (shifted(h) - shifted(-h)) / (2.0 * h);
            let analytic = match axis {
                0 => g.gradient[atom].x,
                1 => g.gradient[atom].y,
                _ => g.gradient[atom].z,
            };
            worst = worst.max((analytic - fd).abs());
            scale = scale.max(analytic.abs());
        }
    }
    eprintln!(
        "    in a transverse field: max |analytic - finite difference| = {worst:.3e} eV/Bohr \
         (scale {scale:.3e})"
    );
    assert!(
        scale > 0.01,
        "the forces are only {scale:.3e}, so this comparison shows nothing"
    );
    assert!(
        worst < 1.0e-5,
        "the periodic gradient in a field is off by {worst:.3e} eV/Bohr"
    );
}

/// A molecule alone in a large cell, in a field along a non-periodic axis, must give the isolated
/// molecule's answer.
///
/// This is the check that says the periodic path has not quietly changed what the field *means* --
/// the sign convention, the units, whether the nuclear half is counted. Two code paths that share
/// only `crate::dipole`, one number.
#[test]
fn a_field_on_an_isolated_molecule_agrees_between_the_two_paths() {
    let params = Am1Parameters::standard().unwrap();
    let field = Vec3::new(0.0, 0.0, 0.005);

    let molecular = run_am1(
        &Molecule::new(water_atoms()),
        &params,
        &Am1Options {
            electric_field: Some(field),
            e_tol: 1.0e-11,
            p_tol: 1.0e-10,
            ..Am1Options::default()
        },
    )
    .unwrap();

    // Periodic along `x` only, with a long repeat, and the field along `z` -- a direction the
    // lattice does not touch.
    let boxed = Molecule::new(water_atoms()).with_cell(
        Lattice::from_vectors(
            Vec3::new(60.0, 0.0, 0.0),
            Vec3::new(0.0, 60.0, 0.0),
            Vec3::new(0.0, 0.0, 60.0),
            [true, false, false],
        )
        .unwrap(),
    );
    let periodic = run_pbc_scf(
        &boxed,
        &params,
        &PbcOptions {
            kmesh: KMesh::Gamma,
            realspace_cutoff: 55.0,
            exchange_cutoff: Some(30.0),
            smearing_ev: 0.0,
            e_tol: 1.0e-11,
            p_tol: 1.0e-10,
            max_scf: 800,
            electric_field: Some(field),
            ..PbcOptions::default()
        },
    )
    .unwrap();

    let diff = (molecular.total_ev - periodic.total_ev).abs();
    eprintln!(
        "    molecular {:.9} eV, periodic (60 Bohr chain) {:.9} eV, difference {diff:.3e}",
        molecular.total_ev, periodic.total_ev
    );
    // Not exactly equal: the periodic cell still carries the interaction with its own images
    // along `x`, which is what the 60 Bohr repeat makes small rather than zero.
    assert!(
        diff < 1.0e-3,
        "the two paths disagree about the same molecule in the same field by {diff:.3e} eV"
    );
}

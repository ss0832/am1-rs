// SPDX-License-Identifier: GPL-3.0-or-later

//! A finite electric field **along** a periodic direction, by the Berry-phase electric enthalpy.
//!
//! # What this is for
//!
//! `F.R` is unbounded along a periodic direction, so a field there is not a perturbation the Bloch
//! construction can represent. What replaces it is the electric enthalpy `E - Omega E.P` with `P`
//! the Berry-phase polarization -- Nunes and Gonze's construction. A field *orthogonal* to every
//! lattice vector needs none of this and goes through `PbcOptions::electric_field`; that is
//! `tests/pbc_external_field.rs`.
//!
//! # The check that matters, and where it applies
//!
//! **`alpha = Omega dP/dE` by finite differences of this must equal the CPHF polarizability.**
//! Every other property here -- that the phase moves, that the enthalpy falls, that zero field
//! changes nothing -- is satisfied by a version whose coupling constant is off by a factor, which
//! is exactly the failure mode a hand-derived Berry-phase prefactor invites. It caught one: the
//! first draft symmetrized the field operator as `(M + M')/2`, which halves the occupied-virtual
//! coupling, and gave 0.56 of the CPHF value.
//!
//! The magnitude check runs first on a **hydrogen-only** cell, then on a p-block one. That order
//! is historical and worth keeping: until the Berry link operator carried the on-site `s`-`p`
//! moment it tracked only the charge *centres*, so the two formalisms agreed to 0.03 % where no
//! atom has a p orbital and disagreed by 12 % where one does. Fixing the link operator closed
//! that -- both now agree to better than 0.5 % -- and the two cells together are what says so.

use am1_rs::lattice::Lattice;
use am1_rs::math::Vec3;
use am1_rs::pbc::{dielectric_tensor, run_finite_field, FiniteFieldOptions, KMesh, PbcOptions};
use am1_rs::{Am1Parameters, Atom, Molecule};

const ANG: f64 = 1.0 / 0.529167;

/// Two hydrogens per cubic cell: **s orbitals only**, so the dipole operator has no on-site
/// `s`-`p` moment and the Berry phase and the CPHF are computing the same object.
///
/// Off-axis and off-centre on purpose, so no component of the polarizability vanishes by symmetry
/// and every axis is a real comparison.
fn h2_crystal(a_ang: f64) -> Molecule {
    let a = a_ang * ANG;
    Molecule::new(vec![
        Atom {
            z: 1,
            position: Vec3::zero(),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.80, 0.15, 0.05) * ANG,
        },
    ])
    .with_cell(Lattice::cubic(a).unwrap())
}

/// One water per cubic cell: fully periodic, polar, and carrying p orbitals on the oxygen.
fn water_crystal(a_ang: f64) -> Molecule {
    let a = a_ang * ANG;
    let atoms: Vec<Atom> = [
        (8u8, [0.0, 0.0, 0.0]),
        (1, [0.9614, 0.0, 0.0]),
        (1, [-0.2246, 0.9348, 0.0]),
    ]
    .iter()
    .map(|(z, r)| Atom {
        z: *z,
        position: Vec3::new(r[0], r[1], r[2]) * ANG,
    })
    .collect();
    Molecule::new(atoms).with_cell(Lattice::cubic(a).unwrap())
}

fn options(mesh: usize) -> PbcOptions {
    PbcOptions {
        kmesh: KMesh::MonkhorstPack([mesh, mesh, mesh]),
        fold_time_reversal: false,
        realspace_cutoff: 30.0,
        exchange_cutoff: Some(12.0),
        smearing_ev: 0.0,
        e_tol: 1.0e-12,
        p_tol: 1.0e-11,
        max_scf: 800,
        mixing: 0.3,
        ..PbcOptions::default()
    }
}

/// `Omega dP_alpha/dE_alpha` by central differences of the finite-field polarization, in Bohr^3.
fn finite_field_alpha(
    molecule: &Molecule,
    params: &Am1Parameters,
    opts: &PbcOptions,
    axis: usize,
    h: f64,
) -> f64 {
    use am1_rs::constants::HARTREE_TO_EV;
    let ff = FiniteFieldOptions::default();
    let volume = molecule.cell.unwrap().volume();
    let p = |sign: f64| -> f64 {
        let mut f = Vec3::zero();
        match axis {
            0 => f.x = sign * h,
            1 => f.y = sign * h,
            _ => f.z = sign * h,
        }
        let r = run_finite_field(molecule, params, opts, f, &ff).unwrap();
        let e = r.electronic_polarization;
        match axis {
            0 => e.x,
            1 => e.y,
            _ => e.z,
        }
    };
    // The **electronic** polarization: the ionic half does not respond to a clamped-ion field, and
    // leaving it out keeps the polarization quantum's branch out of the difference too.
    volume * (p(1.0) - p(-1.0)) / (2.0 * h) * HARTREE_TO_EV
}

/// **The magnitude check**, on the system where the two formalisms compute the same object.
#[test]
fn the_finite_field_polarizability_matches_the_cphf_one() {
    let params = Am1Parameters::standard().unwrap();
    let cell = h2_crystal(5.0);
    let opts = options(6);

    let cphf = dielectric_tensor(&cell, &params, &opts).unwrap().0;
    // This cell is nearly linear along `x`, so its transverse polarizabilities are one and two
    // orders of magnitude smaller. They are still good comparisons -- 0.04 % and 0.19 % -- but a
    // purely relative tolerance would judge them against their own size rather than against the
    // tensor's, so the error is measured against the largest component with each axis' own value
    // as a floor.
    let scale = (0..3).map(|a| cphf[a][a].abs()).fold(0.0_f64, f64::max);
    assert!(
        scale > 1.0,
        "the CPHF polarizability is only {scale:.4} Bohr^3, so this comparison shows nothing"
    );

    let h = 1.0e-4;
    let mut worst = 0.0_f64;
    #[allow(clippy::needless_range_loop)]
    for axis in 0..3 {
        let ff = finite_field_alpha(&cell, &params, &opts, axis, h);
        let reference = cphf[axis][axis];
        let rel = (ff - reference).abs() / reference.abs().max(0.01 * scale);
        eprintln!(
            "    axis {axis}: finite field {ff:9.5} Bohr^3, CPHF {reference:9.5} Bohr^3 \
             ({:.3} % apart)",
            100.0 * (ff - reference).abs() / reference.abs()
        );
        worst = worst.max(rel);
    }
    // What is left at 6 points per string is the Berry phase's own discretization, and the next
    // test measures that it is by watching it fall. A wrong coupling constant would sit at tens of
    // percent -- the draft that halved the field operator sat at 44.
    assert!(
        worst < 0.01,
        "the finite-field and CPHF polarizabilities differ by {:.2} %, which is more than the \
         Berry phase's discretization explains at this string length",
        100.0 * worst
    );
}
/// The residual against the CPHF must **fall as the string lengthens**, and fall like `1/J^2`.
///
/// That is what separates "the constant is right and the phase is discretized" from "the constant
/// is a little wrong". A wrong constant does not converge away.
#[test]
fn the_string_discretization_converges_away() {
    let params = Am1Parameters::standard().unwrap();
    let cell = h2_crystal(5.0);
    let h = 1.0e-4;

    let reference = dielectric_tensor(&cell, &params, &options(6)).unwrap().0[0][0];
    let mut errors = Vec::new();
    for mesh in [4usize, 6, 8] {
        let ff = finite_field_alpha(&cell, &params, &options(mesh), 0, h);
        let err = (ff - reference).abs() / reference.abs();
        eprintln!(
            "    J = {mesh}: alpha_xx = {ff:9.5} Bohr^3, {:.4} % from CPHF",
            100.0 * err
        );
        errors.push(err);
    }
    assert!(
        errors[2] < errors[0],
        "the error did not fall with the string length: {errors:?}"
    );
    // `1/J^2` from 4 to 8 is a factor of four; anything above half is not a discretization.
    assert!(
        errors[2] < 0.5 * errors[0],
        "the error fell only from {:.4} % to {:.4} %, which is not the O(1/J^2) a discretization \
         gives",
        100.0 * errors[0],
        100.0 * errors[2]
    );
}

/// The same agreement on a **p-block** cell, where the on-site `s`-`p` moment is in play.
///
/// This is the test that changed. Until the link operator carried the on-site moment, the Berry
/// phase tracked only the charge *centres*: it disagreed with the CPHF by 12 % on this cell, and
/// this file asserted that gap as a property. It is gone -- the two now agree to 0.05 % -- and
/// what is asserted is the agreement.
///
/// The correction was one line of physics in a matrix that had been a diagonal phase: the exact
/// link element is `⟨χ_μ| e^{-i b·r} |χ_ν⟩`, and expanding `r = τ_a + (r - τ_a)` leaves
/// `e^{-i b·τ_a}` times the on-site dipole rotation, whose generator is exactly the `dd` the CPHF
/// dipole operator already used. Both now read it from the same parameter.
#[test]
fn a_p_block_cell_agrees_too_once_the_on_site_moment_is_carried() {
    let params = Am1Parameters::standard().unwrap();
    let water = water_crystal(6.5);
    let cphf = dielectric_tensor(&water, &params, &options(6)).unwrap().0;
    let h = 1.0e-4;

    let ratios: Vec<f64> = [4usize, 6]
        .iter()
        .map(|&mesh| finite_field_alpha(&water, &params, &options(mesh), 0, h) / cphf[0][0])
        .collect();
    eprintln!(
        "    water alpha_xx, finite field / CPHF: J=4 {:.4}, J=6 {:.4}",
        ratios[0], ratios[1]
    );
    assert!(
        (ratios[1] - 1.0).abs() < 0.005,
        "the finite-field and CPHF polarizabilities differ by {:.2} % on a p-block cell",
        100.0 * (ratios[1] - 1.0).abs()
    );
    // And still converging toward 1 rather than sitting somewhere near it.
    assert!(
        (ratios[1] - 1.0).abs() < (ratios[0] - 1.0).abs(),
        "the residual did not shrink with the string length: {ratios:?}"
    );
}

/// A **planar** cell's out-of-plane response, which is entirely the on-site moment.
///
/// Water lies in the `xy` plane, so every `tau_z` is zero and the mirror `z -> -z` is a symmetry of
/// the Hamiltonian at every k. With a link operator that was only `e^{-i b·tau}`, that mirror made
/// the occupied bands parity eigenstates, the link overlaps block-diagonal in parity, and the
/// field operator unable to mix them -- so `alpha_zz` came out **exactly zero** while the CPHF
/// gave 0.2556.
///
/// The on-site moment is what couples `s` to `p_z` on the oxygen, so it is the whole of this
/// component. It is therefore the sharpest single check that the moment is present and correctly
/// signed: the answer moves from exactly zero to the right number, and a wrong sign would move it
/// to the wrong one rather than merely scaling it.
#[test]
fn a_planar_cell_has_the_right_out_of_plane_response() {
    let params = Am1Parameters::standard().unwrap();
    let water = water_crystal(6.5);
    let ff = finite_field_alpha(&water, &params, &options(6), 2, 1.0e-4);
    let cphf = dielectric_tensor(&water, &params, &options(6)).unwrap().0[2][2];
    eprintln!(
        "    planar water: finite field alpha_zz = {ff:.5}, CPHF {cphf:.5} Bohr^3 ({:.2} % apart)",
        100.0 * (ff - cphf).abs() / cphf.abs()
    );
    assert!(
        cphf.abs() > 0.05,
        "the CPHF out-of-plane response is {cphf:.3e}, too small for this to say anything"
    );
    assert!(
        (ff - cphf).abs() < 0.01 * cphf.abs(),
        "the out-of-plane response is {ff:.5} against the CPHF's {cphf:.5}; it is entirely the \
         on-site moment, so this is that moment being wrong rather than a discretization"
    );
}

/// The field must polarize the cell, and lower the enthalpy for **both** signs.
///
/// The enthalpy is what is minimized, so it falls whichever way the field points -- unlike the
/// energy, which rises for one sign when the cell already has a dipole. Checking both is what
/// separates the two quantities.
#[test]
fn the_field_polarizes_the_cell_and_lowers_the_enthalpy() {
    let params = Am1Parameters::standard().unwrap();
    let cell = h2_crystal(5.0);
    let opts = options(6);
    let ff = FiniteFieldOptions::default();

    let zero = run_finite_field(&cell, &params, &opts, Vec3::zero(), &ff).unwrap();
    let f = 2.0e-3;
    let plus = run_finite_field(&cell, &params, &opts, Vec3::new(f, 0.0, 0.0), &ff).unwrap();
    let minus = run_finite_field(&cell, &params, &opts, Vec3::new(-f, 0.0, 0.0), &ff).unwrap();

    eprintln!(
        "    P_x: {:+.6e} / {:+.6e} / {:+.6e} e/Bohr^2  (-F, 0, +F)",
        minus.electronic_polarization.x,
        zero.electronic_polarization.x,
        plus.electronic_polarization.x
    );
    eprintln!(
        "    enthalpy {:.9} / {:.9} / {:.9} eV,  energy {:.9} / {:.9} / {:.9}",
        minus.enthalpy_ev,
        zero.enthalpy_ev,
        plus.enthalpy_ev,
        minus.scf.total_ev,
        zero.scf.total_ev,
        plus.scf.total_ev
    );

    let d_plus = plus.electronic_polarization.x - zero.electronic_polarization.x;
    let d_minus = minus.electronic_polarization.x - zero.electronic_polarization.x;
    assert!(
        d_plus * d_minus < 0.0,
        "the polarization moved the same way for both field signs: {d_plus:+.3e} and {d_minus:+.3e}"
    );
    assert!(
        plus.enthalpy_ev < zero.enthalpy_ev && minus.enthalpy_ev < zero.enthalpy_ev,
        "the enthalpy did not fall in the field: {:.9} / {:.9} against {:.9}",
        plus.enthalpy_ev,
        minus.enthalpy_ev,
        zero.enthalpy_ev
    );
}

/// Zero field must reproduce the ordinary SCF exactly, and the phase `pbc::berry` reports.
///
/// The finite-field path adds an operator that is identically zero at zero field, so anything it
/// changes there is a bug in the machinery rather than in the physics.
#[test]
fn zero_field_reproduces_the_plain_scf_and_the_berry_phase() {
    use am1_rs::pbc::{berry::berry_polarization, run_pbc_scf};
    let params = Am1Parameters::standard().unwrap();
    let cell = water_crystal(6.5);
    let opts = options(6);

    let plain = run_pbc_scf(&cell, &params, &opts).unwrap();
    let ff = run_finite_field(
        &cell,
        &params,
        &opts,
        Vec3::zero(),
        &FiniteFieldOptions::default(),
    )
    .unwrap();
    let de = (plain.total_ev - ff.scf.total_ev).abs();
    eprintln!("    zero field: |E(plain) - E(finite field)| = {de:.3e} eV");
    assert!(de < 1.0e-9, "zero field changed the energy by {de:.3e} eV");

    let berry = berry_polarization(&cell, &params, &opts, 6).unwrap();
    let mut worst = 0.0_f64;
    for alpha in 0..3 {
        worst = worst.max((berry.phase[alpha] - ff.phase[alpha]).abs());
    }
    eprintln!("    phases agree to {worst:.3e} turns");
    assert!(
        worst < 1.0e-9,
        "the two Berry phases disagree by {worst:.3e} turns"
    );
}

/// The two field routes are not allowed to overlap silently.
#[test]
fn the_two_field_routes_do_not_overlap() {
    let params = Am1Parameters::standard().unwrap();
    let cell = h2_crystal(5.0);
    let err = run_finite_field(
        &cell,
        &params,
        &PbcOptions {
            electric_field: Some(Vec3::new(0.0, 0.0, 1.0e-3)),
            ..options(6)
        },
        Vec3::new(0.0, 0.0, 1.0e-3),
        &FiniteFieldOptions::default(),
    )
    .unwrap_err()
    .to_string();
    eprintln!("    {err}");
    assert!(
        err.contains("two treatments of the same perturbation"),
        "expected a refusal naming the overlap: {err}"
    );
}

/// A mesh too coarse to carry a Berry phase is refused, naming the direction and the length.
#[test]
fn a_mesh_too_coarse_for_a_string_is_refused() {
    let params = Am1Parameters::standard().unwrap();
    let cell = h2_crystal(5.0);
    let err = run_finite_field(
        &cell,
        &params,
        &PbcOptions {
            kmesh: KMesh::MonkhorstPack([2, 2, 2]),
            ..options(2)
        },
        Vec3::new(0.0, 0.0, 1.0e-3),
        &FiniteFieldOptions::default(),
    )
    .unwrap_err()
    .to_string();
    eprintln!("    {err}");
    assert!(
        err.contains("at least 3 k points"),
        "expected a refusal naming the string length: {err}"
    );
}

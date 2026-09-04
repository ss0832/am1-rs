// SPDX-License-Identifier: GPL-3.0-or-later

//! Γ-point periodic Hessian, against a finite difference of the periodic analytic gradient.
//!
//! There is no separate periodic Hessian implementation, and that is the point worth testing.
//! The Bloch phase `e^{ik·T}` is 1 at `k = 0`, so `P(0,T) = P(Γ)` for *every* translation: the
//! density multiplying each image pair's integrals is the same matrix the molecular code already
//! holds, and the Γ Hessian is the molecular assembly run over the image pair list. Exactly the
//! same argument the Γ energy and gradient already rest on.
//!
//! What that leaves to check is that every piece which *is* periodic-specific was carried
//! through to second order:
//!
//! * the image pairs, including an atom paired with **its own image** — which must contribute
//!   nothing, since moving the atom moves the image with it;
//! * the exchange taper, whose own first *and second* derivatives enter through the product
//!   rule, and dropping either leaves a matrix that is not the second derivative of the energy
//!   being reported;
//! * the CPHF response, whose right-hand side has to be built from the same pair list and the
//!   same taper as the energy.
//!
//! The reference is a central difference of `pbc_gradient`, which shares no code with the
//! second-derivative path: it uses `Dual` where this uses `Dual2`, and re-converges the SCF at
//! every displaced geometry.

use am1_rs::lattice::Lattice;
use am1_rs::math::Vec3;
use am1_rs::pbc::{pbc_energy_and_gradient, KMesh, PbcOptions};
use am1_rs::{analytic_hessian, Am1Options, Am1Parameters, Atom, Matrix, Molecule};

const ANG: f64 = 1.0 / 0.529167;

/// A chain of H2 units along x, distorted so nothing vanishes by symmetry.
fn distorted_chain(a: f64) -> Molecule {
    Molecule::new(vec![
        Atom {
            z: 1,
            position: Vec3::new(0.0, 0.05 * ANG, 0.0),
        },
        Atom {
            z: 1,
            position: Vec3::new(1.35 * ANG, -0.03 * ANG, 0.02 * ANG),
        },
    ])
    .with_cell(
        Lattice::from_vectors(
            Vec3::new(a, 0.0, 0.0),
            Vec3::new(0.0, 55.0, 0.0),
            Vec3::new(0.0, 0.0, 55.0),
            [true, false, false],
        )
        .unwrap(),
    )
}

/// Water in a cubic cell, tilted so the force constants are generic.
fn water_crystal(a: f64) -> Molecule {
    Molecule::new(vec![
        Atom {
            z: 8,
            position: Vec3::new(0.1, 0.05, -0.02),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.9614, 0.0, 0.0) * ANG,
        },
        Atom {
            z: 1,
            position: Vec3::new(-0.2246, 0.9348, 0.05) * ANG,
        },
    ])
    .with_cell(Lattice::cubic(a).unwrap())
}

/// Matching molecular and periodic option sets, so the two paths differ only in the pair list.
fn options(realspace: f64, exchange: f64) -> (Am1Options, PbcOptions) {
    (
        Am1Options {
            realspace_cutoff: realspace,
            exchange_cutoff: Some(exchange),
            e_tol: 1.0e-11,
            p_tol: 1.0e-10,
            max_scf: 800,
            ..Am1Options::default()
        },
        PbcOptions {
            kmesh: KMesh::Gamma,
            fold_time_reversal: false,
            realspace_cutoff: realspace,
            exchange_cutoff: Some(exchange),
            smearing_ev: 0.0,
            e_tol: 1.0e-11,
            p_tol: 1.0e-10,
            max_scf: 800,
            mixing: 0.3,
            ..PbcOptions::default()
        },
    )
}

/// Central difference of the periodic analytic gradient, column by column.
fn finite_difference_hessian(
    molecule: &Molecule,
    params: &Am1Parameters,
    pbc: &PbcOptions,
    step: f64,
) -> Matrix {
    let nat = molecule.atoms.len();
    let ndof = 3 * nat;
    let mut h = Matrix::zeros(ndof, ndof);
    for atom in 0..nat {
        for axis in 0..3 {
            let shifted = |d: f64| {
                let mut m = molecule.clone();
                let p = &mut m.atoms[atom].position;
                match axis {
                    0 => p.x += d,
                    1 => p.y += d,
                    _ => p.z += d,
                }
                m
            };
            let (_, gp) = pbc_energy_and_gradient(&shifted(step), params, pbc).unwrap();
            let (_, gm) = pbc_energy_and_gradient(&shifted(-step), params, pbc).unwrap();
            for other in 0..nat {
                for k in 0..3 {
                    let component = |v: &Vec3| match k {
                        0 => v.x,
                        1 => v.y,
                        _ => v.z,
                    };
                    h[(3 * other + k, 3 * atom + axis)] = (component(&gp.gradient[other])
                        - component(&gm.gradient[other]))
                        / (2.0 * step);
                }
            }
        }
    }
    h
}

fn check(molecule: &Molecule, realspace: f64, exchange: f64, tol: f64, label: &str) {
    let params = Am1Parameters::standard().unwrap();
    let (scf_opts, pbc_opts) = options(realspace, exchange);

    let analytic = analytic_hessian(molecule, &params, &scf_opts, 1.0e-3).unwrap();
    // `h = 1e-4`: a central difference of a gradient has a roundoff floor around `eps·|g|/h`, and
    // a smaller step would be measuring arithmetic instead of the second derivative.
    let numeric = finite_difference_hessian(molecule, &params, &pbc_opts, 1.0e-4);

    let mut worst = 0.0_f64;
    let mut worst_at = (0, 0);
    for i in 0..analytic.rows {
        for j in 0..analytic.cols {
            let d = (analytic[(i, j)] - numeric[(i, j)]).abs();
            if d > worst {
                worst = d;
                worst_at = (i, j);
            }
        }
    }
    eprintln!(
        "    {label}: max |analytic − finite difference| = {worst:.3e} eV/Bohr² at {worst_at:?}"
    );
    assert!(
        worst < tol,
        "{label}: periodic Hessian off by {worst:.3e} eV/Bohr²"
    );
}

#[test]
fn the_hessian_of_a_chain_matches_finite_differences_at_gamma() {
    check(&distorted_chain(6.0), 40.0, 12.0, 2.0e-5, "H2 chain, Gamma");
}

#[test]
fn the_hessian_of_a_water_crystal_matches_finite_differences_at_gamma() {
    check(
        &water_crystal(14.0),
        40.0,
        12.0,
        5.0e-5,
        "water crystal, Gamma",
    );
}

#[test]
fn the_hessian_with_the_exchange_taper_fully_engaged_matches_finite_differences() {
    // A tight cell with a cutoff comparable to it, so a large fraction of the image pairs sit
    // inside the taper's transition region. This is where `taper''` matters: a Hessian that
    // carried only `taper` and `taper'` would pass the looser cases above and fail here.
    check(
        &water_crystal(11.0),
        40.0,
        9.0,
        5.0e-5,
        "dense water crystal, taper active",
    );
}

#[test]
fn a_rigid_translation_costs_nothing() {
    // The acoustic sum rule. Independent of any finite difference: translating every atom by the
    // same vector cannot change a periodic energy, so every 3x3 row-sum of the force constants
    // must vanish. This is also what catches a self-image pair contributing when it should not —
    // `(a, a+T)` moves rigidly with `a`, so it must add nothing, and if it did the row sums
    // would not cancel.
    let params = Am1Parameters::standard().unwrap();
    let (scf_opts, _) = options(40.0, 12.0);
    let molecule = water_crystal(14.0);
    let h = analytic_hessian(&molecule, &params, &scf_opts, 1.0e-3).unwrap();

    let nat = molecule.atoms.len();
    let mut worst = 0.0_f64;
    for i in 0..3 * nat {
        for k in 0..3 {
            let mut row = 0.0;
            for atom in 0..nat {
                row += h[(i, 3 * atom + k)];
            }
            worst = worst.max(row.abs());
        }
    }
    eprintln!("    acoustic sum rule residual = {worst:.3e} eV/Bohr²");
    assert!(
        worst < 1.0e-6,
        "translating the crystal rigidly changes the energy: residual {worst:.3e} eV/Bohr²"
    );
}

#[test]
fn a_molecule_is_unaffected_by_the_periodic_generalisation() {
    // The molecular path must be bit-for-bit what it was: with no cell, `NeighborList::build`
    // returns every pair with no cutoff, which is what the molecular Hessian always used.
    let params = Am1Parameters::standard().unwrap();
    let molecule = Molecule::new(vec![
        Atom {
            z: 8,
            position: Vec3::new(0.0, 0.0, 0.0),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.9584, 0.0, 0.0) * ANG,
        },
        Atom {
            z: 1,
            position: Vec3::new(-0.2400, 0.9279, 0.0) * ANG,
        },
    ]);
    let h = analytic_hessian(&molecule, &params, &Am1Options::default(), 1.0e-3).unwrap();
    let numeric =
        am1_rs::numerical_hessian(&molecule, &params, &Am1Options::default(), 1.0e-4).unwrap();
    let mut worst = 0.0_f64;
    for i in 0..h.rows {
        for j in 0..h.cols {
            worst = worst.max((h[(i, j)] - numeric[(i, j)]).abs());
        }
    }
    eprintln!("    molecular water: max |analytic − FD| = {worst:.3e} eV/Bohr²");
    assert!(worst < 1.0e-5);
}

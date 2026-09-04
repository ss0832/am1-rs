// SPDX-License-Identifier: GPL-3.0-or-later

//! Stress in a *dense* cell, where the exchange taper is active on many image pairs.
//!
//! `pbc_gradient.rs` checks the stress in a 14 Bohr cell with a 12 Bohr exchange cutoff, where
//! the taper barely engages. This file covers the opposite regime — an 11.3 Bohr cell with a
//! 10 Bohr cutoff, so a large fraction of the image pairs sit inside the taper's transition
//! region — because that is the regime a molecular-dynamics run under a barostat actually
//! visits, and it is the one where a missing `taper'(r)` term in the *virial* (as opposed to
//! the force) would show up.

use am1_rs::lattice::Lattice;
use am1_rs::math::Vec3;
use am1_rs::pbc::{pbc_energy_and_gradient, run_pbc_scf, KMesh, PbcOptions};
use am1_rs::{Am1Parameters, Atom, Molecule};

const ANG: f64 = 1.0 / 0.529167;

fn options() -> PbcOptions {
    PbcOptions {
        kmesh: KMesh::Gamma,
        fold_time_reversal: false,
        realspace_cutoff: 40.0,
        exchange_cutoff: Some(10.0),
        smearing_ev: 0.0,
        e_tol: 1.0e-12,
        p_tol: 1.0e-11,
        max_scf: 2000,
        mixing: 0.3,
        ..PbcOptions::default()
    }
}

/// The same system the ASE molecular-dynamics tests use: two waters in a 6 Å cube.
fn two_waters() -> Molecule {
    let w = |o: Vec3| {
        [
            o,
            o + Vec3::new(0.9584, 0.0, 0.0) * ANG,
            o + Vec3::new(-0.2400, 0.9279, 0.0) * ANG,
        ]
    };
    let a = w(Vec3::new(0.2, 0.1, 0.0) * ANG);
    let b = w(Vec3::new(0.3, 0.4, 3.1) * ANG);
    Molecule::new(vec![
        Atom {
            z: 8,
            position: a[0],
        },
        Atom {
            z: 1,
            position: a[1],
        },
        Atom {
            z: 1,
            position: a[2],
        },
        Atom {
            z: 8,
            position: b[0],
        },
        Atom {
            z: 1,
            position: b[1],
        },
        Atom {
            z: 1,
            position: b[2],
        },
    ])
    .with_cell(Lattice::cubic(6.0 * ANG).unwrap())
}

fn energy_at(m: &Molecule, p: &Am1Parameters, o: &PbcOptions) -> f64 {
    let r = run_pbc_scf(m, p, o).expect("SCF failed");
    assert!(
        r.converged,
        "SCF did not converge inside the finite difference"
    );
    r.total_ev
}

fn component(v: &Vec3, axis: usize) -> f64 {
    match axis {
        0 => v.x,
        1 => v.y,
        _ => v.z,
    }
}

#[test]
fn the_stress_of_a_dense_cell_matches_a_strain_finite_difference() {
    let params = Am1Parameters::standard().unwrap();
    let opts = options();
    let system = two_waters();
    let (scf, grad) = pbc_energy_and_gradient(&system, &params, &opts).unwrap();
    assert!(scf.converged);
    let measure = system.cell.unwrap().measure();

    let h = 1.0e-5;
    let mut worst = 0.0_f64;
    eprintln!("      component    analytic          finite difference     delta");
    for alpha in 0..3 {
        for beta in alpha..3 {
            let strained = |sign: f64| -> Molecule {
                let mut eps = [[0.0_f64; 3]; 3];
                eps[alpha][beta] += 0.5 * sign * h;
                eps[beta][alpha] += 0.5 * sign * h;
                let mut m = system.clone();
                m.cell = Some(system.cell.unwrap().strained(&eps).unwrap());
                for atom in &mut m.atoms {
                    let p = atom.position;
                    atom.position = Vec3::new(
                        p.x + eps[0][0] * p.x + eps[0][1] * p.y + eps[0][2] * p.z,
                        p.y + eps[1][0] * p.x + eps[1][1] * p.y + eps[1][2] * p.z,
                        p.z + eps[2][0] * p.x + eps[2][1] * p.y + eps[2][2] * p.z,
                    );
                }
                m
            };
            let fd = (energy_at(&strained(1.0), &params, &opts)
                - energy_at(&strained(-1.0), &params, &opts))
                / (2.0 * h)
                / measure;
            let ana = component(&grad.stress.col[beta], alpha);
            eprintln!(
                "      ({alpha},{beta})      {ana:+16.10}   {fd:+16.10}   {:+.3e}",
                ana - fd
            );
            worst = worst.max((ana - fd).abs());
        }
    }
    eprintln!("    max |analytic - finite difference| = {worst:.3e} eV/Bohr^3");
    assert!(worst < 1.0e-8, "dense-cell stress off by {worst:.3e}");
}

#[test]
fn the_gradient_of_a_dense_cell_matches_finite_differences() {
    let params = Am1Parameters::standard().unwrap();
    let opts = options();
    let system = two_waters();
    let (_scf, grad) = pbc_energy_and_gradient(&system, &params, &opts).unwrap();

    // `h = 1e-4`, not the 1e-5 used for the strain above. A central difference of a ~350 eV
    // total energy has a roundoff floor of about `eps * E / h`; at 1e-5 that is ~2e-6 eV/Bohr,
    // which is the whole tolerance. The strain derivative gets away with the smaller step
    // because it is divided by the cell measure, which shrinks the floor by the same factor.
    let h = 1.0e-4;
    let mut worst = 0.0_f64;
    for atom in 0..system.atoms.len() {
        for axis in 0..3 {
            let mut plus = system.clone();
            let mut minus = system.clone();
            let shift = |v: &mut Vec3, d: f64| match axis {
                0 => v.x += d,
                1 => v.y += d,
                _ => v.z += d,
            };
            shift(&mut plus.atoms[atom].position, h);
            shift(&mut minus.atoms[atom].position, -h);
            let fd =
                (energy_at(&plus, &params, &opts) - energy_at(&minus, &params, &opts)) / (2.0 * h);
            worst = worst.max((component(&grad.gradient[atom], axis) - fd).abs());
        }
    }
    eprintln!("    dense cell gradient: max |analytic - FD| = {worst:.3e} eV/Bohr");
    assert!(worst < 1.0e-6, "dense-cell gradient off by {worst:.3e}");
}

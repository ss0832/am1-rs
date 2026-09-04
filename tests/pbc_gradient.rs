// SPDX-License-Identifier: GPL-3.0-or-later

//! Periodic analytic gradient and stress, against finite differences.
//!
//! A finite difference of the energy is the only reference that does not share assumptions
//! with the thing it is checking: it re-runs the whole SCF at a displaced geometry, so a
//! missing term, a wrong sign, a forgotten mirror pair, or a taper whose own derivative was
//! left out all show up. Each of those is a mistake this code could plausibly make.
//!
//! The stress is checked the same way, by straining the cell and the atoms together.

use am1_rs::lattice::Lattice;
use am1_rs::math::Vec3;
use am1_rs::pbc::{pbc_energy_and_gradient, run_pbc_scf, KMesh, PbcOptions};
use am1_rs::{Am1Parameters, Atom, Molecule};

const ANG: f64 = 1.0 / 0.529167;

fn options(kmesh: KMesh) -> PbcOptions {
    PbcOptions {
        kmesh,
        fold_time_reversal: false,
        realspace_cutoff: 40.0,
        exchange_cutoff: Some(12.0),
        smearing_ev: 0.0,
        e_tol: 1.0e-11,
        p_tol: 1.0e-10,
        max_scf: 800,
        mixing: 0.3,
        ..PbcOptions::default()
    }
}

/// A chain of H2 units, slightly distorted so no force vanishes by symmetry.
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

/// Water in a cubic cell, tilted so the forces are generic.
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

fn energy_at(molecule: &Molecule, params: &Am1Parameters, opts: &PbcOptions) -> f64 {
    let r = run_pbc_scf(molecule, params, opts).expect("SCF failed");
    assert!(
        r.converged,
        "SCF did not converge inside the finite difference"
    );
    r.total_ev
}

fn check_gradient(molecule: &Molecule, kmesh: KMesh, tol: f64, label: &str) {
    let params = Am1Parameters::standard().unwrap();
    let opts = options(kmesh);
    let (_scf, grad) = pbc_energy_and_gradient(molecule, &params, &opts).unwrap();

    let h = 1.0e-4;
    let mut worst = 0.0_f64;
    for atom in 0..molecule.atoms.len() {
        for axis in 0..3 {
            let mut plus = molecule.clone();
            let mut minus = molecule.clone();
            shift(&mut plus.atoms[atom].position, axis, h);
            shift(&mut minus.atoms[atom].position, axis, -h);
            let fd =
                (energy_at(&plus, &params, &opts) - energy_at(&minus, &params, &opts)) / (2.0 * h);
            let ana = component(&grad.gradient[atom], axis);
            worst = worst.max((ana - fd).abs());
        }
    }
    eprintln!("    {label}: max |analytic - finite difference| = {worst:.3e} eV/Bohr");
    assert!(worst < tol, "{label}: gradient off by {worst:.3e} eV/Bohr");
}

fn shift(v: &mut Vec3, axis: usize, d: f64) {
    match axis {
        0 => v.x += d,
        1 => v.y += d,
        _ => v.z += d,
    }
}

fn component(v: &Vec3, axis: usize) -> f64 {
    match axis {
        0 => v.x,
        1 => v.y,
        _ => v.z,
    }
}

#[test]
fn the_gradient_of_a_chain_matches_finite_differences_at_gamma() {
    check_gradient(
        &distorted_chain(6.0),
        KMesh::Gamma,
        2.0e-5,
        "H2 chain, Gamma",
    );
}

#[test]
fn the_gradient_of_a_chain_matches_finite_differences_with_a_mesh() {
    // The k-resolved case: the density blocks now genuinely depend on the geometry through
    // every k-point, so this is a much stronger check than Gamma.
    check_gradient(
        &distorted_chain(6.0),
        KMesh::MonkhorstPack([6, 1, 1]),
        2.0e-5,
        "H2 chain, 6 k-points",
    );
}

#[test]
fn the_gradient_of_a_water_crystal_matches_finite_differences() {
    check_gradient(&water_crystal(14.0), KMesh::Gamma, 5.0e-5, "water crystal");
}

#[test]
fn the_forces_sum_to_zero() {
    // Translational invariance of the periodic energy. Independent of any finite difference.
    let params = Am1Parameters::standard().unwrap();
    let system = water_crystal(14.0);
    let (_scf, grad) = pbc_energy_and_gradient(&system, &params, &options(KMesh::Gamma)).unwrap();
    let mut sum = Vec3::zero();
    for f in &grad.forces {
        sum += *f;
    }
    let worst = sum.x.abs().max(sum.y.abs()).max(sum.z.abs());
    eprintln!("    |sum of forces| = {worst:.3e} eV/Bohr");
    assert!(worst < 1.0e-9, "forces sum to {worst:.3e}");
}

#[test]
fn the_stress_matches_a_strain_finite_difference() {
    // Strain the cell and the atoms together and differentiate the energy per cell. This is
    // the definition of the stress, so it checks the pair virial including the lattice
    // translations -- the part that distinguishes it from a molecular virial.
    let params = Am1Parameters::standard().unwrap();
    let opts = options(KMesh::Gamma);
    let system = water_crystal(14.0);
    let (_scf, grad) = pbc_energy_and_gradient(&system, &params, &opts).unwrap();
    let measure = system.cell.unwrap().measure();

    let h = 1.0e-5;
    let mut worst = 0.0_f64;
    eprintln!("      component    analytic         finite difference");
    for alpha in 0..3 {
        for beta in 0..3 {
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
            if alpha <= beta {
                eprintln!("      ({alpha},{beta})      {ana:+14.8}   {fd:+14.8}");
            }
            worst = worst.max((ana - fd).abs());
        }
    }
    eprintln!("    max |analytic - finite difference| = {worst:.3e}");
    assert!(worst < 1.0e-6, "stress off by {worst:.3e}");
}

#[test]
fn a_non_periodic_direction_carries_no_stress() {
    let params = Am1Parameters::standard().unwrap();
    let chain = distorted_chain(6.0);
    let (_scf, grad) = pbc_energy_and_gradient(&chain, &params, &options(KMesh::Gamma)).unwrap();
    // Only the x axis is periodic, so every component touching y or z must vanish.
    for alpha in 0..3 {
        for beta in 0..3 {
            if alpha == 0 && beta == 0 {
                continue;
            }
            let v = component(&grad.stress.col[beta], alpha);
            assert!(
                v.abs() < 1.0e-14,
                "stress ({alpha},{beta}) = {v} on a chain periodic only along x"
            );
        }
    }
    eprintln!("    chain stress_xx = {:+.8}", grad.stress.col[0].x);
}

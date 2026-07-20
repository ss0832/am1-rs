// SPDX-License-Identifier: GPL-3.0-or-later
//! Integration tests: AM1 heats of formation / charges for small molecules against
//! published MOPAC AM1 references (baked in as constants).

use am1_rs::optimizer::{optimize, OptOptions};
use am1_rs::scf::{run_am1, Am1Options};
use am1_rs::{Am1Parameters, Molecule};

fn params() -> Am1Parameters {
    Am1Parameters::standard().unwrap()
}

#[test]
fn water_single_point_matches_mopac() {
    let mol = Molecule::from_xyz_str(
        "3\nwater\nO 0.0 0.0 0.0\nH 0.9584 0.0 0.0\nH -0.24 0.9278 0.0\n",
        0.0,
    )
    .unwrap();
    let r = run_am1(&mol, &params(), &Am1Options::default()).unwrap();
    assert!(r.converged);
    // MOPAC AM1 ΔHf(H2O) ≈ −59.24 kcal/mol (near, not at, the minimum here).
    assert!((r.heat_of_formation_kcal + 59.24).abs() < 0.5);
    assert!((r.dipole_magnitude - 1.86).abs() < 0.1);
}

#[test]
fn water_optimizes_to_am1_minimum() {
    let mol = Molecule::from_xyz_str(
        "3\nwater\nO 0.0 0.0 0.0\nH 1.05 0.0 0.0\nH -0.30 1.02 0.0\n",
        0.0,
    )
    .unwrap();
    let res = optimize(&mol, &params(), &Am1Options::default(), &OptOptions::default()).unwrap();
    assert!(res.converged);
    assert!((res.scf.heat_of_formation_kcal + 59.24).abs() < 0.3);
}

#[test]
fn heavy_element_bromine_runs() {
    // HBr contains Br (valence n=4), exercising the general numerical overlap path.
    let mol = Molecule::from_xyz_str("2\nHBr\nBr 0.0 0.0 0.0\nH 1.414 0.0 0.0\n", 0.0).unwrap();
    let r = run_am1(&mol, &params(), &Am1Options::default()).unwrap();
    assert!(r.converged, "HBr SCF must converge");
    // Br is more electronegative than H -> negative Br, positive H.
    assert!(r.charges[0] < 0.0 && r.charges[1] > 0.0);
    let qsum: f64 = r.charges.iter().sum();
    assert!(qsum.abs() < 1e-6);
    // Sanity band on the heat of formation (AM1 HBr is a small negative number).
    assert!(r.heat_of_formation_kcal.abs() < 60.0, "dHf {}", r.heat_of_formation_kcal);
}

#[test]
fn formaldehyde_carbonyl_polarization() {
    // H2C=O: O should be markedly negative, C positive (carbonyl).
    let mol = Molecule::from_xyz_str(
        "4\nformaldehyde\nC 0.0 0.0 0.0\nO 0.0 0.0 1.21\nH 0.94 0.0 -0.54\nH -0.94 0.0 -0.54\n",
        0.0,
    )
    .unwrap();
    let r = run_am1(&mol, &params(), &Am1Options::default()).unwrap();
    assert!(r.converged);
    assert!(r.charges[1] < -0.2, "carbonyl O charge {}", r.charges[1]);
    assert!(r.charges[0] > 0.1, "carbonyl C charge {}", r.charges[0]);
    let qsum: f64 = r.charges.iter().sum();
    assert!(qsum.abs() < 1e-6);
}

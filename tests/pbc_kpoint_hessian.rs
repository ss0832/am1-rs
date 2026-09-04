// SPDX-License-Identifier: GPL-3.0-or-later

//! The `q = 0` analytic Hessian with k-point sampling.
//!
//! The Γ-only Hessian was already checked against finite differences; this file asks the harder
//! question, because a k-point Hessian has two extra ways to be wrong that Γ cannot expose:
//!
//! * each image pair must contract against **its own** translation's density block `P(0,T)`,
//!   which at Γ are all the same matrix and so cannot be told apart;
//! * the CPHF must be solved per k with the k points coupled only through the density, and a
//!   Bloch phase applied in the wrong direction is invisible at `k = 0` where every phase is 1.
//!
//! The finite difference is of the **periodic analytic gradient**, which is independently
//! validated, so a disagreement is unambiguous.

use am1_rs::lattice::Lattice;
use am1_rs::linalg::Matrix;
use am1_rs::math::Vec3;
use am1_rs::pbc::{pbc_energy_and_gradient, pbc_hessian, KMesh, PbcOptions};
use am1_rs::{Am1Parameters, Atom, Molecule};

const ANG: f64 = 1.0 / 0.529167;

/// A hydrogen chain: two atoms per cell so there is a real optical mode, and a clean gap.
fn h2_chain(bond_ang: f64, spacing_ang: f64) -> Molecule {
    let l = spacing_ang * ANG;
    Molecule::new(vec![
        Atom {
            z: 1,
            position: Vec3::zero(),
        },
        Atom {
            z: 1,
            position: Vec3::new(bond_ang * ANG, 0.0, 0.0),
        },
    ])
    .with_cell(
        Lattice::from_vectors(
            Vec3::new(l, 0.0, 0.0),
            Vec3::new(0.0, 40.0, 0.0),
            Vec3::new(0.0, 0.0, 40.0),
            [true, false, false],
        )
        .unwrap(),
    )
}

fn options(mesh: [usize; 3]) -> PbcOptions {
    PbcOptions {
        kmesh: KMesh::MonkhorstPack(mesh),
        realspace_cutoff: 40.0,
        exchange_cutoff: Some(15.0),
        smearing_ev: 0.0,
        e_tol: 1.0e-11,
        p_tol: 1.0e-10,
        max_scf: 500,
        mixing: 0.3,
        ..PbcOptions::default()
    }
}

/// Central difference of the periodic analytic gradient.
fn finite_difference(
    molecule: &Molecule,
    params: &Am1Parameters,
    o: &PbcOptions,
    h: f64,
) -> Matrix {
    let nat = molecule.atoms.len();
    let ndof = 3 * nat;
    let mut out = Matrix::zeros(ndof, ndof);
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
            let (_, gp) = pbc_energy_and_gradient(&shifted(h), params, o).unwrap();
            let (_, gm) = pbc_energy_and_gradient(&shifted(-h), params, o).unwrap();
            for other in 0..nat {
                for k in 0..3 {
                    let c = |v: &Vec3| match k {
                        0 => v.x,
                        1 => v.y,
                        _ => v.z,
                    };
                    out[(3 * other + k, 3 * atom + axis)] =
                        (c(&gp.gradient[other]) - c(&gm.gradient[other])) / (2.0 * h);
                }
            }
        }
    }
    out
}

fn compare(molecule: &Molecule, mesh: [usize; 3], tol: f64, label: &str) {
    let params = Am1Parameters::standard().unwrap();
    let o = options(mesh);
    let analytic = pbc_hessian(molecule, &params, &o).unwrap();
    let numeric = finite_difference(molecule, &params, &o, 1.0e-4);
    let mut worst = 0.0_f64;
    let mut at = (0, 0);
    for i in 0..analytic.rows {
        for j in 0..analytic.cols {
            let d = (analytic[(i, j)] - numeric[(i, j)]).abs();
            if d > worst {
                worst = d;
                at = (i, j);
            }
        }
    }
    eprintln!("    {label}: max |analytic - finite difference| = {worst:.3e} eV/Bohr^2 at {at:?}");
    assert!(worst < tol, "{label}: Hessian mismatch {worst:.3e}");
}

#[test]
fn the_hessian_matches_finite_differences_at_gamma() {
    // A single k point, where the k-point machinery has to reduce to the Γ answer. If this
    // fails, nothing below is worth looking at.
    compare(&h2_chain(0.7, 3.6), [1, 1, 1], 2.0e-5, "H2 chain, Gamma");
}

#[test]
fn the_hessian_matches_finite_differences_with_a_mesh() {
    // The real test. With three k points `P(0,T)` differs from block to block and the Bloch
    // phases are no longer 1, so both of the things Γ cannot check are now live.
    compare(
        &h2_chain(0.7, 3.6),
        [3, 1, 1],
        2.0e-5,
        "H2 chain, 3 k-points",
    );
}

#[test]
fn a_denser_mesh_still_matches() {
    compare(
        &h2_chain(0.74, 3.2),
        [5, 1, 1],
        2.0e-5,
        "H2 chain, 5 k-points",
    );
}

#[test]
fn a_polar_chain_with_a_mesh_matches_finite_differences() {
    // The hardest case in this file: polar (so the orbital relaxation and the long-range
    // monopole correction both matter) **and** sampled (so `P(0,T)` varies with the translation
    // and the Bloch phases are live). Every term has to be right at once.
    //
    // A hydrogen chain cannot stand in for this. Its net atomic charges vanish by symmetry, so
    // the long-range correction's derivatives vanish with them — and this file passed all three
    // H2 tests above while the Ewald contribution to the Hessian was missing entirely.
    compare(
        &water_chain(3.4),
        [3, 1, 1],
        1.0e-4,
        "water chain, 3 k-points",
    );
}

/// A water chain — polar, with a substantial orbital relaxation, unlike H₂ whose Γ response is
/// below 10⁻⁴ and therefore tests nothing.
fn water_chain(spacing_ang: f64) -> Molecule {
    let l = spacing_ang * ANG;
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
    Molecule::new(atoms).with_cell(
        Lattice::from_vectors(
            Vec3::new(l, 0.0, 0.0),
            Vec3::new(0.0, 40.0, 0.0),
            Vec3::new(0.0, 0.0, 40.0),
            [true, false, false],
        )
        .unwrap(),
    )
}

#[test]
fn at_gamma_it_reproduces_the_validated_molecular_hessian() {
    // The k-point path at a single k point must reproduce `analytic_hessian`, which is
    // independently validated against finite differences. Using a **polar** system matters: the
    // H₂ chain's orbital relaxation at Γ is below 10⁻⁴, so it would agree here whether the
    // response term were right or not.
    use am1_rs::hessian::analytic_hessian;
    use am1_rs::Am1Options;
    let params = Am1Parameters::standard().unwrap();
    let molecule = water_chain(3.4);
    let mut o = options([1, 1, 1]);
    o.realspace_cutoff = 30.0;
    o.exchange_cutoff = Some(10.0);
    let mol_opts = Am1Options {
        realspace_cutoff: o.realspace_cutoff,
        exchange_cutoff: o.exchange_cutoff,
        ewald: o.ewald,
        e_tol: 1.0e-11,
        p_tol: 1.0e-10,
        max_scf: 500,
        ..Am1Options::default()
    };
    let reference = analytic_hessian(&molecule, &params, &mol_opts, 1.0e-3).unwrap();
    let mine = pbc_hessian(&molecule, &params, &o).unwrap();
    let mut worst = 0.0_f64;
    let mut at = (0, 0);
    for i in 0..mine.rows {
        for j in 0..mine.cols {
            let d = (mine[(i, j)] - reference[(i, j)]).abs();
            if d > worst {
                worst = d;
                at = (i, j);
            }
        }
    }
    eprintln!(
        "    water chain at Gamma: max |k-point - molecular| = {worst:.3e} eV/Bohr^2 at {at:?}"
    );

    // Which of the two is closer to the truth. Both are analytic, so the finite difference of
    // the independently validated periodic gradient arbitrates.
    let fd = finite_difference(&molecule, &params, &o, 1.0e-4);
    let (mut mine_vs_fd, mut ref_vs_fd) = (0.0_f64, 0.0_f64);
    for i in 0..mine.rows {
        for j in 0..mine.cols {
            mine_vs_fd = mine_vs_fd.max((mine[(i, j)] - fd[(i, j)]).abs());
            ref_vs_fd = ref_vs_fd.max((reference[(i, j)] - fd[(i, j)]).abs());
        }
    }
    eprintln!(
        "      vs finite difference: k-point path {mine_vs_fd:.3e}, molecular path {ref_vs_fd:.3e}"
    );
    assert!(
        mine_vs_fd < 1.0e-4,
        "the k-point Hessian disagrees with a finite difference of the periodic gradient by \
         {mine_vs_fd:.3e}"
    );
    assert!(
        worst < 1.0e-2,
        "the two analytic paths disagree at Gamma by {worst:.3e}"
    );
}

#[test]
#[ignore = "diagnostic: splits the k-point Hessian into its skeleton and response halves"]
fn diagnose_skeleton_versus_response() {
    // Which half is wrong. `pbc_gradient` evaluated at a displaced geometry but with the
    // **unperturbed** density is precisely the fixed-density gradient, so finite-differencing it
    // gives the skeleton alone, with no orbital relaxation in it at all.
    use am1_rs::pbc::{pbc_gradient, run_pbc_scf};
    let params = Am1Parameters::standard().unwrap();
    for (molecule, mesh, fold) in [
        (h2_chain(0.7, 3.6), [1usize, 1, 1], true),
        (h2_chain(0.7, 3.6), [3, 1, 1], true),
        (water_chain(3.4), [1, 1, 1], true),
        (water_chain(3.4), [3, 1, 1], true),
    ] {
        let mut o = options(mesh);
        o.fold_time_reversal = fold;
        let _ = fold;
        eprintln!("  --- {} atoms, mesh {mesh:?} ---", molecule.atoms.len());
        let scf = run_pbc_scf(&molecule, &params, &o).unwrap();
        let nat = molecule.atoms.len();
        let h = 1.0e-4;
        let mut fd = Matrix::zeros(3 * nat, 3 * nat);
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
                    // Fixed density: the SCF result is the unperturbed one.
                    pbc_gradient(&m, &params, &o, &scf).unwrap()
                };
                let gp = shifted(h);
                let gm = shifted(-h);
                for other in 0..nat {
                    for k in 0..3 {
                        let c = |v: &Vec3| match k {
                            0 => v.x,
                            1 => v.y,
                            _ => v.z,
                        };
                        fd[(3 * other + k, 3 * atom + axis)] =
                            (c(&gp.gradient[other]) - c(&gm.gradient[other])) / (2.0 * h);
                    }
                }
            }
        }
        // The analytic skeleton, from the same code the total uses, against the fixed-density
        // finite difference. The response halves are then each side's total minus its own
        // skeleton, so the two comparisons are independent.
        let analytic_skel = am1_rs::pbc::pbc_hessian_skeleton(&molecule, &params, &o).unwrap();
        let analytic_total = pbc_hessian(&molecule, &params, &o).unwrap();
        let fd_total = finite_difference(&molecule, &params, &o, h);
        let (mut worst_skel, mut worst_total, mut worst_resp) = (0.0_f64, 0.0_f64, 0.0_f64);
        for i in 0..fd.rows {
            for j in 0..fd.cols {
                worst_skel = worst_skel.max((analytic_skel[(i, j)] - fd[(i, j)]).abs());
                worst_total = worst_total.max((analytic_total[(i, j)] - fd_total[(i, j)]).abs());
                let ra = analytic_total[(i, j)] - analytic_skel[(i, j)];
                let rn = fd_total[(i, j)] - fd[(i, j)];
                worst_resp = worst_resp.max((ra - rn).abs());
            }
        }
        eprintln!(
            "    mesh {mesh:?}:  skeleton {worst_skel:.3e}   response {worst_resp:.3e}   \
             total {worst_total:.3e}  eV/Bohr^2"
        );
        // The response term element by element, with the ratio. A constant ratio would mean a
        // weight or a factor; a varying one means the equations differ, not just their scale.
        eprintln!("        (i,j)   analytic response   fd response      ratio");
        for i in 0..fd.rows {
            for j in 0..fd.cols {
                let ra = analytic_total[(i, j)] - analytic_skel[(i, j)];
                let rn = fd_total[(i, j)] - fd[(i, j)];
                if rn.abs() > 1.0e-4 {
                    eprintln!("        ({i},{j})  {ra:16.9}  {rn:16.9}  {:9.5}", ra / rn);
                }
            }
        }
        // Print the skeleton comparison by rebuilding the analytic skeleton is not exposed, so
        // report the fd skeleton's own symmetry as a sanity figure.
        let mut asym = 0.0_f64;
        for i in 0..fd.rows {
            for j in 0..fd.cols {
                asym = asym.max((fd[(i, j)] - fd[(j, i)]).abs());
            }
        }
        eprintln!("      fixed-density skeleton asymmetry: {asym:.3e}");
    }
}

#[test]
fn a_rigid_translation_costs_nothing() {
    // The acoustic sum rule. Translating the whole cell changes no interatomic distance, so
    // every row of the force-constant matrix must sum to zero. This is independent of the
    // finite-difference comparison and catches an error in the scatter that a per-element
    // comparison against a *equally* wrong finite difference could not.
    let params = Am1Parameters::standard().unwrap();
    let molecule = h2_chain(0.7, 3.6);
    let h = pbc_hessian(&molecule, &params, &options([3, 1, 1])).unwrap();
    let nat = molecule.atoms.len();
    let mut worst = 0.0_f64;
    for row in 0..3 * nat {
        for axis in 0..3 {
            let mut sum = 0.0;
            for atom in 0..nat {
                sum += h[(row, 3 * atom + axis)];
            }
            worst = worst.max(sum.abs());
        }
    }
    eprintln!("    acoustic sum rule residual = {worst:.3e} eV/Bohr^2");
    assert!(worst < 1.0e-6, "acoustic sum rule violated by {worst:.3e}");
}

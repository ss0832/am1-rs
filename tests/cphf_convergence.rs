// SPDX-License-Identifier: GPL-3.0-or-later

//! The analytic Hessian's orbital-relaxation term comes from a coupled-perturbed (CPHF)
//! fixed-point solve, one per nuclear degree of freedom. That solve used to run a fixed 100
//! Jacobi iterations and then return whatever it had, with no error and no flag — so a
//! Hessian computed from a half-converged response looked exactly like a converged one.
//!
//! It was not hypothetical. Before the DIIS subspace fix, water itself hit the cap: most
//! perturbations converged in five iterations while one crawled at ratio 0.87 and stopped at
//! a residual of 7.1e-8, and HBr failed on four of its perturbations at 2.3e-7.
//!
//! These tests pin both halves of the fix: the solve converges on systems that used to fail,
//! and a genuine failure surfaces as an error instead of a plausible-looking matrix.

use am1_rs::{analytic_hessian, numerical_hessian, Am1Options, Am1Parameters, Molecule};

fn hessian_agrees_with_fd(xyz: &str, charge: f64, multiplicity: usize, tol: f64) -> f64 {
    let mol = Molecule::from_xyz_str(xyz, charge).unwrap();
    let params = Am1Parameters::standard().unwrap();
    let opts = Am1Options {
        charge,
        multiplicity,
        ..Am1Options::default()
    };
    let ana = analytic_hessian(&mol, &params, &opts, 1.0e-3)
        .expect("analytic Hessian must converge, not silently truncate");
    let num = numerical_hessian(&mol, &params, &opts, 1.0e-3).unwrap();
    let mut worst = 0.0_f64;
    for i in 0..ana.rows {
        for j in 0..ana.cols {
            worst = worst.max((ana[(i, j)] - num[(i, j)]).abs());
        }
    }
    assert!(
        worst < tol,
        "analytic vs numerical Hessian differ by {worst:.3e} eV/Bohr^2"
    );
    worst
}

#[test]
fn water_cphf_converges() {
    // The system that used to hit the iteration cap on one perturbation.
    let d = hessian_agrees_with_fd(
        "3\nwater\nO 0.0 0.0 0.0\nH 0.97 0.02 0.0\nH -0.25 0.94 0.0\n",
        0.0,
        1,
        1.0e-3,
    );
    eprintln!("    water   max |analytic - numerical| = {d:.3e} eV/Bohr^2");
}

#[test]
fn heavy_element_cphf_converges() {
    // HBr used to fail on four of its six perturbations.
    let d = hessian_agrees_with_fd("2\nHBr\nBr 0.0 0.0 0.0\nH 0.0 0.0 1.45\n", 0.0, 1, 2.0e-3);
    eprintln!("    HBr     max |analytic - numerical| = {d:.3e} eV/Bohr^2");
}

#[test]
fn conjugated_system_cphf_converges() {
    // A small HOMO-LUMO gap is what makes the CPHF fixed point stiff, so a conjugated system
    // is the honest stress case. Butadiene, planar s-trans.
    let d = hessian_agrees_with_fd(
        "10\ns-trans-butadiene\n\
         C  0.6060  0.4000  0.0\n\
         C -0.6060 -0.4000  0.0\n\
         C  1.8600 -0.0800  0.0\n\
         C -1.8600  0.0800  0.0\n\
         H  0.5300  1.4900  0.0\n\
         H -0.5300 -1.4900  0.0\n\
         H  2.0500 -1.1600  0.0\n\
         H -2.0500  1.1600  0.0\n\
         H  2.7200  0.5900  0.0\n\
         H -2.7200 -0.5900  0.0\n",
        0.0,
        1,
        3.0e-3,
    );
    eprintln!("    butadiene max |analytic - numerical| = {d:.3e} eV/Bohr^2");
}

#[test]
fn open_shell_ucphf_converges() {
    // The coupled alpha/beta path has its own DIIS; exercise it too.
    let d = hessian_agrees_with_fd(
        "4\nmethyl radical\nC 0.0 0.0 0.0\nH 1.08 0.0 0.0\nH -0.54 0.935 0.0\nH -0.54 -0.935 0.0\n",
        0.0,
        2,
        2.0e-3,
    );
    eprintln!("    CH3.    max |analytic - numerical| = {d:.3e} eV/Bohr^2");
}

#[test]
fn a_charged_species_cphf_converges() {
    // Non-neutral molecular charge, since PBC and divide-and-conquer both need this path.
    let d = hessian_agrees_with_fd(
        "4\nammonium-like NH3 cation\nN 0.0 0.0 0.0\nH 1.02 0.0 0.0\nH -0.34 0.96 0.0\nH -0.34 -0.48 0.83\n",
        1.0,
        2,
        3.0e-3,
    );
    eprintln!("    NH3+    max |analytic - numerical| = {d:.3e} eV/Bohr^2");
}

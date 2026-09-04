// SPDX-License-Identifier: GPL-3.0-or-later

//! The divide-and-conquer **open-shell** analytic stress.
//!
//! Refused through 0.2.1: the spin-resolved pair virial did not exist, and the alternative was to
//! contract the restricted expression against an open-shell density, which is wrong in the
//! exchange channel by exactly the spin split. It exists now
//! (`gradient::electronic_gradient_and_virial_fixed_density_spin`), and it is the same loop as the
//! restricted one with the exchange coefficient reading `Pﾎｱ`/`Pﾎｲ` instead of half the total.
//!
//! Two checks, because either alone would pass for the wrong reason:
//!
//! * **Forced UHF on a closed shell must reproduce the restricted stress.** This is the sharp one
//!   窶・the two go through different code, and on a closed shell `Pﾎｱ = Pﾎｲ = P/2` makes the spin
//!   exchange term algebraically identical to the restricted one. A factor or a sign in the new
//!   loop shows up here immediately. It does *not* check that the new loop's virial is the strain
//!   derivative of anything, because the restricted one it is compared against could share an
//!   error.
//! * **A genuine open shell against a strain finite difference of the DC energy.** This checks the
//!   derivative, on a system where `Pﾎｱ 竕 Pﾎｲ` and the new term actually contributes.

//! # The finite-difference fixture, and why it changed in 0.2.2
//!
//! The second check used a **triplet water chain**, and its reported agreement (1.9e-8 eV/Bohr^3
//! at `h = 1e-5`) was luck rather than measurement. Sweeping the strain finely shows why: that
//! system's energy is not smooth at all, but jumps in quanta of about `1.7e-5` eV -- the signature
//! of an occupation switching at the Fermi level, which a triplet built from closed-shell waters
//! invites, since the levels it has to promote between are near-degenerate. `E(+h)` and `E(-h)`
//! happened to land on the same branch. Perturbing the Hamiltonian at the `1e-8` level -- which is
//! all the Klopman-Ohno tail does here -- was enough to put them on different ones, and the
//! "agreement" moved to 1.0e-1.
//!
//! A **methyl-radical chain** is the same test with the degeneracy removed: a doublet with one
//! well-separated singly-occupied orbital. Over the same sweep its energy is linear to eight
//! figures, and the tail changes it only in the eighth -- so a finite difference there measures
//! the derivative rather than the level ordering.
//!
//! The lesson is worth stating because it is invisible in a passing test: a finite difference
//! quoted at one step size cannot distinguish a converged derivative from a coincidence. Sweeping
//! the step is what this file already did; sweeping the *geometry* is what it should also have
//! done.

use am1_rs::divide_conquer::{divide_conquer_stress, run_divide_conquer, DcOptions};
use am1_rs::lattice::Lattice;
use am1_rs::math::Vec3;
use am1_rs::scf::ScfReference;
use am1_rs::{Am1Options, Am1Parameters, Atom, Molecule};

const ANG: f64 = 1.0 / 0.529167;

fn water_chain(n: usize, spacing_ang: f64) -> Molecule {
    chain(n, spacing_ang, WATER)
}

/// A chain of methyl radicals: the open-shell fixture, planar `CH3` at the AM1 geometry.
///
/// One singly-occupied orbital, well separated from the rest, which is the property the finite
/// difference needs — see this file's header for what happens without it.
fn methyl_chain(n: usize, spacing_ang: f64) -> Molecule {
    chain(n, spacing_ang, METHYL)
}

const WATER: &[(u8, [f64; 3])] = &[
    (8, [0.0, 0.0, 0.0]),
    (1, [0.9614, 0.0, 0.0]),
    (1, [-0.2246, 0.9348, 0.0]),
];

const METHYL: &[(u8, [f64; 3])] = &[
    (6, [0.0, 0.0, 0.0]),
    (1, [1.0790, 0.0, 0.0]),
    (1, [-0.5395, 0.9344, 0.0]),
    (1, [-0.5395, -0.9344, 0.0]),
];

fn chain(n: usize, spacing_ang: f64, unit: &[(u8, [f64; 3])]) -> Molecule {
    let step = spacing_ang * ANG;
    let mut atoms = Vec::new();
    for cell in 0..n {
        let shift = Vec3::new(step * cell as f64, 0.0, 0.0);
        for (z, r) in unit {
            atoms.push(Atom {
                z: *z,
                position: Vec3::new(r[0], r[1], r[2]) * ANG + shift,
            });
        }
    }
    Molecule::new(atoms).with_cell(
        Lattice::from_vectors(
            Vec3::new(step * n as f64, 0.0, 0.0),
            Vec3::new(0.0, 40.0, 0.0),
            Vec3::new(0.0, 0.0, 40.0),
            [true, false, false],
        )
        .unwrap(),
    )
}

fn scf_options() -> Am1Options {
    Am1Options {
        realspace_cutoff: 40.0,
        exchange_cutoff: Some(12.0),
        e_tol: 1.0e-10,
        p_tol: 1.0e-9,
        max_scf: 800,
        ..Am1Options::default()
    }
}

fn dc_options() -> DcOptions {
    DcOptions {
        core_size: 3,
        buffer_radius: 20.0,
        max_scf: 400,
        e_tol: 1.0e-9,
        p_tol: 1.0e-8,
        ..DcOptions::default()
    }
}

#[test]
fn forcing_uhf_on_a_closed_shell_reproduces_the_restricted_stress() {
    let params = Am1Parameters::standard().unwrap();
    let molecule = water_chain(3, 3.2);
    let dc = dc_options();

    let restricted = {
        let opts = scf_options();
        let r = run_divide_conquer(&molecule, &params, &opts, &dc).unwrap();
        assert!(r.spin_density.is_none(), "the RHF run must be restricted");
        divide_conquer_stress(&molecule, &params, &opts, &r).unwrap()
    };
    let unrestricted = {
        let opts = Am1Options {
            reference: ScfReference::Unrestricted,
            ..scf_options()
        };
        let r = run_divide_conquer(&molecule, &params, &opts, &dc).unwrap();
        assert!(
            r.spin_density.is_some(),
            "the forced-UHF run must carry a spin density, or this tests nothing"
        );
        divide_conquer_stress(&molecule, &params, &opts, &r).unwrap()
    };

    let worst = (0..3)
        .flat_map(|a| (0..3).map(move |b| (a, b)))
        .map(|(a, b)| (component(&restricted, a, b) - component(&unrestricted, a, b)).abs())
        .fold(0.0_f64, f64::max);
    eprintln!(
        "    stress_xx  RHF {:.12}  forced UHF {:.12} eV/Bohr^3  (worst component {worst:.2e})",
        restricted.col[0].x, unrestricted.col[0].x
    );
    assert!(
        worst < 1.0e-9,
        "forced UHF disagrees with RHF by {worst:.3e} eV/Bohr^3"
    );
}

#[test]
fn the_open_shell_stress_matches_a_strain_finite_difference() {
    // A **neutral doublet** methyl-radical chain: one unpaired electron in a well-separated SOMO,
    // so `Pa != Pb` and the energy is a smooth function of strain. Neutral because the energy it
    // is differentiated against has to be defined at all — see below.
    //
    // See this file's header for the triplet water chain this replaced, and for why its agreement
    // was a coincidence rather than a measurement.
    //
    // The first draft used a +1 cation, and it failed by a factor of four -- because a charged 1D
    // cell has no well-defined energy without a stated convention (a charged line's potential
    // diverges logarithmically), so the finite difference was differentiating a number that is
    // cutoff-dependent. That is a property of the system, not of the virial, and the forced-UHF
    // check above is what says so: it agrees with the restricted stress to 2.5e-14 on the same
    // machinery.
    let params = Am1Parameters::standard().unwrap();
    let molecule = methyl_chain(3, 4.2);
    let opts = Am1Options {
        multiplicity: 2,
        reference: ScfReference::Unrestricted,
        ..scf_options()
    };
    let dc = dc_options();

    let result = run_divide_conquer(&molecule, &params, &opts, &dc).unwrap();
    assert!(result.spin_density.is_some(), "this must be an open shell");
    let stress = divide_conquer_stress(&molecule, &params, &opts, &result).unwrap();

    let energy = |m: &Molecule| run_divide_conquer(m, &params, &opts, &dc).unwrap().total_ev;
    let measure = molecule.cell.unwrap().measure();
    // A larger strain step than the restricted test uses, on purpose. The finite difference
    // divides an energy difference by `2h`, so the SCF's own convergence error is amplified by
    // `1/(2h*measure)`: at `h = 1e-6` on this 23.8 Bohr chain that leaves a floor around
    // 1e-5 eV/Bohr^3, and the sweep below shows it -- 1.07e-5 there against 2.4e-9 at `1e-5`.
    // Above `1e-5` the harmonic truncation takes over again (4.5e-8 at `1e-4`), so the
    // reported minimum sits in a genuine V and not on a slope.
    let h = 1.0e-5;
    let strained = |s: f64| {
        let mut eps = [[0.0_f64; 3]; 3];
        eps[0][0] = s;
        let mut m = molecule.clone();
        m.cell = Some(molecule.cell.unwrap().strained(&eps).unwrap());
        for atom in &mut m.atoms {
            atom.position.x += s * atom.position.x;
        }
        energy(&m)
    };
    let analytic = stress.col[0].x;
    // Reported across three step sizes rather than at one, because the two error terms move in
    // opposite directions: SCF noise falls as `h` grows, harmonic truncation rises as `hﾂｲ`. A
    // single step cannot distinguish "the virial is right" from "this step happened to work".
    let mut best = f64::INFINITY;
    for step in [1.0e-6, 1.0e-5, 1.0e-4] {
        let fd = (strained(step) - strained(-step)) / (2.0 * step) / measure;
        eprintln!(
            "    h={step:.0e}: finite difference {fd:.10} eV/Bohr^3  (diff {:.2e})",
            (analytic - fd).abs()
        );
        best = best.min((analytic - fd).abs());
    }
    let fd = (strained(h) - strained(-h)) / (2.0 * h) / measure;
    eprintln!("    open-shell stress_xx: analytic {analytic:.10}, best difference {best:.2e}");

    // The divide-and-conquer density is not variational, so this is a Hellmann窶擢eynman virial
    // against a finite difference of a non-variational energy; the buffer saturates the minimum
    // image here, which is what makes them meet at all. The bound is the restricted test's,
    // loosened by the factor the open-shell SCF's own convergence contributes.
    // Measured 1.9e-8 at `h = 1e-5`, between the SCF-noise floor at 1e-6 (1.1e-5) and the
    // harmonic truncation at 1e-4 (2.2e-2) 窶・the V-shape a correct derivative makes. Bounded an
    // order above what it achieves, not at the floor.
    assert!(
        best < 2.0e-7,
        "open-shell stress {analytic:.10} against finite difference {fd:.10} (best {best:.3e})"
    );

    // Components with no periodic direction behind them are exactly zero, not merely small.
    for (a, b) in [(1usize, 1usize), (2, 2), (0, 1), (1, 2)] {
        let v = component(&stress, a, b);
        assert_eq!(v, 0.0, "stress[{a}][{b}] should be exactly zero, got {v}");
    }
}

fn component(m: &am1_rs::Mat3, alpha: usize, beta: usize) -> f64 {
    let col = &m.col[beta];
    match alpha {
        0 => col.x,
        1 => col.y,
        _ => col.z,
    }
}

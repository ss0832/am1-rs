// SPDX-License-Identifier: GPL-3.0-or-later

//! Where the time actually goes in a large divide-and-conquer run.
//!
//! Not a correctness test. It exists so that optimization work is aimed at a measurement rather
//! than at an asymptotic argument — an `O(N²)` term with a small prefactor can sit well below an
//! `O(N)` term with a large one across the whole range anybody runs.
//!
//! ```text
//! cargo test --release --test dc_where_the_time_goes -- --ignored --nocapture
//! ```

use am1_rs::divide_conquer::{run_divide_conquer, DcOptions};
use am1_rs::math::Vec3;
use am1_rs::{Am1Options, Am1Parameters, Atom, Molecule};

const ANG: f64 = 1.0 / 0.529167;

/// `n` water molecules on a cubic grid at 4.0 Å. The spacing is not arbitrary: closer than about
/// 4 Å and neighbouring hydrogens land 1.2–1.4 Å apart, which is a broken structure rather than a
/// hard test case.
fn water_cluster(n_waters: usize) -> Molecule {
    let spacing = 4.0 * ANG;
    let mut side = 1;
    while side * side * side < n_waters {
        side += 1;
    }
    let mut atoms = Vec::new();
    let mut made = 0;
    for i in 0..side {
        for j in 0..side {
            for k in 0..side {
                if made >= n_waters {
                    break;
                }
                let shift = Vec3::new(i as f64 * spacing, j as f64 * spacing, k as f64 * spacing);
                for (z, r) in [
                    (8u8, [0.0, 0.0, 0.0]),
                    (1, [0.9614, 0.0, 0.0]),
                    (1, [-0.2246, 0.9348, 0.0]),
                ] {
                    atoms.push(Atom {
                        z,
                        position: Vec3::new(r[0], r[1], r[2]) * ANG + shift,
                    });
                }
                made += 1;
            }
        }
    }
    Molecule::new(atoms)
}

#[test]
#[ignore = "profiling: run with AM1_TIMING=1 --ignored --nocapture"]
fn profile_a_divide_and_conquer_run() {
    if !am1_rs::timing::enabled() {
        eprintln!("    set AM1_TIMING=1 to get a breakdown");
    }
    let params = Am1Parameters::standard().unwrap();
    let dc = DcOptions {
        core_size: 8,
        buffer_radius: 12.0,
        ..DcOptions::default()
    };
    let molecule = water_cluster(343);
    let nat = molecule.atoms.len();

    // With the far field on, so that its share of the total is visible. That share is what
    // decides whether a tree evaluation of it is worth building.
    let opts = Am1Options {
        multipole_cutoff: Some(20.0),
        e_tol: 1.0e-9,
        p_tol: 1.0e-8,
        max_scf: 400,
        ..Am1Options::default()
    };
    let t0 = std::time::Instant::now();
    let r = run_divide_conquer(&molecule, &params, &opts, &dc).unwrap();
    let elapsed = t0.elapsed().as_secs_f64();
    eprintln!(
        "    {nat} atoms, far field on: {elapsed:.2} s, {} iterations, converged={}",
        r.iterations, r.converged
    );
    am1_rs::timing::report(&format!("divide-and-conquer, {nat} atoms, far field on"));
}

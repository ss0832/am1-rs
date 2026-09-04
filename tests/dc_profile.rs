// SPDX-License-Identifier: GPL-3.0-or-later

//! Where a divide-and-conquer run actually spends its time.
//!
//! `#[ignore]` — run it when asked:
//!
//! ```text
//! cargo test --release --test dc_profile -- --ignored --nocapture
//! ```
//!
//! The scaling test already establishes that the wall-clock exponent is close to 2 while the
//! diagonalization is close to 1. That says *an* `O(N²)` term dominates; it does not say which,
//! and the answer decides what is worth optimizing. An `O(N²)` term with a small prefactor can
//! sit well below an `O(N)` term with a large one across the whole range anybody runs.

use am1_rs::divide_conquer::{run_divide_conquer, DcOptions};
use am1_rs::math::Vec3;
use am1_rs::{Am1Options, Am1Parameters, Atom, Molecule};

const ANG: f64 = 1.0 / 0.529167;

/// `n` water molecules on a cubic grid at 4.0 Å, which is far enough that nothing clashes.
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
#[ignore = "profiling: run with --ignored --nocapture"]
fn where_the_time_goes() {
    let params = Am1Parameters::standard().unwrap();
    let opts = Am1Options {
        e_tol: 1.0e-8,
        p_tol: 1.0e-7,
        max_scf: 200,
        multipole_cutoff: std::env::var("AM1_MULTIPOLE")
            .ok()
            .and_then(|v| v.parse().ok()),
        ..Am1Options::default()
    };
    let dc = DcOptions {
        core_size: 8,
        buffer_radius: 12.0,
        ..DcOptions::default()
    };

    eprintln!("  waters  atoms |  wall(s)   iters |  per-phase seconds");
    for n in [27usize, 64, 125, 216, 343] {
        let molecule = water_cluster(n);
        let nat = molecule.atoms.len();
        let start = std::time::Instant::now();
        let r = run_divide_conquer(&molecule, &params, &opts, &dc).unwrap();
        let wall = start.elapsed().as_secs_f64();
        eprintln!("  {n:6} {nat:6} | {wall:8.2}  {:6} |", r.iterations);
        // The accumulator is cleared by `report`, so this has to be the only caller — see
        // `crate::timing`.
        am1_rs::timing::report(&format!("divide-and-conquer, {nat} atoms"));
    }
}

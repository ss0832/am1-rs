// SPDX-License-Identifier: GPL-3.0-or-later

//! How many SCF iterations divide-and-conquer needs as the system grows.
//!
//! Diagnostic, not a correctness test: the scaling benchmark spent an implausibly long time on a
//! 1029-atom cluster, and the question is whether the cost per iteration grew or the iteration
//! *count* did. Those have completely different fixes, and a wall-clock number cannot tell them
//! apart.
//!
//! ```text
//! cargo test --release --test dc_convergence_probe -- --ignored --nocapture
//! ```

use std::time::Instant;

use am1_rs::divide_conquer::{run_divide_conquer, DcOptions};
use am1_rs::fermi::Filling;
use am1_rs::{Am1Options, Am1Parameters, Atom, Molecule, Vec3};

const ANG: f64 = 1.0 / 0.529167;

/// See `tests/scaling.rs` for why this is 4.0 Å and not ice's 2.76: these molecules get
/// pseudo-random orientations, so at ice spacing hydrogens end up pointing at each other and the
/// structure is clashed — which is what made the SCF fail at 1029 atoms in the first place.
fn water_cube(side: usize) -> Molecule {
    let spacing = 4.0 * ANG;
    let mut atoms = Vec::new();
    let mut placed = 0usize;
    for i in 0..side {
        for j in 0..side {
            for k in 0..side {
                let centre = Vec3::new(i as f64, j as f64, k as f64) * spacing;
                let t = (placed as f64) * 2.399_963_2;
                let (s, c) = t.sin_cos();
                let u = Vec3::new(c, s, 0.35).normalized();
                let w = Vec3::new(-s, c, 0.0).normalized();
                atoms.push(Atom {
                    z: 8,
                    position: centre,
                });
                atoms.push(Atom {
                    z: 1,
                    position: centre + u * (0.9584 * ANG),
                });
                atoms.push(Atom {
                    z: 1,
                    position: centre + (u * -0.2440 + w * 0.9698) * (0.9584 * ANG),
                });
                placed += 1;
            }
        }
    }
    Molecule::new(atoms)
}

/// How the subsystem-size spread depends on how the core count factorizes.
///
/// Pure geometry — no SCF — so it runs in seconds and can be swept freely. The scaling benchmark
/// showed `Σ n_α³` per atom oscillating by a factor of 2.8 rather than settling, with the largest
/// subsystem going 297, 302, 344, **233**, 391 AOs across cube sides 5–9. The dip at side 8 is
/// the clue: 1536 atoms at `core_size = 12` is exactly 128 cores, a power of two, so recursive
/// bisection lays them out as a regular 8×4×4 grid of near-cubic boxes. 2187 atoms wants 183
/// cores, which no sequence of binary splits tiles evenly, so some boxes come out elongated — and
/// an elongated core has far more atoms within a buffer radius of it than a compact one of the
/// same atom count.
#[test]
#[ignore = "diagnostic; run with --ignored"]
fn subsystem_size_spread_against_core_count() {
    use am1_rs::divide_conquer::{build_subsystems, partition_atoms};

    let params = Am1Parameters::standard().unwrap();
    println!("\n  side  atoms  cores | core atoms   subsystem AOs (min/mean/max)  max/mean");
    println!("  {}", "-".repeat(76));

    for side in [5usize, 6, 7, 8, 9] {
        let molecule = water_cube(side);
        let basis = am1_rs::basis::Basis::build(&molecule, &params).unwrap();
        let cores = partition_atoms(&molecule, 12);
        let subs = build_subsystems(&molecule, &basis, &cores, 11.0);

        let sizes: Vec<usize> = subs.iter().map(|s| s.nao()).collect();
        let core_sizes: Vec<usize> = cores.iter().map(|c| c.len()).collect();
        let mean = sizes.iter().sum::<usize>() as f64 / sizes.len() as f64;
        let max = *sizes.iter().max().unwrap();
        println!(
            "  {side:4}  {:5}  {:5} | {:2}–{:2}        {:4} / {mean:6.0} / {max:4}          {:5.2}",
            molecule.atoms.len(),
            cores.len(),
            core_sizes.iter().min().unwrap(),
            core_sizes.iter().max().unwrap(),
            sizes.iter().min().unwrap(),
            max as f64 / mean,
        );
    }
}

/// Which knob rescues the 1029-atom case.
///
/// The size sweep shows a cliff — 14, 14, 14 iterations and then 300-and-fails — not a drift, so
/// the cause is not a tolerance that scales badly with system size. A cliff at a fixed geometry
/// family points at the electronic structure: as the system grows, the density of subsystem
/// levels near the common chemical potential grows with it, so a small shift in μ moves many
/// fractional occupations at once and the density can slosh between two patterns instead of
/// settling.
///
/// If that is what is happening, more smearing (which flattens the occupation's response to μ)
/// or slower mixing should fix it, and a deeper DIIS history should not. `max_scf` is kept low so
/// the failures are cheap.
#[test]
#[ignore = "diagnostic; run with --ignored"]
fn what_fixes_the_large_case() {
    let params = Am1Parameters::standard().unwrap();
    let molecule = water_cube(7);
    println!(
        "\n  {} atoms\n  smearing  mixing |  iters  converged     wall",
        molecule.atoms.len()
    );
    println!("  {}", "-".repeat(52));

    for (kt, mixing) in [
        (0.05, 0.4), // the current default: known to fail here
        (0.20, 0.4), // more smearing
        (0.50, 0.4), // much more smearing
        (0.05, 0.1), // slower mixing instead
        (0.20, 0.1), // both
    ] {
        let start = Instant::now();
        let dc = run_divide_conquer(
            &molecule,
            &params,
            &Am1Options::default(),
            &DcOptions {
                core_size: 12,
                buffer_radius: 11.0,
                filling: Filling::Fermi { kt },
                e_tol: 1.0e-7,
                p_tol: 1.0e-6,
                max_scf: 60,
                mixing,
                ..DcOptions::default()
            },
        )
        .unwrap();
        println!(
            "  {kt:8.2}  {mixing:6.2} |  {:5}  {:9}  {:7.1}s",
            dc.iterations,
            dc.converged,
            start.elapsed().as_secs_f64()
        );
    }
}

#[test]
#[ignore = "diagnostic; run with --ignored"]
fn iteration_count_against_system_size() {
    let params = Am1Parameters::standard().unwrap();
    println!("\n  side  atoms |  subs  largest  iters  converged   wall    s/iter");
    println!("  {}", "-".repeat(66));

    for side in [4usize, 5, 6, 7] {
        let molecule = water_cube(side);
        let nat = molecule.atoms.len();
        let start = Instant::now();
        let dc = run_divide_conquer(
            &molecule,
            &params,
            &Am1Options::default(),
            &DcOptions {
                core_size: 12,
                buffer_radius: 11.0,
                filling: Filling::Fermi { kt: 0.05 },
                e_tol: 1.0e-7,
                p_tol: 1.0e-6,
                max_scf: 300,
                mixing: 0.4,
                ..DcOptions::default()
            },
        )
        .unwrap();
        let wall = start.elapsed().as_secs_f64();
        println!(
            "  {side:4}  {nat:5} |  {:4}  {:7}  {:5}  {:9}   {wall:6.1}s  {:6.2}s",
            dc.subsystems,
            dc.largest_subsystem_aos,
            dc.iterations,
            dc.converged,
            wall / dc.iterations as f64,
        );
    }
}

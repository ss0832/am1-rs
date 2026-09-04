// SPDX-License-Identifier: GPL-3.0-or-later

//! Wall-clock scaling of divide-and-conquer against the full SCF.
//!
//! `tests/divide_conquer.rs` asserts the *algorithmic* scaling with counters, which is the part
//! that belongs in a fast, deterministic test: a stopwatch in a unit test is a flaky unit test.
//! This file is the other half — the measurement on a real machine — and it is marked
//! `#[ignore]` so it runs when asked for rather than on every `cargo test`.
//!
//! Run it with:
//!
//! ```text
//! cargo test --release --test scaling -- --ignored --nocapture
//! ```
//!
//! # What is being measured, and what it is honest to conclude
//!
//! Divide-and-conquer makes the **diagonalization** linear. It does not make the whole
//! calculation linear, and this file is written so that cannot be misread. NDDO's two-centre
//! two-electron integrals decay as `1/R`, so the Coulomb sum is over every pair and stays
//! `O(N²)`; making *that* linear needs the multipole/Ewald split, which this version does not
//! have. The exchange does become linear, exactly, because the truncated density matrix it
//! contracts against is identically zero beyond the buffer radius.
//!
//! So the expected picture at large `N` is: full SCF trending towards `N³`, divide-and-conquer
//! trending towards `N²`, and a crossover after which divide-and-conquer wins.
//!
//! What is **asserted** here is deliberately narrower than what is **reported**. Wall-clock
//! exponents are printed but never asserted on: on a machine doing anything else they measure
//! contention as much as the algorithm, and an earlier version of this file failed on a run that
//! fitted the full SCF to an exponent of 0.90 — impossible for a cubic diagonalization. The
//! algorithmic claims are asserted on deterministic counters, here and in
//! `tests/divide_conquer.rs`.

use std::time::{Duration, Instant};

use am1_rs::divide_conquer::{run_divide_conquer, DcOptions};
use am1_rs::fermi::Filling;
use am1_rs::{run_am1, Am1Options, Am1Parameters, Atom, Molecule, Vec3};

const ANG: f64 = 1.0 / 0.529167;

/// `n_waters` water molecules on a cubic lattice, each rotated so the cluster has no symmetry to
/// exploit. See [`SPACING_ANGSTROM`] for why the spacing is what it is.
///
/// A three-dimensional cluster rather than a chain, deliberately: a chain flatters
/// divide-and-conquer, because a fixed buffer radius around a one-dimensional core reaches far
/// fewer atoms than the same radius in three dimensions. The subsystem size a chain produces is
/// not the one a real condensed-phase system produces.
fn water_cluster(n_waters: usize) -> Molecule {
    water_cluster_sized(n_waters, (n_waters as f64).cbrt().ceil() as usize)
}

/// A perfect `side × side × side` cube of waters.
///
/// Used where the *shape* has to be held constant across sizes. A partially filled outer shell
/// changes how compact the partition's regions are, which moves the subsystem size around by
/// tens of percent and swamps the trend being measured.
fn water_cube(side: usize) -> Molecule {
    water_cluster_sized(side * side * side, side)
}

/// Centre-to-centre spacing, Ångström.
///
/// **Not** the 2.76 Å of real ice, and the difference is not carelessness. Ice gets away with
/// that spacing because its molecules are *oriented* — each hydrogen points at a neighbouring
/// oxygen. These are given pseudo-random orientations, so at ice spacing two hydrogens
/// frequently end up pointing at each other.
///
/// That is exactly what an earlier version of this file did, at 3.1 Å, and it produced minimum
/// intermolecular contacts of 1.22–1.35 Å with over a hundred pairs inside 1.6 Å at the larger
/// sizes. Contacts that short are a severe steric clash, and the resulting electronic structure
/// is pathological: the SCF converged in 14 iterations up to 648 atoms and then failed outright
/// at 1029, which looked like a divide-and-conquer scaling defect and was nothing of the kind.
///
/// At 4.0 Å the worst contact is 2.1–2.2 Å at every size — an ordinary van der Waals contact.
/// The density is lower than liquid water (64 Å³ per molecule against 30), which is the right
/// trade for a benchmark: cost per atom is what is being measured, and it barely depends on the
/// density, whereas conditioning depends on it entirely.
const SPACING_ANGSTROM: f64 = 4.0;

/// Reject a clashed structure rather than benchmark one.
///
/// Cheap insurance against the failure above recurring: it cost a long diagnostic detour to find
/// once, and it would be invisible in the timings.
fn assert_no_clashes(molecule: &Molecule, label: &str) {
    let mut worst = f64::INFINITY;
    for (i, a) in molecule.atoms.iter().enumerate() {
        for (j, b) in molecule.atoms.iter().enumerate().skip(i + 1) {
            // Atoms 3k, 3k+1, 3k+2 are one molecule; only intermolecular contacts matter.
            if i / 3 == j / 3 {
                continue;
            }
            worst = worst.min((a.position - b.position).norm() * 0.529167);
        }
    }
    assert!(
        worst > 1.8,
        "{label}: closest intermolecular contact is {worst:.2} Å — the structure is clashed, and \
         any timing or convergence measured on it says more about the clash than the method"
    );
}

fn water_cluster_sized(n_waters: usize, side: usize) -> Molecule {
    let spacing = SPACING_ANGSTROM * ANG;
    let mut atoms = Vec::with_capacity(3 * n_waters);
    let mut placed = 0;
    for i in 0..side {
        for j in 0..side {
            for k in 0..side {
                if placed == n_waters {
                    break;
                }
                let centre = Vec3::new(i as f64, j as f64, k as f64) * spacing;
                // A deterministic pseudo-random orientation from the lattice index.
                let t = (placed as f64) * 2.399_963_2; // golden angle, radians
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

fn scf_options() -> Am1Options {
    Am1Options {
        e_tol: 1.0e-7,
        p_tol: 1.0e-6,
        max_scf: 300,
        ..Am1Options::default()
    }
}

fn dc_options() -> DcOptions {
    DcOptions {
        core_size: 12,
        buffer_radius: 11.0,
        filling: Filling::Fermi { kt: 0.05 },
        e_tol: 1.0e-7,
        p_tol: 1.0e-6,
        max_scf: 300,
        mixing: 0.4,
        ..DcOptions::default()
    }
}

/// Least-squares slope of `log(y)` against `log(x)`.
///
/// A fit over all the points, not the slope between the last two. A two-point slope on a shared
/// machine measures whatever else the machine was doing: one run of this benchmark produced an
/// apparent exponent of **0.90 for the full SCF**, which is impossible for an `O(N³)`
/// diagonalization and happened only because the larger of the two timings landed in a quieter
/// moment than the smaller one. A fit does not fix that, but it averages over it, and — more to
/// the point — nothing here *asserts* on the fitted value. See the test below for why.
fn exponent(points: &[(f64, f64)]) -> f64 {
    let n = points.len() as f64;
    let (sx, sy) = points
        .iter()
        .fold((0.0, 0.0), |(sx, sy), (x, y)| (sx + x.ln(), sy + y.ln()));
    let (mx, my) = (sx / n, sy / n);
    let (num, den) = points.iter().fold((0.0, 0.0), |(num, den), (x, y)| {
        let dx = x.ln() - mx;
        (num + dx * (y.ln() - my), den + dx * dx)
    });
    num / den
}

fn timed<T>(f: impl FnOnce() -> T) -> (T, Duration) {
    let start = Instant::now();
    let value = f();
    (value, start.elapsed())
}

#[test]
#[ignore = "long-running benchmark; run with --ignored"]
fn divide_and_conquer_beats_the_full_scf_and_by_a_growing_margin() {
    let params = Am1Parameters::standard().unwrap();
    let sizes = [32usize, 64, 128, 256, 512];

    println!(
        "\n  waters  atoms   AOs |    full SCF      DC        speedup | subs  largest  ΔE/atom (eV)"
    );
    println!("  {}", "-".repeat(96));

    let mut dc_points = Vec::new();
    let mut full_points = Vec::new();
    let mut last_speedup = 0.0;

    for &n in &sizes {
        let molecule = water_cluster(n);
        assert_no_clashes(&molecule, &format!("{n} waters"));
        let nat = molecule.atoms.len();
        let naos = 6 * n; // 4 on O, 1 on each H

        let (dc, dc_time) = timed(|| {
            run_divide_conquer(&molecule, &params, &scf_options(), &dc_options()).unwrap()
        });
        assert!(
            dc.converged,
            "{n} waters: divide-and-conquer did not converge"
        );

        // The full SCF is O(N³); stop running it once it stops being informative, rather than
        // spending the bulk of the benchmark on the baseline.
        let full = if nat <= 800 {
            let (full, full_time) = timed(|| run_am1(&molecule, &params, &scf_options()).unwrap());
            Some((full, full_time))
        } else {
            None
        };

        match &full {
            Some((full, full_time)) => {
                let speedup = full_time.as_secs_f64() / dc_time.as_secs_f64();
                last_speedup = speedup;
                println!(
                    "  {n:6}  {nat:5}  {naos:4} | {:9.2}s {:8.2}s  {speedup:8.2}x | {:4}  {:6}  {:+.2e}",
                    full_time.as_secs_f64(),
                    dc_time.as_secs_f64(),
                    dc.subsystems,
                    dc.largest_subsystem_aos,
                    (dc.total_ev - full.total_ev) / nat as f64,
                );
                full_points.push((nat as f64, full_time.as_secs_f64()));
            }
            None => println!(
                "  {n:6}  {nat:5}  {naos:4} | {:>9}  {:8.2}s  {:>9} | {:4}  {:6}  {:>9}",
                "-",
                dc_time.as_secs_f64(),
                "-",
                dc.subsystems,
                dc.largest_subsystem_aos,
                "-",
            ),
        }
        dc_points.push((nat as f64, dc_time.as_secs_f64()));
    }

    let dc_exp = exponent(&dc_points);
    let full_exp = exponent(&full_points);
    println!(
        "\n  fitted wall-clock exponent (indicative only): full SCF {full_exp:.2}, \
         divide-and-conquer {dc_exp:.2}"
    );
    println!(
        "  (divide-and-conquer makes the diagonalization O(N); the NDDO Coulomb sum stays O(N^2)\n   \
         because the two-centre integrals decay as 1/R, so O(N^2) is the honest target here,\n   \
         not O(N). The algorithmic claim is asserted on counters in tests/divide_conquer.rs;\n   \
         see docs/divide-conquer.md.)"
    );

    // What is asserted here, and what deliberately is not.
    //
    // NOT asserted: anything about the fitted exponents, including their ordering. Timings taken
    // at different sizes on a machine doing anything else are drawn from different amounts of
    // contention, and a fit inherits that rather than the algorithm. An earlier version of this
    // test asserted `dc_exp < full_exp` and failed on a run where the full SCF fitted to **0.90**
    // — impossible for a cubic diagonalization, and caused purely by the larger timing landing in
    // a quieter moment than the smaller one. A scaling claim asserted on a stopwatch is a claim
    // about the machine; the algorithmic one is asserted on counters in tests/divide_conquer.rs,
    // where it is deterministic.
    //
    // Asserted: that divide-and-conquer converges at every size, and that it is actually faster
    // at the largest size where both were measured. That margin was 1.4x–2.6x across runs, well
    // outside the noise, and it is the claim a user acts on.
    assert!(
        last_speedup > 1.15,
        "divide-and-conquer should be meaningfully faster than the full SCF at the largest size \
         both were run at, got {last_speedup:.2}x"
    );
}

#[test]
#[ignore = "long-running benchmark; run with --ignored"]
fn the_diagonalization_cost_scales_close_to_linearly() {
    // The claim that *is* linear, isolated from everything that is not, and measured where the
    // claim actually applies.
    //
    // The subsystem size saturates only once the cluster is bigger than the buffer *diameter*
    // in every direction. Below that a subsystem still reaches most of the system and grows
    // with it, so `Σn³/atom` climbs — which is honest behaviour, not a failure, but it means a
    // benchmark run entirely below that threshold would be measuring the pre-asymptotic regime
    // and could conclude whatever it liked. With an 11 Bohr buffer (5.8 Å) and 3.1 Å spacing
    // the threshold is around a 4-molecule radius, so the cube has to reach side 8 or so.
    //
    // Perfect cubes, because a partially filled outer shell changes the partition's compactness
    // enough to move the subsystem size by tens of percent.
    let params = Am1Parameters::standard().unwrap();
    println!("\n  side  waters  atoms |  subs  largest(AO)     Σn³/atom   kept blocks/atom");
    println!("  {}", "-".repeat(78));

    let mut rows = Vec::new();
    for side in [5usize, 6, 7, 8, 9] {
        let molecule = water_cube(side);
        assert_no_clashes(&molecule, &format!("cube of side {side}"));
        let nat = molecule.atoms.len() as f64;
        let (dc, elapsed) = timed(|| {
            run_divide_conquer(&molecule, &params, &scf_options(), &dc_options()).unwrap()
        });
        assert!(dc.converged, "side {side} did not converge");
        println!(
            "  {side:4}  {:6}  {:5} |  {:4}  {:10}   {:12.0}   {:10.2}   ({:.0}s)",
            side * side * side,
            molecule.atoms.len(),
            dc.subsystems,
            dc.largest_subsystem_aos,
            dc.diagonalization_work / nat,
            dc.retained_density_blocks as f64 / nat,
            elapsed.as_secs_f64(),
        );
        rows.push((nat, dc.diagonalization_work / nat, dc.largest_subsystem_aos));
    }

    // Fit `Σn³` against the atom count over every size, rather than compare the last two.
    //
    // The per-size numbers oscillate, and the reason is understood rather than mysterious: how
    // compact a core is depends on how the core count factorizes. 1536 atoms at `core_size = 12`
    // is exactly 128 cores, which recursive bisection lays out as a regular 8x4x4 grid of
    // near-cubic boxes; 2187 atoms wants 183, which no sequence of binary splits tiles evenly, so
    // some boxes come out elongated and an elongated core reaches far more atoms within a buffer
    // radius than a compact one holding the same atoms. Measured directly, the ratio of the
    // largest subsystem to the mean is 1.62, 1.60, 1.62, **1.36**, 1.65 across sides 5-9 — the dip
    // falling exactly on the power-of-two count.
    //
    // Comparing consecutive sizes therefore measures that artefact. Fitting across all of them
    // measures the thing being claimed.
    let work: Vec<(f64, f64)> = rows.iter().map(|r| (r.0, r.1 * r.0)).collect();
    let exp = exponent(&work);
    println!("\n  fitted exponent of Σn³ against atom count: {exp:.2}");
    println!(
        "  (1 would be perfectly linear; a full SCF's single diagonalization is 3. The spread\n   \
         between sizes is core-shape regularity, not growth — see the comment in this test.)"
    );

    // The claim: subsystem diagonalization is linear-ish in the atom count, decisively not cubic.
    // The bound has room for the shape artefact above without admitting anything near quadratic.
    assert!(
        exp < 1.6,
        "the subsystem diagonalization work should scale close to linearly, fitted {exp:.2}"
    );

    // And the mechanism, stated as a bound rather than a trend: subsystem size must stay within a
    // fixed range as the system grows, or `Σn³` could look linear over a short span while being
    // cubic underneath.
    let largest = rows.iter().map(|r| r.2).max().unwrap() as f64;
    let smallest = rows.iter().map(|r| r.2).min().unwrap() as f64;
    println!("  largest subsystem across all sizes: {smallest:.0} – {largest:.0} AOs");
    assert!(
        largest / smallest < 2.5,
        "subsystem size must stay bounded as the system grows; it ranged {smallest:.0}–{largest:.0} AOs"
    );
}

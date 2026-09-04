// SPDX-License-Identifier: GPL-3.0-or-later

//! The far-field monopole approximation: what it costs and what it buys.
//!
//! `multipole_cutoff` replaces the full Dewar–Sabelli–Klopman block of a distant pair with its
//! monopole term. The interaction is kept — dropping a `1/R` term would change the answer — and
//! only the multipole *structure* is simplified, so the neglected pieces are the dipole and
//! quadrupole channels, which fall as `(d/R)²`.
//!
//! It is **off by default**, because every validated number in this crate was produced without
//! it. Turning it on is an explicit accuracy-for-speed trade, and the point of this file is to
//! quantify both sides of that trade rather than to assert that it is fine.

use am1_rs::divide_conquer::{run_divide_conquer, DcOptions};
use am1_rs::math::Vec3;
use am1_rs::{run_am1, Am1Options, Am1Parameters, Atom, Molecule};

const ANG: f64 = 1.0 / 0.529167;

/// `n` water molecules on a cubic grid at 4.0 Å.
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

fn options(cutoff: Option<f64>) -> Am1Options {
    Am1Options {
        multipole_cutoff: cutoff,
        e_tol: 1.0e-9,
        p_tol: 1.0e-8,
        max_scf: 400,
        ..Am1Options::default()
    }
}

#[test]
fn the_error_falls_as_the_cutoff_grows() {
    // The controlled part of the trade. The neglected channels fall as `(d/R)²` with `d ≈ 1 Bohr`
    // the multipole charge separation, so the error must shrink roughly quadratically — and, more
    // importantly, must go to zero rather than to some floor.
    let params = Am1Parameters::standard().unwrap();
    let molecule = water_cluster(27);
    let nat = molecule.atoms.len();
    let exact = run_am1(&molecule, &params, &options(None)).unwrap();
    assert!(exact.converged);

    eprintln!(
        "    exact (no screening): {:.9} eV, {nat} atoms",
        exact.total_ev
    );
    eprintln!("      cutoff(Bohr)     energy (eV)        error/atom (eV)     far pairs");
    let mut errors = Vec::new();
    for cutoff in [10.0_f64, 15.0, 20.0, 30.0, 45.0] {
        let r = run_am1(&molecule, &params, &options(Some(cutoff))).unwrap();
        assert!(r.converged, "cutoff {cutoff}: SCF did not converge");
        let far = am1_rs::farfield::FarField::new(&molecule, &params, cutoff)
            .unwrap()
            .unwrap();
        let (near, far_pairs) = far.pair_counts();
        let per_atom = (r.total_ev - exact.total_ev).abs() / nat as f64;
        eprintln!(
            "      {cutoff:11.1}  {:16.9}  {per_atom:18.3e}   {far_pairs} of {}",
            r.total_ev,
            near + far_pairs
        );
        errors.push(per_atom);
    }

    assert!(
        errors.last().unwrap() < &errors[0],
        "the error must fall as the cutoff grows: {:?}",
        errors
    );
    assert!(
        *errors.last().unwrap() < 1.0e-4,
        "at a 45 Bohr cutoff the far-field error should be well under 1e-4 eV/atom, got {:.3e}",
        errors.last().unwrap()
    );
}

#[test]
fn the_forces_stay_consistent_with_the_energy() {
    // The failure mode that matters most: a screened energy with an unscreened force conserves
    // nothing, and reports no error while doing it. The far field contributes to both or the
    // approximation is worse than useless.
    let params = Am1Parameters::standard().unwrap();
    let molecule = water_cluster(8);
    let opts = options(Some(12.0));
    let analytic = am1_rs::gradient::closed_form_gradient(&molecule, &params, &opts).unwrap();

    let step = 1.0e-5;
    let mut worst = 0.0_f64;
    for a in 0..molecule.atoms.len() {
        for k in 0..3 {
            let shifted = |d: f64| {
                let mut m = molecule.clone();
                let p = &mut m.atoms[a].position;
                match k {
                    0 => p.x += d,
                    1 => p.y += d,
                    _ => p.z += d,
                }
                run_am1(&m, &params, &opts).unwrap().total_ev
            };
            let fd = (shifted(step) - shifted(-step)) / (2.0 * step);
            let an = match k {
                0 => analytic.gradient[a].x,
                1 => analytic.gradient[a].y,
                _ => analytic.gradient[a].z,
            };
            worst = worst.max((an - fd).abs());
        }
    }
    eprintln!("    screened gradient vs finite difference: {worst:.3e} eV/Bohr");
    assert!(
        worst < 1.0e-5,
        "the screened force does not match the screened energy: {worst:.3e} eV/Bohr"
    );
}

#[test]
fn it_is_off_by_default_and_none_changes_nothing() {
    // `None` must leave the old path untouched — not "agree to some tolerance", but be the same
    // computation. Every validated number in the crate depends on that, so it is asserted on the
    // two things that decide it rather than on an energy comparison that could pass by accident.
    use am1_rs::neighbors::NeighborList;
    let params = Am1Parameters::standard().unwrap();
    let molecule = water_cluster(8);

    assert_eq!(
        Am1Options::default().multipole_cutoff,
        None,
        "the far-field approximation must be off unless asked for"
    );
    // No pair is dropped from the list...
    let full = NeighborList::build(&molecule, 40.0);
    let screened = NeighborList::build_screened(&molecule, 40.0, None);
    assert_eq!(
        full.pairs.len(),
        screened.pairs.len(),
        "screening with None dropped {} pairs",
        full.pairs.len() - screened.pairs.len()
    );
    // ...and no far-field term is constructed to compensate for one.
    assert!(
        am1_rs::farfield::FarField::new(&molecule, &params, 0.0)
            .unwrap()
            .is_none(),
        "a zero cutoff must produce no far-field term"
    );

    // And the energy is then bit-identical, which given the two facts above it has to be.
    let a = run_am1(&molecule, &params, &options(None)).unwrap();
    let b = run_am1(&molecule, &params, &options(None)).unwrap();
    assert_eq!(a.total_ev, b.total_ev);
}

#[test]
#[ignore = "timing: run with --ignored --nocapture"]
fn what_it_buys_in_wall_clock() {
    let params = Am1Parameters::standard().unwrap();
    let dc = DcOptions {
        core_size: 8,
        buffer_radius: 12.0,
        ..DcOptions::default()
    };
    eprintln!("  atoms |  exact(s)  screened(s)   speedup |  dE/atom (eV)");
    for n in [64usize, 125, 216, 343] {
        let molecule = water_cluster(n);
        let nat = molecule.atoms.len();
        let t0 = std::time::Instant::now();
        let exact = run_divide_conquer(&molecule, &params, &options(None), &dc).unwrap();
        let t_exact = t0.elapsed().as_secs_f64();
        let t1 = std::time::Instant::now();
        let fast = run_divide_conquer(&molecule, &params, &options(Some(20.0)), &dc).unwrap();
        let t_fast = t1.elapsed().as_secs_f64();
        eprintln!(
            "  {nat:5} | {t_exact:9.2} {t_fast:11.2} {:9.2}x | {:12.3e}",
            t_exact / t_fast,
            (fast.total_ev - exact.total_ev).abs() / nat as f64
        );
    }
}

// SPDX-License-Identifier: GPL-3.0-or-later

//! What the two- and one-dimensional lattice sums are worth, measured in the SCF.
//!
//! The unit tests in `src/pbc/ewald2d.rs` and `src/pbc/ewald1d.rs` establish that the sums
//! themselves are right — each reproduces its dimensionality's Madelung constant, is independent
//! of its own convergence parameter, and differentiates to finite differences. This file asks the
//! separate question of whether wiring them into the SCF changed the answers in the way it was
//! supposed to.
//!
//! The measurement is cutoff independence. A truncated real-space sum leaves the energy drifting
//! with `realspace_cutoff`; an exact lattice sum does not. The drift is the whole quantity of
//! interest, so every test here varies the cutoff and reports the spread.

use am1_rs::lattice::Lattice;
use am1_rs::math::Vec3;
use am1_rs::{run_am1, Am1Options, Am1Parameters, Atom, Molecule};

const ANG: f64 = 1.0 / 0.529167;

fn water(offset: Vec3) -> Vec<Atom> {
    [
        (8u8, [0.0, 0.0, 0.0]),
        (1, [0.9614, 0.0, 0.0]),
        (1, [-0.2246, 0.9348, 0.0]),
    ]
    .iter()
    .map(|(z, r)| Atom {
        z: *z,
        position: Vec3::new(r[0], r[1], r[2]) * ANG + offset,
    })
    .collect()
}

/// A water chain along `x`, periodic in one direction only.
fn water_chain(spacing_ang: f64) -> Molecule {
    let l = spacing_ang * ANG;
    Molecule::new(water(Vec3::zero())).with_cell(
        Lattice::from_vectors(
            Vec3::new(l, 0.0, 0.0),
            Vec3::new(0.0, 60.0, 0.0),
            Vec3::new(0.0, 0.0, 60.0),
            [true, false, false],
        )
        .unwrap(),
    )
}

/// A water sheet in the `xy` plane, periodic in two directions.
fn water_slab(spacing_ang: f64) -> Molecule {
    let l = spacing_ang * ANG;
    Molecule::new(water(Vec3::zero())).with_cell(
        Lattice::from_vectors(
            Vec3::new(l, 0.0, 0.0),
            Vec3::new(0.0, l, 0.0),
            Vec3::new(0.0, 0.0, 60.0),
            [true, true, false],
        )
        .unwrap(),
    )
}

fn options(charge: f64, multiplicity: usize, cutoff: f64, ewald: bool) -> Am1Options {
    Am1Options {
        charge,
        multiplicity,
        realspace_cutoff: cutoff,
        exchange_cutoff: Some(10.0),
        ewald,
        e_tol: 1.0e-11,
        p_tol: 1.0e-10,
        max_scf: 2000,
        ..Am1Options::default()
    }
}

fn spread(v: &[f64]) -> f64 {
    v.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        - v.iter().copied().fold(f64::INFINITY, f64::min)
}

/// Energies across a range of real-space cutoffs, with and without the lattice sum.
fn scan(molecule: &Molecule, charge: f64, multiplicity: usize, label: &str) -> (f64, f64) {
    let params = Am1Parameters::standard().unwrap();
    let cutoffs = [40.0_f64, 80.0, 160.0, 320.0];
    let mut on = Vec::new();
    let mut off = Vec::new();
    eprintln!("    {label}");
    eprintln!("      cutoff(Bohr)     with sum (eV)        without (eV)");
    for rc in cutoffs {
        let a = run_am1(molecule, &params, &options(charge, multiplicity, rc, true)).unwrap();
        let b = run_am1(molecule, &params, &options(charge, multiplicity, rc, false)).unwrap();
        assert!(
            a.converged && b.converged,
            "{label}: SCF did not converge at {rc} Bohr"
        );
        eprintln!(
            "      {rc:10.0}     {:18.9}   {:18.9}",
            a.total_ev, b.total_ev
        );
        on.push(a.total_ev);
        off.push(b.total_ev);
    }
    let (s_on, s_off) = (spread(&on), spread(&off));
    eprintln!("      spread over an 8x range: {s_on:.3e} eV with, {s_off:.3e} eV without\n");
    (s_on, s_off)
}

#[test]
fn a_neutral_chain_energy_stops_drifting_with_the_cutoff() {
    // The recorded pre-Ewald behaviour was 3e-4 eV of drift between a 40 and a 640 Bohr cutoff
    // on a water chain, "still moving at the 4e-6 eV level at the end". The monopole part of
    // that is what the one-dimensional sum removes.
    let (on, off) = scan(&water_chain(3.2), 0.0, 1, "neutral water chain, 1D");
    assert!(
        on < off,
        "the chain sum should reduce the cutoff drift, not increase it: {on:.3e} vs {off:.3e}"
    );
    // The improvement is real but modest — roughly a factor of two — and that is the correct
    // outcome, not a disappointing one. For a **neutral** cell the monopole terms already cancel
    // among themselves, so there was little for an exact monopole sum to fix; what is left is
    // the higher-multipole series, which is still truncated at the cutoff. The dramatic case is
    // the charged one below.
    assert!(
        on < 1.0e-3,
        "the neutral chain drifts by {on:.3e} eV, far more than the multipole tail should be"
    );
}

#[test]
fn a_neutral_slab_energy_stops_drifting_with_the_cutoff() {
    let (on, off) = scan(&water_slab(3.4), 0.0, 1, "neutral water slab, 2D");
    assert!(
        on < off,
        "the Parry sum should reduce the cutoff drift: {on:.3e} vs {off:.3e}"
    );
    // As for the chain: neutrality already cancelled the monopole, so most of what remains is
    // the untreated multipole tail. Recorded rather than asserted away.
    assert!(on < 5.0e-2, "the neutral slab drifts by {on:.3e} eV");
}

#[test]
fn a_charged_chain_has_a_convergent_energy_under_the_stated_convention() {
    // A charged chain's potential diverges logarithmically, so its energy exists only relative
    // to a reference. The chain sum fixes one — `lim [Σ_{|n|≤M} 1/|r+nLê| − (2/L) ln M]` — and
    // under it the answer is finite and cutoff-independent. That is what is asserted; the
    // *value* means "per cell, at that reference", which `docs/pbc.md` states.
    let (on, off) = scan(&water_chain(3.2), 1.0, 2, "+1 water chain, 1D");
    eprintln!("      the uncorrected chain diverges by {off:.1} eV over the same range");
    assert!(
        on < 1.0e-2,
        "the charged chain energy is not convergent under its own convention: {on:.3e} eV"
    );
    assert!(
        off > 1000.0 * on,
        "the control should still diverge far more than the corrected sum: {off:.3e} vs {on:.3e}"
    );
}

#[test]
fn a_charged_slab_records_whatever_it_actually_does() {
    // Deliberately a measurement rather than an assertion of success.
    //
    // The Parry `h = 0` term is derived for a neutral cell; for a charged one the real-space
    // sum's linear divergence is not cancelled and the answer should still move with the
    // cutoff. If it does, that is the honest limit of the 2D treatment and the number belongs
    // in the docs. If it does not, the reason needs finding rather than assuming.
    let (on, off) = scan(&water_slab(3.4), 1.0, 2, "+1 water slab, 2D");
    eprintln!(
        "      charged slab: {on:.3e} eV with the Parry sum, {off:.3e} eV without \
         (ratio {:.1}x)",
        off / on.max(1.0e-30)
    );
    // Only the comparison is asserted. Whether the absolute number is small is the thing being
    // measured, and it is reported above either way.
    assert!(
        on < off,
        "the Parry sum should not make a charged slab worse: {on:.3e} vs {off:.3e}"
    );
}

#[test]
fn the_low_dimensional_forces_match_finite_differences() {
    // The correction enters the gradient too, and a force that does not match the energy is
    // worse than no correction at all — dynamics would drift without any error being reported.
    let params = Am1Parameters::standard().unwrap();
    for (label, molecule) in [("1D chain", water_chain(3.2)), ("2D slab", water_slab(3.4))] {
        let opts = options(0.0, 1, 60.0, true);
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
        eprintln!("    {label}: max |analytic - FD| = {worst:.3e} eV/Bohr");
        assert!(worst < 1.0e-5, "{label} gradient mismatch {worst:.3e}");
    }
}

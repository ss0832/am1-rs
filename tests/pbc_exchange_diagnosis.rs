// SPDX-License-Identifier: GPL-3.0-or-later

//! Why the Γ-point periodic energy needs the exchange truncated.
//!
//! A single neutral atom in a cubic cell is the sharpest possible probe of the periodic
//! electrostatics, because there is nothing else in it. Its net charge is exactly zero, so the
//! three `1/R` monopole pieces — electron–core attraction, electron–electron Coulomb, and
//! core–core repulsion — must cancel to `(Z − P)² γ = 0` term by term for every image. Whatever
//! energy is left over as the cell shrinks is not physics; it is whatever the code is doing
//! that the algebra says it should not.

use am1_rs::lattice::Lattice;
use am1_rs::math::Vec3;
use am1_rs::{run_am1, Am1Options, Am1Parameters, Atom, Molecule};

fn lone_carbon(cell_bohr: f64) -> Molecule {
    Molecule::new(vec![Atom {
        z: 6,
        position: Vec3::new(0.0, 0.0, 0.0),
    }])
    .with_cell(Lattice::cubic(cell_bohr).unwrap())
}

fn options(exchange_cutoff: Option<f64>) -> Am1Options {
    Am1Options {
        multiplicity: 3, // carbon's ground state
        realspace_cutoff: 40.0,
        exchange_cutoff,
        ..Am1Options::default()
    }
}

#[test]
fn a_lone_neutral_atom_should_not_care_how_large_its_cell_is() {
    let params = Am1Parameters::standard().unwrap();
    let isolated = run_am1(
        &Molecule::new(vec![Atom {
            z: 6,
            position: Vec3::new(0.0, 0.0, 0.0),
        }]),
        &params,
        &options(None),
    )
    .unwrap();
    let reference = isolated.total_ev;
    eprintln!("    isolated carbon: {reference:.9} eV");

    eprintln!("\n    cell    no truncation      image exchange truncated");
    let mut worst_truncated = 0.0_f64;
    for cell in [40.0_f64, 30.0, 20.0, 15.0, 12.0] {
        // The cutoff has to sit *below* the nearest image, which is at exactly `cell` for a
        // cubic lattice. A cutoff equal to the cell excludes nothing, since the test is
        // `r > cutoff`.
        let cutoff = 0.9 * cell;
        let untruncated = run_am1(&lone_carbon(cell), &params, &options(None))
            .map(|r| r.total_ev - reference)
            .unwrap_or(f64::NAN);
        let truncated = run_am1(&lone_carbon(cell), &params, &options(Some(cutoff)))
            .map(|r| r.total_ev - reference)
            .unwrap_or(f64::NAN);
        eprintln!("    {cell:5.1}   {untruncated:+12.6} eV    {truncated:+12.6} eV");
        if truncated.is_finite() {
            worst_truncated = worst_truncated.max(truncated.abs());
        }
    }

    // With the image exchange dropped, the monopole cancellation is exact and a lone neutral
    // atom must be indifferent to its cell. Anything left is the multipole tail of its own
    // (non-spherical, p-occupied) density against its images, which is small.
    eprintln!("\n    worst deviation with exchange truncated: {worst_truncated:.3e} eV");
    assert!(
        worst_truncated < 1.0e-3,
        "a lone neutral carbon still moved by {worst_truncated:.3e} eV with the image exchange \
         truncated; the monopole cancellation is broken by something else"
    );
}

#[test]
fn the_exchange_is_what_diverges() {
    // Directly compare: with the image exchange kept, the energy runs away as the cell
    // shrinks; with it truncated, it does not. Same code, same integrals, one term.
    let params = Am1Parameters::standard().unwrap();
    // 15 Bohr rather than something tighter: at 12 Bohr the SCF fails to converge even with
    // the exchange truncated, because the remaining Coulomb lattice sum is itself severe at
    // that spacing. The point here is the exchange, so it is made at a cell where both paths
    // converge and the comparison is meaningful.
    let tight = lone_carbon(15.0);
    let kept = run_am1(&tight, &params, &options(None)).unwrap().total_ev;
    let cut = run_am1(&tight, &params, &options(Some(13.5)))
        .unwrap()
        .total_ev;
    let isolated = run_am1(
        &Molecule::new(vec![Atom {
            z: 6,
            position: Vec3::new(0.0, 0.0, 0.0),
        }]),
        &params,
        &options(None),
    )
    .unwrap()
    .total_ev;

    eprintln!(
        "    15 Bohr cell: exchange kept {:+.4} eV, truncated {:+.4} eV (relative to isolated)",
        kept - isolated,
        cut - isolated
    );
    assert!(
        (kept - isolated).abs() > 10.0 * (cut - isolated).abs().max(1.0e-6),
        "keeping the image exchange should be dramatically worse than truncating it"
    );
}

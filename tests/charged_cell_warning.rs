// SPDX-License-Identifier: GPL-3.0-or-later

//! What the charged-cell warning says, against what a charged cell actually does.
//!
//! The text raised through every surface (a `RuntimeWarning` in ASE, a line in both CLIs) said
//! that no compensating background is applied "because Ewald summation is not implemented", that
//! "THE TOTAL ENERGY IS NOT CONVERGED", and quoted a swing of −331 eV to +72 eV across a range of
//! real-space cutoffs.
//!
//! Ewald summation has been implemented since 0.2.0, in all three dimensionalities, and is on by
//! default. The quoted numbers are the pre-Ewald measurement. So the warning described a version
//! of the code that no longer existed, and it told users their converged 3D energies were
//! meaningless.
//!
//! This file measures the cutoff dependence that the warning is *about*, so the replacement text
//! is a statement of what was measured rather than another thing that will go stale.

use am1_rs::lattice::Lattice;
use am1_rs::math::Vec3;
use am1_rs::pbc::{run_pbc_scf, PbcOptions};
use am1_rs::{Am1Parameters, Atom, Molecule};

const ANG: f64 = 1.0 / 0.529167;

fn water(shift: Vec3) -> Vec<Atom> {
    [
        (8u8, [0.0, 0.0, 0.0]),
        (1, [0.9614, 0.0, 0.0]),
        (1, [-0.2246, 0.9348, 0.0]),
    ]
    .iter()
    .map(|(z, r)| Atom {
        z: *z,
        position: Vec3::new(r[0], r[1], r[2]) * ANG + shift,
    })
    .collect()
}

fn cubic_cell(a_angstrom: f64) -> Molecule {
    let a = a_angstrom * ANG;
    Molecule::new(water(Vec3::zero())).with_cell(
        Lattice::from_vectors(
            Vec3::new(a, 0.0, 0.0),
            Vec3::new(0.0, a, 0.0),
            Vec3::new(0.0, 0.0, a),
            [true, true, true],
        )
        .unwrap(),
    )
}

fn chain(n: usize, spacing_ang: f64) -> Molecule {
    let step = spacing_ang * ANG;
    let mut atoms = Vec::new();
    for k in 0..n {
        atoms.extend(water(Vec3::new(step * k as f64, 0.0, 0.0)));
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

fn opts(charge: f64, multiplicity: usize, cutoff: f64, ewald: bool) -> PbcOptions {
    PbcOptions {
        charge,
        multiplicity,
        realspace_cutoff: cutoff,
        exchange_cutoff: Some(12.0),
        ewald,
        e_tol: 1.0e-10,
        p_tol: 1.0e-9,
        max_scf: 800,
        ..PbcOptions::default()
    }
}

fn spread(values: &[f64]) -> f64 {
    let hi = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let lo = values.iter().copied().fold(f64::INFINITY, f64::min);
    hi - lo
}

#[test]
fn ewald_is_what_makes_a_charged_three_dimensional_energy_converge() {
    let params = Am1Parameters::standard().unwrap();
    let cell = cubic_cell(8.0);
    let cutoffs = [20.0, 40.0, 80.0, 130.0];

    let sweep = |ewald: bool| -> Vec<f64> {
        cutoffs
            .iter()
            .map(|&rc| {
                run_pbc_scf(&cell, &params, &opts(1.0, 2, rc, ewald))
                    .unwrap()
                    .total_ev
            })
            .collect()
    };
    let with = sweep(true);
    let without = sweep(false);

    eprintln!("    +1 water cell, 8 A cube, cutoffs {cutoffs:?} Bohr:");
    eprintln!(
        "      with Ewald   : {:?}  spread {:.3} eV",
        with.iter().map(|v| format!("{v:.3}")).collect::<Vec<_>>(),
        spread(&with)
    );
    eprintln!(
        "      without Ewald: {:?}  spread {:.1} eV",
        without
            .iter()
            .map(|v| format!("{v:.1}"))
            .collect::<Vec<_>>(),
        spread(&without)
    );

    // The claim the warning must not contradict: with Ewald on — the default — a charged 3D cell
    // is converged to well under an eV across a 6.5x range of cutoff, where without it the energy
    // moves by hundreds.
    assert!(
        spread(&with) < 1.0,
        "with Ewald the charged cell moved {:.3} eV across the cutoff range",
        spread(&with)
    );
    assert!(
        spread(&without) > 100.0,
        "without Ewald it should move by hundreds of eV; got {:.1}",
        spread(&without)
    );
}

#[test]
fn a_charged_cell_still_warns_and_the_text_matches_the_dimensionality() {
    let params = Am1Parameters::standard().unwrap();

    let bulk = run_pbc_scf(&cubic_cell(8.0), &params, &opts(1.0, 2, 40.0, true)).unwrap();
    let wire = run_pbc_scf(&chain(3, 3.2), &params, &opts(1.0, 2, 40.0, true)).unwrap();
    let neutral = run_pbc_scf(&cubic_cell(8.0), &params, &opts(0.0, 1, 40.0, true)).unwrap();

    assert!(
        neutral.charged_cell_warning.is_none(),
        "a neutral cell must not warn"
    );
    let bulk_text = bulk.charged_cell_warning.clone().expect("3D warns");
    let wire_text = wire.charged_cell_warning.clone().expect("1D warns");
    eprintln!("    3D: {bulk_text}");
    eprintln!("    1D: {wire_text}");

    // The two cases are genuinely different and the text must say so: in 3D the tin-foil Ewald
    // sum defines the energy, in 1D the neutralizing background's placement is a convention this
    // crate has not chosen, so the absolute energy is not defined at all.
    assert!(
        bulk_text != wire_text,
        "3D and 1D charged cells need different warnings; both said:\n{bulk_text}"
    );
    assert!(
        !bulk_text.contains("not implemented"),
        "the 3D warning still claims Ewald is not implemented:\n{bulk_text}"
    );
    for text in [&bulk_text, &wire_text] {
        assert!(
            text.is_ascii(),
            "CLI output has to encode under cp932 and the C locale:\n{text}"
        );
    }
}

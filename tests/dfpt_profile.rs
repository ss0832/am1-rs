// SPDX-License-Identifier: GPL-3.0-or-later

//! Where the time goes in a DFPT phonon calculation.
//!
//! Ignored by default: it is a measurement, not an assertion. It exists because the periodic
//! response is the most expensive thing this crate does — the structures in the phonon battery
//! took 30 to 90 seconds each — and because guessing which part is the expensive one has a poor
//! record here. The Bloch sum turned out to be 41 % of a periodic SCF, more than the
//! eigendecompositions it feeds, and nothing about reading the code suggested that.
//!
//! Run with:
//!
//! ```text
//! AM1_TIMING=1 cargo test --test dfpt_profile -- --ignored --nocapture
//! ```

use am1_rs::lattice::Lattice;
use am1_rs::math::Vec3;
use am1_rs::params::Am1Parameters;
use am1_rs::pbc::dfpt::{frequencies_dfpt_with, DfptOptions};
use am1_rs::pbc::kpoints::{KMesh, KPoint};
use am1_rs::pbc::PbcOptions;
use am1_rs::system::{Atom, Molecule};

const ANG: f64 = am1_rs::constants::ANGSTROM_TO_BOHR;

/// Zincblende AlP: a 3D cell whose SCF and CPSCF both converge quickly, so the measurement is of
/// the contraction rather than of a solver that did not finish.
///
/// Rutile GeO2 was the first choice and is not usable here: its CPSCF stops at the 200-iteration
/// cap with a residual of 8.2e-8 against a 1e-10 tolerance, after 354 seconds. That is worth
/// knowing on its own -- the response solve is the stiff part of a phonon run on a dense oxide --
/// but a profile of a run that failed measures the wrong thing.
fn alp() -> Molecule {
    let a = 5.4635 * ANG;
    let cell = Lattice::from_vectors(
        Vec3::new(0.0, a / 2.0, a / 2.0),
        Vec3::new(a / 2.0, 0.0, a / 2.0),
        Vec3::new(a / 2.0, a / 2.0, 0.0),
        [true; 3],
    )
    .unwrap();
    Molecule::new(vec![
        Atom {
            z: 13,
            position: Vec3::zero(),
        },
        Atom {
            z: 15,
            position: Vec3::new(a / 4.0, a / 4.0, a / 4.0),
        },
    ])
    .with_cell(cell)
}

#[test]
#[ignore]
fn where_the_time_goes_in_a_dfpt_phonon() {
    let molecule = alp();
    let params = Am1Parameters::standard().unwrap();
    let options = PbcOptions {
        kmesh: KMesh::MonkhorstPack([3, 3, 3]),
        fold_time_reversal: false,
        max_scf: 400,
        ..Default::default()
    };
    // Interleaved A/B in one process, best of three: a background load that arrives partway
    // through hits both arms. Cross-run wall clock on this machine moves by 2.5x on unchanged
    // arithmetic, which is more than the effect being measured.
    let (mut sparse, mut dense) = (f64::INFINITY, f64::INFINITY);
    let mut reference: Option<Vec<f64>> = None;
    for _ in 0..3 {
        for (force_dense, slot) in [(false, &mut sparse), (true, &mut dense)] {
            let dfpt = DfptOptions {
                dense_contraction: Some(force_dense),
                ..DfptOptions::default()
            };
            let t = std::time::Instant::now();
            let freqs = frequencies_dfpt_with(&molecule, &params, &options, &dfpt, KPoint::gamma())
                .unwrap();
            *slot = slot.min(t.elapsed().as_secs_f64());
            match &reference {
                None => reference = Some(freqs),
                Some(r) => {
                    // The bound is loose *in cm^-1* on purpose. The two routes agree on the
                    // dynamical matrix to 1e-15 (tests/pbc_dfpt_contraction.rs); what is compared
                    // here is its square root, and the acoustic modes at gamma sit at zero where
                    // d(omega)/d(lambda) = 1/(2 omega) has no bound. A 1e-15 difference in an
                    // eigenvalue therefore shows as a microhertz-scale difference in a mode that
                    // is nominally zero, which is arithmetic and not disagreement. The optic
                    // modes, where the derivative is finite, match to 1e-12.
                    let worst = r
                        .iter()
                        .zip(&freqs)
                        .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
                    assert!(worst < 1.0e-3, "the two routes disagree by {worst:e} cm^-1");
                }
            }
        }
    }
    let freqs = reference.unwrap();
    eprintln!(
        "AlP gamma: sparse {:.2} s, dense {:.2} s ({:.2}x)   modes {}",
        sparse,
        dense,
        sparse / dense,
        freqs
            .iter()
            .map(|f| format!("{f:.0}"))
            .collect::<Vec<_>>()
            .join(" ")
    );
    am1_rs::timing::report(&format!("dfpt, {} atoms", molecule.atoms.len()));
    assert_eq!(freqs.len(), 3 * molecule.atoms.len());
}

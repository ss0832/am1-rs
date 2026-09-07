// SPDX-License-Identifier: GPL-3.0-or-later

//! The per-element reference tables, checked against the one thing they claim: that they are
//! indexed by **atomic number**.
//!
//! This is not a style point. MOPAC's own tabulations skip La–Yb (57–70) entirely — the
//! lanthanides have no NDDO parameters, so its arrays are indexed by *parameter slot* and
//! everything above caesium sits fourteen places too low. Transcribed as-is into a Z-indexed
//! array, mercury's mass 200.59 lands at index 66 and `MASS[80]` is left at `0.0`.
//!
//! A zero mass is not a wrong frequency. Every vibrational quantity divides by `√(M_a M_b)`, so
//! it is a division by zero: the dynamical matrix fills with `±∞` and the failure surfaces as
//! "eigendecomposition failed" from a phonon calculation, three layers from the table. Mercury is
//! the only element above Z = 56 that AM1 parameterizes, which is why this stayed hidden until a
//! fluorite HgF₂ phonon run.
//!
//! Hence two kinds of test here: anchors at the elements either side of the gap, which pin the
//! indexing directly, and a structural check over the whole table, which catches a shift
//! introduced anywhere rather than only at the places someone thought to anchor.

use am1_rs::data_tables::{atomic_mass, EHEAT_KCAL, MASS, N_P, N_S, QN};
use am1_rs::math::Vec3;
use am1_rs::params::Am1Parameters;
use am1_rs::NddoMethod;

/// Atomic number → mass, at the elements that bracket the lanthanide gap.
///
/// Caesium and barium are below it and were always right; lanthanum is the first element the gap
/// swallowed; lutetium is what used to sit at index 57; mercury is the only one of these that any
/// parameter set can actually reach, and radon is the last entry, where a shift would run off the
/// end of the array into the zero padding.
#[test]
fn masses_are_indexed_by_atomic_number_across_the_lanthanide_gap() {
    let anchors = [
        (55, 132.91), // Cs — below the gap
        (56, 137.33), // Ba — below the gap
        (57, 138.91), // La — the first element MOPAC's slot indexing omits
        (70, 173.05), // Yb — the last one it omits
        (71, 174.97), // Lu — what used to occupy index 57
        (72, 178.49), // Hf
        (79, 196.97), // Au
        (80, 200.59), // Hg — the one element above Ba that AM1 parameterizes
        (82, 207.2),  // Pb
        (86, 222.0),  // Rn — the last entry
    ];
    for (z, expected) in anchors {
        assert!(
            (MASS[z] - expected).abs() < 0.02,
            "MASS[{z}] = {} but element {z} weighs {expected} amu",
            MASS[z]
        );
    }
}

/// Every element in the table, checked by a property instead of a value.
///
/// Atomic mass rises with atomic number, with exactly three exceptions in this range — Ar/K,
/// Co/Ni and Te/I, where the heavier element has the lower standard atomic weight. Any
/// transcription shift breaks monotonicity somewhere it should not, so this covers the entries no
/// anchor names.
#[test]
fn atomic_mass_increases_with_atomic_number_except_where_it_famously_does_not() {
    const KNOWN_INVERSIONS: [usize; 3] = [19, 28, 53]; // K after Ar, Ni after Co, I after Te
    for z in 2..MASS.len() {
        if KNOWN_INVERSIONS.contains(&z) {
            assert!(
                MASS[z] < MASS[z - 1],
                "Z={z} is supposed to be one of the three mass inversions"
            );
            continue;
        }
        assert!(
            MASS[z] > MASS[z - 1],
            "MASS[{z}] = {} is not heavier than MASS[{}] = {}; the table has a gap or a shift",
            MASS[z],
            z - 1,
            MASS[z - 1]
        );
    }
}

/// No parameterized element may be missing from any per-element table.
///
/// The failure this guards against is silent in three of the four tables — a wrong `N_S` or
/// `EHEAT` shifts an energy by a plausible amount — and fatal in the fourth. Iterating over what
/// the parameter sets actually cover, rather than a hand-written list, means a newly parameterized
/// element is checked the day it is added.
#[test]
fn every_parameterized_element_has_reference_data() {
    for method in [NddoMethod::Am1, NddoMethod::Rm1] {
        let params = Am1Parameters::for_method(method).unwrap();
        for z in 1u8..=86 {
            if params.element(z).is_err() {
                continue;
            }
            assert!(
                atomic_mass(z).is_ok(),
                "{method:?} parameterizes Z={z} but no atomic mass is tabulated for it"
            );
            assert!(QN[z as usize] > 0, "{method:?} Z={z} has no valence shell");
            assert!(
                N_S[z as usize] + N_P[z as usize] > 0.0,
                "{method:?} Z={z} has an empty valence configuration"
            );
            // ΔH_f is genuinely zero for the noble gases MOPAC does not tabulate, so this asks
            // only that it is not *negative* — a shifted table produces a neighbour's value, and
            // the anchor test below pins the one element where the shift mattered.
            assert!(EHEAT_KCAL[z as usize] >= 0.0);
        }
    }
}

/// Mercury's heat of formation, and the emptiness of the slot its shifted copy used to occupy.
///
/// `EHEAT_KCAL` carried the same fourteen-place shift, but it had been papered over: someone
/// wrote 14.690 into index 80 by hand and left the shifted copy at index 66, so the symptom was
/// invisible for the only element that reaches it while every entry in between was still the
/// wrong atom's value. Dysprosium (66) has no MOPAC value at all, so a nonzero entry there means
/// the shifted copy is back.
#[test]
fn heats_of_formation_are_indexed_by_atomic_number_too() {
    assert!((EHEAT_KCAL[80] - 14.690).abs() < 1e-9, "Hg ΔH_f");
    assert_eq!(EHEAT_KCAL[66], 0.0, "Dy has no MOPAC heat of formation");
    assert!((EHEAT_KCAL[72] - 148.0).abs() < 1e-9, "Hf ΔH_f");
    assert!((EHEAT_KCAL[82] - 46.62).abs() < 1e-9, "Pb ΔH_f");
}

/// A missing mass is reported where it is missing, not where it divides.
#[test]
fn an_untabulated_mass_is_an_error_rather_than_an_infinity() {
    let err = atomic_mass(0).unwrap_err().to_string();
    assert!(
        err.contains("atomic mass") && err.contains("Z=0"),
        "unhelpful message: {err}"
    );
}

/// The end-to-end version: a mercury vibrational calculation produces finite frequencies.
///
/// The table test above would pass with the division still unguarded, and the guard would pass
/// with the table still wrong. This is the case that was actually broken — mass-weighting a
/// Hessian that contains mercury — and it needs both.
#[test]
fn mercury_vibrations_are_finite() {
    use am1_rs::system::{Atom, Molecule};
    // HgF₂, linear, at a roughly sensible bond length (positions are in Bohr). The numbers are
    // not the point; being numbers at all is.
    let b = 1.93 * am1_rs::constants::ANGSTROM_TO_BOHR;
    let molecule = Molecule::new(vec![
        Atom {
            z: 80,
            position: Vec3::zero(),
        },
        Atom {
            z: 9,
            position: Vec3::new(0.0, 0.0, b),
        },
        Atom {
            z: 9,
            position: Vec3::new(0.0, 0.0, -b),
        },
    ]);
    let params = Am1Parameters::standard().unwrap();
    let options = am1_rs::scf::Am1Options::default();
    let modes = am1_rs::hessian::vibrational_analysis(&molecule, &params, &options, 1.0e-3)
        .expect("a mercury vibrational analysis must not fail in the mass weighting");
    assert!(
        modes.all_frequencies_cm.iter().all(|f| f.is_finite()),
        "non-finite frequencies: {:?}",
        modes.all_frequencies_cm
    );
    // Linear triatomic: 5 rigid-body directions, 4 vibrations.
    assert_eq!(modes.rigid_body_count, 5);
    assert_eq!(modes.frequencies_cm.len(), 4);
}

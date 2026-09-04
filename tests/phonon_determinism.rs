// SPDX-License-Identifier: GPL-3.0-or-later

//! The phonon path must give the same answer twice.
//!
//! # The defect this pins
//!
//! `ForceConstants::blocks` is a `HashMap<ImageOffset, Matrix>`, and three float sums iterated it
//! directly: the Bloch sum `D(q) = Σ_T Φ(T) e^{iq·T}`, the acoustic-sum-rule residual, and the
//! acoustic-sum-rule *correction*. Rust seeds each `HashMap` instance from a thread-local counter,
//! so two maps built from the same insertions in the same process iterate in different orders — and
//! floating-point addition is not associative.
//!
//! The result was a phonon spectrum that changed between identical calls. Measured on a water
//! crystal in a 4.5 Å cube with a 2×1×1 supercell: five identical `lo_to_frequencies` calls in one
//! process agreed on four and differed by **1798 cm⁻¹** on the fifth, one O–H stretch collapsing
//! into a near-zero mode. The periodic SCF underneath was bit-identical every time — same energy to
//! the last digit, same 115 iterations — which is what located the problem in the phonon assembly
//! rather than in the electronic structure.
//!
//! The third sum is the one that made it visible: the correction is *subtracted* from the on-site
//! block, so an order-dependent value there changes `Φ` itself and every `D(q)` built from it.
//!
//! # Why it is asserted bit-for-bit
//!
//! A tolerance would let the defect back in. The claim is not "the spectrum is stable to 1e-8" —
//! it is that the same input produces the same output, which is a property of the code and not of
//! the conditioning of any particular crystal. The system below is deliberately the ill-conditioned
//! one that exposed it: a compressed water cell whose spectrum has soft and near-degenerate modes,
//! where a last-bit difference in `Φ` is enough to reorder eigenvectors.

use am1_rs::lattice::Lattice;
use am1_rs::math::Vec3;
use am1_rs::pbc::phonon::{build_supercell, ForceConstants};
use am1_rs::pbc::KPoint;
use am1_rs::{Am1Options, Am1Parameters, Atom, Molecule};

const ANG: f64 = 1.0 / 0.529167;

/// A water molecule in a 4.5 Å cube — compressed, polar, and ill-conditioned on purpose.
fn water_crystal() -> Molecule {
    let a = 4.5 * ANG;
    let atoms = [
        (8u8, [0.0, 0.0, 0.0]),
        (1, [0.9584, 0.0, 0.0]),
        (1, [-0.2400, 0.9278, 0.0]),
    ]
    .iter()
    .map(|(z, r)| Atom {
        z: *z,
        position: Vec3::new(r[0], r[1], r[2]) * ANG,
    })
    .collect();
    Molecule::new(atoms).with_cell(
        Lattice::from_vectors(
            Vec3::new(a, 0.0, 0.0),
            Vec3::new(0.0, a, 0.0),
            Vec3::new(0.0, 0.0, a),
            [true, true, true],
        )
        .unwrap(),
    )
}

fn options() -> Am1Options {
    Am1Options {
        realspace_cutoff: 40.0,
        exchange_cutoff: Some(20.0),
        e_tol: 1.0e-11,
        p_tol: 1.0e-10,
        max_scf: 800,
        ..Am1Options::default()
    }
}

fn spectrum(enforce_asr: bool) -> Vec<f64> {
    let params = Am1Parameters::standard().unwrap();
    let primitive = water_crystal();
    let mut fc =
        ForceConstants::from_supercell(&primitive, &params, &options(), [2, 1, 1]).unwrap();
    if enforce_asr {
        fc.enforce_acoustic_sum_rule();
    }
    fc.frequencies(KPoint {
        fractional: [0.0, 0.0, 0.0],
        weight: 1.0,
    })
    .unwrap()
}

#[test]
fn repeated_phonon_calculations_are_bit_identical() {
    // Both with and without the sum-rule correction: the correction has its own sum over the same
    // map, and it is the one that feeds back into `Φ`.
    for enforce in [false, true] {
        let first = spectrum(enforce);
        for repeat in 1..5 {
            let again = spectrum(enforce);
            assert_eq!(
                first.len(),
                again.len(),
                "enforce_asr={enforce}: repeat {repeat} returned a different number of modes"
            );
            for (k, (a, b)) in first.iter().zip(&again).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "enforce_asr={enforce}: repeat {repeat}, mode {k}: {a} != {b}. The phonon \
                     assembly summed over a HashMap, whose iteration order differs between \
                     instances in one process."
                );
            }
        }
        eprintln!(
            "    enforce_asr={enforce}: 5 runs bit-identical, {} modes, lowest {:.4} cm^-1",
            first.len(),
            first[0]
        );
    }
}

/// The sum-rule residual is a diagnostic a caller reads before deciding whether to correct, so it
/// must not move either.
#[test]
fn the_acoustic_sum_rule_residual_is_reproducible() {
    let params = Am1Parameters::standard().unwrap();
    let primitive = water_crystal();
    let build = || {
        ForceConstants::from_supercell(&primitive, &params, &options(), [2, 1, 1])
            .unwrap()
            .acoustic_sum_rule_error()
    };
    let first = build();
    for _ in 0..4 {
        assert_eq!(first.to_bits(), build().to_bits());
    }
    eprintln!("    acoustic sum rule residual {first:.3e} eV/Bohr^2, reproducible");
}

/// The supercell builder is upstream of all of it; if it were order-dependent the rest could not
/// be reproducible either. Cheap to assert, and it localizes a future regression.
#[test]
fn the_supercell_is_built_the_same_way_twice() {
    let primitive = water_crystal();
    let a = build_supercell(&primitive, [2, 1, 1]).unwrap();
    let b = build_supercell(&primitive, [2, 1, 1]).unwrap();
    assert_eq!(a.atoms.len(), b.atoms.len());
    for (x, y) in a.atoms.iter().zip(&b.atoms) {
        assert_eq!(x.z, y.z);
        assert_eq!(x.position.x.to_bits(), y.position.x.to_bits());
        assert_eq!(x.position.y.to_bits(), y.position.y.to_bits());
        assert_eq!(x.position.z.to_bits(), y.position.z.to_bits());
    }
}

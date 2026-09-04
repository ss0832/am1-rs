// SPDX-License-Identifier: GPL-3.0-or-later

//! Phonons: `Φ(T)`, `D(q)`, and the band structure.
//!
//! # The test that actually pins the construction down
//!
//! [`the_commensurate_q_fold_back_onto_the_supercell_spectrum`] is the load-bearing one, and it
//! is exact rather than approximate. An `N`-fold supercell has `3·N·nat` vibrational modes at Γ;
//! the same physics described in the primitive cell is `3·nat` modes at each of `N` commensurate
//! `q`. Those two sets must be **identical**, because they are the same eigenproblem in two
//! bases. A wrong atom ordering in the supercell, a mis-assigned translation, a transposed
//! block, or a sign error in the Bloch phase each break it.
//!
//! # A comparison that looks obvious and is wrong
//!
//! It is tempting to check `Σ_T Φ(0,T)` from an `N`-fold supercell against the primitive cell's
//! own Γ Hessian. They do not agree, and should not.
//!
//! Γ on an `N`-fold supercell is equivalent to the primitive cell sampled at `N` **k-points** —
//! that is precisely what `tests/pbc_kpoints.rs` asserts for the energy. The primitive cell's Γ
//! Hessian is the primitive cell at *one* k-point, which is a different (and, for a periodic
//! solid, worse) electronic structure. The two force-constant matrices therefore describe
//! different Hamiltonians. Measured on an H₂ chain, they differ by 0.76 eV/Bohr², which is the
//! size of the k-sampling error and not a bug.

use am1_rs::lattice::Lattice;
use am1_rs::math::Vec3;
use am1_rs::pbc::phonon::{build_supercell, q_path, ForceConstants};
use am1_rs::pbc::KPoint;
use am1_rs::{vibrational_analysis, Am1Options, Am1Parameters, Atom, Molecule};

const ANG: f64 = 1.0 / 0.529167;
/// AM1's own H₂ bond length, 0.6766 Å — measured, not the experimental 0.741 Å.
///
/// Using the experimental value instead puts the molecule well up the repulsive wall (AM1 gives
/// −27.471 eV there against −27.536 eV at its own minimum) and produces a genuine imaginary
/// mode. That is correct physics for a stretched molecule and useless as a test of the phonon
/// machinery.
const H2_BOND: f64 = 0.6766;

fn options() -> Am1Options {
    Am1Options {
        realspace_cutoff: 60.0,
        exchange_cutoff: Some(20.0),
        e_tol: 1.0e-11,
        p_tol: 1.0e-10,
        max_scf: 800,
        ..Am1Options::default()
    }
}

/// A chain of H₂ units along x: one-dimensional, gapped, with real dispersion.
fn hydrogen_chain(a: f64) -> Molecule {
    Molecule::new(vec![
        Atom {
            z: 1,
            position: Vec3::new(0.0, 0.0, 0.0),
        },
        Atom {
            z: 1,
            position: Vec3::new(H2_BOND * ANG, 0.0, 0.0),
        },
    ])
    .with_cell(
        Lattice::from_vectors(
            Vec3::new(a, 0.0, 0.0),
            Vec3::new(0.0, 60.0, 0.0),
            Vec3::new(0.0, 0.0, 60.0),
            [true, false, false],
        )
        .unwrap(),
    )
}

fn gamma() -> KPoint {
    KPoint {
        fractional: [0.0, 0.0, 0.0],
        weight: 1.0,
    }
}

#[test]
fn the_commensurate_q_fold_back_onto_the_supercell_spectrum() {
    // The exact identity, and the sharpest test available. `3·N·nat` modes at Γ of the supercell
    // versus `3·nat` modes at each of the `N` commensurate q — the same eigenproblem in two
    // bases, so the sorted frequency lists must match to numerical precision.
    let params = Am1Parameters::standard().unwrap();
    let opts = options();
    let primitive = hydrogen_chain(6.0);
    let repeats = 4;

    let fc = ForceConstants::from_supercell(&primitive, &params, &opts, [repeats, 1, 1]).unwrap();

    // Route A: the supercell's own Γ spectrum.
    let supercell = build_supercell(&primitive, [repeats, 1, 1]).unwrap();
    let mut direct = vibrational_analysis(&supercell, &params, &opts, 1.0e-3)
        .unwrap()
        .frequencies_cm;
    direct.sort_by(|a, b| a.total_cmp(b));

    // Route B: the union over commensurate q of D(q).
    let mut folded: Vec<f64> = Vec::new();
    for q in fc.commensurate_q() {
        folded.extend(fc.frequencies(q).unwrap());
    }
    folded.sort_by(|a, b| a.total_cmp(b));

    assert_eq!(direct.len(), folded.len(), "mode counts differ");
    let worst = direct
        .iter()
        .zip(&folded)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    eprintln!(
        "    {repeats}x supercell: {} modes at Γ vs {} from {} commensurate q\n    \
         max frequency difference: {worst:.4e} cm⁻¹",
        direct.len(),
        folded.len(),
        repeats
    );
    eprintln!(
        "      supercell Γ : {:?}",
        direct.iter().map(|f| format!("{f:.0}")).collect::<Vec<_>>()
    );
    eprintln!(
        "      folded D(q) : {:?}",
        folded.iter().map(|f| format!("{f:.0}")).collect::<Vec<_>>()
    );
    assert!(
        worst < 1.0e-3,
        "the commensurate q do not reproduce the supercell spectrum: {worst:.3e} cm⁻¹"
    );
}

#[test]
fn the_supercell_solution_has_the_periodicity_of_the_primitive_cell() {
    // What makes the folding above exact: the supercell's force constants must depend only on
    // the *difference* of cell indices. If the supercell SCF had broken the primitive
    // periodicity — a real possibility, since nothing constrains it — `Φ(T)` read off the home
    // cell's rows would not represent the other cells and the unfolding would be meaningless.
    let params = Am1Parameters::standard().unwrap();
    let opts = options();
    let repeats = 3usize;
    let primitive = hydrogen_chain(6.0);
    let supercell = build_supercell(&primitive, [repeats, 1, 1]).unwrap();
    let h = am1_rs::analytic_hessian(&supercell, &params, &opts, 1.0e-3).unwrap();

    let nat = primitive.atoms.len();
    let mut worst = 0.0_f64;
    for shift in 0..repeats {
        for cell_i in 0..repeats {
            let cell_j = (cell_i + shift) % repeats;
            for a in 0..nat {
                for b in 0..nat {
                    for i in 0..3 {
                        for j in 0..3 {
                            let moved = h[(3 * (cell_i * nat + a) + i, 3 * (cell_j * nat + b) + j)];
                            let home = h[(3 * a + i, 3 * (shift * nat + b) + j)];
                            worst = worst.max((moved - home).abs());
                        }
                    }
                }
            }
        }
    }
    eprintln!("    largest departure from primitive-cell periodicity: {worst:.3e} eV/Bohr²");
    // The bound is set by SCF convergence, not by symmetry. Genuine symmetry breaking — the
    // supercell settling into a state the primitive cell cannot represent — would show up as a
    // force-constant difference of order the force constants themselves, which are O(1) here.
    assert!(
        worst < 1.0e-6,
        "the supercell force constants are not translationally periodic: {worst:.3e} eV/Bohr²"
    );
}

#[test]
fn three_acoustic_modes_go_to_zero_at_gamma() {
    // Translating the whole crystal costs nothing, so `D(Γ)` has three zero eigenvalues. The
    // residual before enforcement measures the truncation of `Φ(T)` at the supercell boundary,
    // which is the approximation this construction actually makes.
    let params = Am1Parameters::standard().unwrap();
    let mut fc =
        ForceConstants::from_supercell(&hydrogen_chain(6.0), &params, &options(), [4, 1, 1])
            .unwrap();

    let before = fc.acoustic_sum_rule_error();
    fc.enforce_acoustic_sum_rule();
    let after = fc.acoustic_sum_rule_error();
    let fixed = fc.frequencies(gamma()).unwrap();

    eprintln!(
        "    acoustic sum rule residual: {before:.3e} → {after:.3e} eV/Bohr²\n    \
         Γ frequencies: {:?}",
        fixed.iter().map(|f| format!("{f:.1}")).collect::<Vec<_>>()
    );

    assert!(
        after < 1.0e-10,
        "enforcing the sum rule did not enforce it: {after:.3e}"
    );
    let zeros = fixed.iter().filter(|f| f.abs() < 1.0).count();
    assert!(
        zeros >= 3,
        "expected three zero modes at Γ, got {zeros} in {fixed:?}"
    );
}

#[test]
fn the_frequencies_are_physical_at_a_relaxed_geometry() {
    // A physics check rather than a numerical one, at AM1's own H₂ bond length. The
    // intramolecular stretch of a free H₂ is 4341 cm⁻¹ in AM1; in a chain it shifts but stays in
    // that region, and nothing should be imaginary at a sensible geometry.
    let params = Am1Parameters::standard().unwrap();
    let mut fc =
        ForceConstants::from_supercell(&hydrogen_chain(6.0), &params, &options(), [4, 1, 1])
            .unwrap();
    fc.enforce_acoustic_sum_rule();
    let freq = fc.frequencies(gamma()).unwrap();
    eprintln!(
        "    Γ frequencies (cm⁻¹): {:?}   [free AM1 H₂ stretch = 4341]",
        freq.iter().map(|f| format!("{f:.0}")).collect::<Vec<_>>()
    );

    assert!(
        freq.iter().all(|f| *f > -100.0),
        "an imaginary mode at a relaxed geometry: {freq:?}"
    );
    let highest = freq.last().copied().unwrap_or(0.0);
    assert!(
        (2500.0..6000.0).contains(&highest),
        "the H–H stretch should be in the region of the free molecule's 4341 cm⁻¹, got {highest:.0}"
    );
}

#[test]
fn the_band_structure_disperses() {
    // Along the periodic axis the spectrum must change with q: that dispersion is the whole
    // reason `Φ(T)` is resolved by translation rather than summed. A `D(q)` that ignored the
    // Bloch phases would be flat.
    let params = Am1Parameters::standard().unwrap();
    let mut fc =
        ForceConstants::from_supercell(&hydrogen_chain(6.0), &params, &options(), [4, 1, 1])
            .unwrap();
    fc.enforce_acoustic_sum_rule();

    let path = q_path(&[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], 4);
    let bands = fc.band_structure(&path).unwrap();

    eprintln!("      q_x    lowest   highest  (cm⁻¹)");
    for (q, row) in path.iter().zip(&bands) {
        eprintln!(
            "    {:6.3}  {:8.1}  {:8.1}",
            q.fractional[0],
            row.iter().cloned().fold(f64::INFINITY, f64::min),
            row.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        );
    }

    for (q, row) in path.iter().zip(&bands) {
        assert!(
            row.iter().all(|f| *f > -100.0),
            "imaginary mode at q = {:?}: {row:?}",
            q.fractional
        );
    }
    let spread: f64 = bands[0]
        .iter()
        .zip(bands.last().unwrap())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    eprintln!("    largest Γ → zone-boundary shift: {spread:.1} cm⁻¹");
    assert!(
        spread > 10.0,
        "the bands are flat between Γ and the zone boundary, so the Bloch phases in D(q) are \
         not doing anything"
    );
}

#[test]
fn the_stiff_mode_converges_with_supercell_size() {
    // The convergence knob — and a distinction worth being explicit about.
    //
    // Supercell size controls **two** things at once here, not one. It sets how far `Φ(T)` is
    // resolved before truncation, and — because Γ on an `N`-fold supercell is the primitive cell
    // at `N` k-points — it also sets the k-sampling of the electronic structure the force
    // constants come from. Both converge together, and neither is separable in this construction.
    //
    // So the honest statement is per-mode. The H–H stretch is stiff and well determined, and it
    // converges quickly: it should stop moving. The transverse modes of an H₂ chain are very
    // soft — a few tens of cm⁻¹, and close enough to zero to change sign — so demanding they
    // agree between a 2x and a 4x cell would be demanding convergence of a quantity that is
    // barely determined. They are reported rather than asserted on.
    let params = Am1Parameters::standard().unwrap();
    let chain = hydrogen_chain(6.0);
    let opts = options();

    let mut stretch = Vec::new();
    for repeats in [2usize, 4, 6] {
        let mut fc =
            ForceConstants::from_supercell(&chain, &params, &opts, [repeats, 1, 1]).unwrap();
        fc.enforce_acoustic_sum_rule();
        let f = fc.frequencies(gamma()).unwrap();
        eprintln!(
            "    {repeats}x supercell, Γ: {:?}",
            f.iter().map(|x| format!("{x:.0}")).collect::<Vec<_>>()
        );
        stretch.push(f.last().copied().unwrap());
    }

    let shift = (stretch[2] - stretch[1]).abs();
    eprintln!(
        "    H–H stretch: {:.1} → {:.1} → {:.1} cm⁻¹ (4x → 6x shift {shift:.1})",
        stretch[0], stretch[1], stretch[2]
    );
    assert!(
        shift < 10.0,
        "the stiff mode has not converged with supercell size: {shift:.1} cm⁻¹ between 4x and 6x"
    );
}

#[test]
fn a_supercell_along_a_non_periodic_axis_is_refused() {
    // Repeating a direction with no periodicity is meaningless, and silently producing a
    // duplicated molecule would be worse than an error.
    let params = Am1Parameters::standard().unwrap();
    let err = ForceConstants::from_supercell(&hydrogen_chain(6.0), &params, &options(), [2, 2, 1])
        .unwrap_err();
    let message = err.to_string();
    eprintln!("    {message}");
    assert!(
        message.contains("not periodic"),
        "unhelpful error: {message}"
    );
}

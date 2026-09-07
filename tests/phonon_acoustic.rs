// SPDX-License-Identifier: GPL-3.0-or-later

//! **Three acoustic modes at `q = 0`, exactly.**
//!
//! Translating a crystal rigidly costs no energy, so `D(0)` annihilates the mass-weighted uniform
//! displacement and three of its eigenvalues are zero. That is not an approximation and not a
//! convergence question: it follows from `Σ_T Σ_b Φ_ab(T) = 0`, which
//! [`ForceConstants::enforce_acoustic_sum_rule`] imposes to machine precision.
//!
//! It is also the single cheapest end-to-end check on the whole phonon path, and the one that
//! catches a defect the interpolation tests cannot: `Σ_T Σ_b Φ_ab(T) = 0` is a statement about the
//! **rows** of `Φ`, and `dynamical_matrix` symmetrizes `D` before diagonalizing it. If `Φ(T)` is
//! not the transpose of `Φ(−T)` then the column sums are *not* zero, symmetrizing mixes the two,
//! and the acoustic modes come out at hundreds of wavenumbers with the sum rule still reading
//! `1e-15`. So this file measures both sums, and their difference, rather than only the frequency.

use am1_rs::lattice::{ImageOffset, Lattice};
use am1_rs::math::Vec3;
use am1_rs::pbc::phonon::ForceConstants;
use am1_rs::pbc::KPoint;
use am1_rs::{Am1Options, Am1Parameters, Atom, Molecule};

const ANG: f64 = 1.0 / 0.529167;

/// The tolerances `native.phonons` ships with, not tighter ones.
///
/// Diamond does not reach `1e-11` here, and that is a statement about the model rather than about
/// the SCF: sixteen carbons in a 2×2×2 fcc supercell put neighbouring atoms 1.54 Å apart, where
/// the Slater overlap between an atom and its periodic image is large — and NDDO's working
/// equations assume it is zero. The test uses what a user gets so that a convergence failure here
/// would be one they would also see.
fn options() -> Am1Options {
    Am1Options {
        realspace_cutoff: 40.0,
        exchange_cutoff: Some(20.0),
        e_tol: 1.0e-10,
        p_tol: 1.0e-9,
        max_scf: 800,
        ..Am1Options::default()
    }
}

fn gamma() -> KPoint {
    KPoint {
        fractional: [0.0, 0.0, 0.0],
        weight: 1.0,
    }
}

/// An H₂ chain along z: one periodic direction, an orthogonal cell.
fn hydrogen_chain() -> Molecule {
    Molecule::new(vec![
        Atom {
            z: 1,
            position: Vec3::zero(),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.0, 0.0, 0.6766 * ANG),
        },
    ])
    .with_cell(
        Lattice::from_vectors(
            Vec3::new(60.0, 0.0, 0.0),
            Vec3::new(0.0, 60.0, 0.0),
            Vec3::new(0.0, 0.0, 3.0 * ANG),
            [false, false, true],
        )
        .unwrap(),
    )
}

/// Graphene: two atoms, a hexagonal cell, two periodic directions.
fn graphene() -> Molecule {
    let a = 2.46 * ANG;
    Molecule::new(vec![
        Atom {
            z: 6,
            position: Vec3::zero(),
        },
        Atom {
            z: 6,
            position: Vec3::new(0.0, a / 3.0_f64.sqrt(), 0.0),
        },
    ])
    .with_cell(
        Lattice::from_vectors(
            Vec3::new(a, 0.0, 0.0),
            Vec3::new(-a / 2.0, a * 3.0_f64.sqrt() / 2.0, 0.0),
            Vec3::new(0.0, 0.0, 20.0 * ANG),
            [true, true, false],
        )
        .unwrap(),
    )
}

/// Diamond in its **face-centred cubic primitive cell** — two atoms, three non-orthogonal
/// lattice vectors at 60°. The skew is the point: an orthogonal cell cannot distinguish a
/// translation label that is right from one that is merely close.
fn diamond() -> Molecule {
    let a = 3.567 * ANG;
    Molecule::new(vec![
        Atom {
            z: 6,
            position: Vec3::zero(),
        },
        Atom {
            z: 6,
            position: Vec3::new(a / 4.0, a / 4.0, a / 4.0),
        },
    ])
    .with_cell(
        Lattice::from_vectors(
            Vec3::new(0.0, a / 2.0, a / 2.0),
            Vec3::new(a / 2.0, 0.0, a / 2.0),
            Vec3::new(a / 2.0, a / 2.0, 0.0),
            [true, true, true],
        )
        .unwrap(),
    )
}

/// `Σ_T Σ_b Φ_{ai,bj}` (rows) and `Σ_T Σ_a Φ_{ai,bj}` (columns), largest magnitude of each.
///
/// The library only reports the first. They are equal for a correctly filed `Φ`, and the gap
/// between them is exactly what a mis-filed translation opens up.
fn sum_rules(fc: &ForceConstants) -> (f64, f64) {
    let n = fc.nat;
    let mut rows = 0.0_f64;
    let mut cols = 0.0_f64;
    for a in 0..n {
        for i in 0..3 {
            for j in 0..3 {
                let mut row = 0.0;
                let mut col = 0.0;
                for block in fc.blocks.values() {
                    for b in 0..n {
                        row += block[(3 * a + i, 3 * b + j)];
                        col += block[(3 * b + j, 3 * a + i)];
                    }
                }
                rows = rows.max(row.abs());
                cols = cols.max(col.abs());
            }
        }
    }
    (rows, cols)
}

/// `max |Φ(T) − Φ(−T)ᵀ|`, the symmetry a rigid translation's zero mode depends on.
fn transpose_defect(fc: &ForceConstants) -> f64 {
    let ndof = 3 * fc.nat;
    let mut worst = 0.0_f64;
    for (offset, block) in &fc.blocks {
        let mirror = fc.blocks.get(&ImageOffset {
            n: [-offset.n[0], -offset.n[1], -offset.n[2]],
        });
        for i in 0..ndof {
            for j in 0..ndof {
                let other = mirror.map_or(0.0, |m| m[(j, i)]);
                worst = worst.max((block[(i, j)] - other).abs());
            }
        }
    }
    worst
}

fn check(label: &str, molecule: Molecule, supercell: [usize; 3]) {
    let params = Am1Parameters::standard().unwrap();
    let mut fc = match ForceConstants::from_supercell(&molecule, &params, &options(), supercell) {
        Ok(fc) => fc,
        Err(e) => panic!("{label}: the supercell force constants could not be built: {e}"),
    };
    let (rows_before, cols_before) = sum_rules(&fc);
    let defect_before = transpose_defect(&fc);
    let scale = fc
        .blocks
        .values()
        .flat_map(|m| m.as_slice().iter())
        .fold(0.0_f64, |x, v| x.max(v.abs()));

    fc.enforce_acoustic_sum_rule();
    let (rows, cols) = sum_rules(&fc);
    let defect = transpose_defect(&fc);
    let mut freqs = fc.frequencies(gamma()).unwrap();
    freqs.sort_by(|a, b| a.partial_cmp(b).unwrap());

    eprintln!("  {label} {supercell:?}  |Phi| ~ {scale:.3e} eV/Bohr^2");
    eprintln!("    row sums                 {rows_before:.3e} -> {rows:.3e}");
    eprintln!("    col sums                 {cols_before:.3e} -> {cols:.3e}");
    eprintln!("    max |Phi(T) - Phi(-T)^T| {defect_before:.3e} -> {defect:.3e}");
    eprintln!(
        "    q = 0: {}",
        freqs
            .iter()
            .map(|f| format!("{f:.1}"))
            .collect::<Vec<_>>()
            .join("  ")
    );

    // Three acoustic modes, whatever the periodicity: a rigid translation in any Cartesian
    // direction costs no energy, and a slab translated normal to itself is still that slab.
    let acoustic = &freqs[..3];
    let worst = acoustic.iter().fold(0.0_f64, |m, f| m.max(f.abs()));
    assert!(
        worst < 1.0,
        "{label}: the three acoustic modes at q = 0 are {acoustic:?}, not zero. \
         Row sums read {rows:.3e} and column sums {cols:.3e}; `dynamical_matrix` symmetrizes, \
         so only the average of the two reaches the spectrum."
    );
    // Both halves of the enforcement, after it has run. The *before* numbers are printed rather
    // than asserted: they are the truncation error the cutoffs leave behind, which is a property
    // of the calculation and not something this test gets to demand.
    //
    // The bound is `1e-6` relative, not roundoff, and that is deliberate: the alternating
    // projection has a fixed point at the *antisymmetric* part of the sum-rule violation — a
    // rotational residue the translational rule does not constrain — which on graphene is 6.5e-6
    // against a |Phi| of 18.7. See `enforce_acoustic_sum_rule`. The frequency assertion above is
    // the one that says the residue does not matter; these two say it did not grow.
    assert!(
        cols < 1.0e-6 * scale.max(1.0),
        "{label}: the column sums survived enforcement at {cols:.3e}"
    );
    assert!(
        defect < 1.0e-6 * scale.max(1.0),
        "{label}: Phi(T) is still not the transpose of Phi(-T), off by {defect:.3e}"
    );
}

#[test]
fn a_chain_has_three_acoustic_modes_at_gamma() {
    check("H2 chain", hydrogen_chain(), [1, 1, 3]);
}

#[test]
fn a_sheet_has_three_acoustic_modes_at_gamma() {
    check("graphene", graphene(), [2, 2, 1]);
}

/// **DFPT has to satisfy the same rule**, and until 0.2.3 it did not impose it at all.
///
/// The supercell route corrects `Φ(T)`; the DFPT route computes the response directly at each `q`
/// and so has no `Φ(T)` to correct — but the *ground state* underneath it still carries the
/// real-space and exchange cutoffs, and what they leave behind lands on the acoustic branch.
/// Measured before the correction: wurtzite ZnO came out `−80, −80, 0, 0, 0` at Γ — five
/// near-zero modes where a crystal has three — and rutile GeO₂ `−166`.
///
/// This checks the rule on the matrix rather than on a big spectrum, so it is cheap and says
/// exactly what it means: at `q = 0` the row sums of `C(0)` vanish, therefore a rigid translation
/// is in the null space, therefore three frequencies are zero.
#[test]
fn dfpt_imposes_the_acoustic_sum_rule_at_gamma() {
    let params = Am1Parameters::standard().unwrap();
    let molecule = hydrogen_chain();
    let opts = am1_rs::pbc::PbcOptions {
        kmesh: am1_rs::pbc::KMesh::MonkhorstPack([1, 1, 4]),
        // A `q`-point response relates `k` to `k + q`, which is a different pairing from the
        // `k`/`−k` merge time-reversal folding performs; the solver refuses a folded mesh by name.
        fold_time_reversal: false,
        realspace_cutoff: 40.0,
        exchange_cutoff: Some(20.0),
        e_tol: 1.0e-10,
        p_tol: 1.0e-9,
        max_scf: 800,
        ..am1_rs::pbc::PbcOptions::default()
    };
    let d = am1_rs::pbc::dynamical_matrix_dfpt(&molecule, &params, &opts, gamma()).unwrap();

    let nat = molecule.atoms.len();
    let mut worst = 0.0_f64;
    let mut scale = 0.0_f64;
    for a in 0..nat {
        for i in 0..3 {
            for j in 0..3 {
                let mut re = 0.0;
                for b in 0..nat {
                    let (r, _) = d.get(3 * a + i, 3 * b + j);
                    re += r;
                    scale = scale.max(r.abs());
                }
                worst = worst.max(re.abs());
            }
        }
    }
    eprintln!("    DFPT row sums at q = 0: {worst:.3e} against a scale of {scale:.3e}");
    assert!(
        worst < 1.0e-9 * scale.max(1.0),
        "the DFPT dynamical matrix at q = 0 does not annihilate a rigid translation: {worst:.3e}"
    );

    let mut freqs = am1_rs::pbc::frequencies_dfpt(&molecule, &params, &opts, gamma()).unwrap();
    freqs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    eprintln!(
        "    DFPT q = 0: {}",
        freqs
            .iter()
            .map(|f| format!("{f:.1}"))
            .collect::<Vec<_>>()
            .join("  ")
    );
    // Three modes **at** zero, not "the three lowest are zero". A rigid translation is in the
    // null space whatever else the spectrum does, and this chain is transversely unstable at this
    // spacing — it has two genuinely imaginary modes at −77 cm⁻¹ that sit below the acoustic
    // ones. Requiring the lowest three to vanish would be asserting that the structure is a
    // minimum, which is a different claim and not this test's.
    let zeros = freqs.iter().filter(|f| f.abs() < 1.0).count();
    assert!(
        zeros >= 3,
        "the sum rule guarantees three zero modes; found {zeros} in {freqs:?}"
    );
}

/// The skew-cell case, and the one that fails without minimum-image folding.
#[test]
fn a_skew_primitive_cell_has_three_acoustic_modes_at_gamma() {
    check("diamond (fcc primitive)", diamond(), [2, 2, 2]);
}

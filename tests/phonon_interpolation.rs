// SPDX-License-Identifier: GPL-3.0-or-later

//! **Phonon bands away from the commensurate `q`.**
//!
//! `tests/pbc_phonon.rs` checks the points a supercell represents exactly. Those points cannot see
//! the defect this file is about, and that is why it survived to 0.2.2.
//!
//! A supercell's Γ Hessian is an aliased sum `H_{(0,a),(t,b)} = Σ_N Φ_ab(T_t + N)` over the
//! supercell lattice. Through 0.2.2 the element was filed under `T_t` — the translation the atom
//! list was *built* with, which `supercell_cells` numbers `0, 1, … n−1` along each axis. For an
//! `n`-fold cell that labels the neighbour on one side `+(n−1)` when it physically sits at `−1`.
//!
//! At `q = m/n` the two labels give the same Bloch phase, because `e^{2πi·m(n−1)/n}` and
//! `e^{−2πi·m/n}` differ by `e^{2πi m} = 1`. Every commensurate test therefore passed. Between
//! those points they differ completely: on a three-fold chain at the zone boundary `q = ½` the
//! phase is `+1` under the old labelling and `−1` under the right one.
//!
//! The three properties below are all consequences of filing the block under its **minimum
//! image**, and all of them fail without it.

use am1_rs::lattice::{ImageOffset, Lattice};
use am1_rs::math::Vec3;
use am1_rs::pbc::phonon::ForceConstants;
use am1_rs::pbc::KPoint;
use am1_rs::{Am1Options, Am1Parameters, Atom, Molecule};

const ANG: f64 = 1.0 / 0.529167;
/// AM1's own H₂ bond length; see the note in `tests/pbc_phonon.rs`.
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

fn q_at(fraction: f64) -> KPoint {
    KPoint {
        fractional: [fraction, 0.0, 0.0],
        weight: 1.0,
    }
}

/// `Φ_ab(T) = Φ_ba(−T)`, because the second derivative of the energy does not care which of two
/// atoms is called first.
///
/// This is an exact symmetry of the real force constants, and it is the cheapest possible probe of
/// whether a block is filed under the right translation: with the offsets running `0, 1, 2` on a
/// three-fold cell, `Φ(−1)` does not exist at all and `Φ(2)` has no partner, so the check cannot
/// even be formed — every block is compared against a zero matrix.
#[test]
fn folded_force_constants_are_symmetric_under_t_to_minus_t() {
    let params = Am1Parameters::standard().unwrap();
    let fc = ForceConstants::from_supercell(&hydrogen_chain(6.0), &params, &options(), [3, 1, 1])
        .unwrap();

    let offsets: Vec<[i32; 3]> = {
        let mut v: Vec<[i32; 3]> = fc.blocks.keys().map(|o| o.n).collect();
        v.sort_unstable();
        v
    };
    eprintln!("    translations in Phi(T): {offsets:?}");
    assert!(
        offsets.contains(&[-1, 0, 0]),
        "no negative translation, so the blocks were filed by index rather than by distance"
    );

    let ndof = 3 * fc.nat;
    let mut scale = 0.0_f64;
    let mut worst = 0.0_f64;
    for (offset, block) in &fc.blocks {
        let negated = ImageOffset {
            n: [-offset.n[0], -offset.n[1], -offset.n[2]],
        };
        let mirror = fc.blocks.get(&negated);
        for i in 0..ndof {
            for j in 0..ndof {
                scale = scale.max(block[(i, j)].abs());
                let other = mirror.map_or(0.0, |m| m[(j, i)]);
                worst = worst.max((block[(i, j)] - other).abs());
            }
        }
    }
    eprintln!("    worst |Phi(T) - Phi(-T)^T| = {worst:.3e} against a scale of {scale:.3e}");
    assert!(
        worst < 1.0e-6 * scale.max(1.0),
        "Phi(T) is not the transpose of Phi(-T): off by {worst}"
    );
}

/// **The two labellings agree exactly where the old tests looked, and nowhere else.**
///
/// The 0.2.2 labelling is reconstructed here rather than described: re-file every folded block
/// under `T mod n ∈ [0, n)` and the result is bit-for-bit what `from_supercell` used to build,
/// because folding only moved blocks between translations that are congruent mod `n` and split
/// values that the modulus puts back together.
///
/// Then the same `ForceConstants` machinery is run on both. At the commensurate `q` the spectra
/// are identical to roundoff — which is exactly why every phonon test in 0.2.2 passed. Between
/// them they are not, and the zone boundary of a three-fold cell is where it is worst.
#[test]
fn the_old_labelling_agrees_only_at_the_commensurate_q() {
    let params = Am1Parameters::standard().unwrap();
    let repeats = 3usize;
    let folded =
        ForceConstants::from_supercell(&hydrogen_chain(6.0), &params, &options(), [repeats, 1, 1])
            .unwrap();

    // The 0.2.2 labelling: every translation wrapped into `[0, n)`.
    let mut wrapped = folded.clone();
    wrapped.blocks.clear();
    let ndof = 3 * folded.nat;
    for (offset, block) in &folded.blocks {
        let n = repeats as i32;
        let key = ImageOffset {
            n: [offset.n[0].rem_euclid(n), offset.n[1], offset.n[2]],
        };
        let slot = wrapped
            .blocks
            .entry(key)
            .or_insert_with(|| am1_rs::linalg::Matrix::zeros(ndof, ndof));
        for i in 0..ndof {
            for j in 0..ndof {
                slot[(i, j)] += block[(i, j)];
            }
        }
    }

    // Commensurate: identical.
    for m in 0..repeats {
        let q = q_at(m as f64 / repeats as f64);
        let a = folded.frequencies(q).unwrap();
        let b = wrapped.frequencies(q).unwrap();
        let worst = a
            .iter()
            .zip(&b)
            .fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()));
        eprintln!("    q = {m}/{repeats}: folded vs wrapped differ by {worst:.3e} cm^-1");
        assert!(
            worst < 1.0e-8,
            "the two labellings must agree at a commensurate q, but differ by {worst}"
        );
    }

    // Anywhere else: not.
    let q = q_at(0.5);
    let a = folded.frequencies(q).unwrap();
    let b = wrapped.frequencies(q).unwrap();
    eprintln!("    q = 1/2 folded : {:?}", rounded(&a));
    eprintln!("    q = 1/2 wrapped: {:?}", rounded(&b));
    let worst = a
        .iter()
        .zip(&b)
        .fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()));
    eprintln!("    q = 1/2: they differ by {worst:.1} cm^-1");
    // The intermolecular branches of this chain sit near 75 cm⁻¹, so 10 cm⁻¹ is already a
    // double-digit percentage of the band it corrupts. The decisive evidence is the next
    // assertion, which is qualitative rather than a matter of degree.
    assert!(
        worst > 10.0,
        "the zone boundary should expose the difference, but it is only {worst} cm^-1"
    );

    // Which of the two is right is settled by `Phi(T) = Phi(-T)^T`, not by the size of the
    // disagreement: the wrapped labelling violates it by the full size of the force constant,
    // because it files the `-1` neighbour under `+2` and leaves `Phi(-1)` empty.
    //
    // Note what does *not* settle it: `omega(q) = omega(-q)` holds for **both**. `Phi` is real,
    // so `D(-q) = D(q)*`, and `hermitianize` makes each of them Hermitian, whose spectrum equals
    // its conjugate's. The evenness of the band is therefore blind to how the blocks are filed —
    // worth knowing, because it is the first property one reaches for.
    let asymmetry = |fc: &ForceConstants| -> f64 {
        let mut worst = 0.0_f64;
        for (offset, block) in &fc.blocks {
            let negated = ImageOffset {
                n: [-offset.n[0], -offset.n[1], -offset.n[2]],
            };
            let mirror = fc.blocks.get(&negated);
            for i in 0..ndof {
                for j in 0..ndof {
                    let other = mirror.map_or(0.0, |m| m[(j, i)]);
                    worst = worst.max((block[(i, j)] - other).abs());
                }
            }
        }
        worst
    };
    let folded_asymmetry = asymmetry(&folded);
    let wrapped_asymmetry = asymmetry(&wrapped);
    eprintln!(
        "    |Phi(T) - Phi(-T)^T|:  folded {folded_asymmetry:.3e}, wrapped {wrapped_asymmetry:.3e}"
    );
    assert!(
        wrapped_asymmetry > 1.0e6 * folded_asymmetry.max(1.0e-12),
        "the wrapped labelling should violate Phi(T) = Phi(-T)^T outright"
    );
}

/// `omega(q) = omega(-q)`, and the band is smooth across the zone boundary.
///
/// A necessary condition, not a sufficient one — see the note in
/// [`the_old_labelling_agrees_only_at_the_commensurate_q`] about why it cannot tell the two
/// labellings apart. It is here to catch a *complex* defect: a sign error in the Bloch phase, or
/// a `Φ(T)` that has picked up an imaginary part, would break it.
#[test]
fn the_band_is_even_in_q_and_continuous_at_the_zone_boundary() {
    let params = Am1Parameters::standard().unwrap();
    let fc = ForceConstants::from_supercell(&hydrogen_chain(6.0), &params, &options(), [3, 1, 1])
        .unwrap();

    for fraction in [0.1, 0.25, 0.4, 0.5] {
        let plus = fc.frequencies(q_at(fraction)).unwrap();
        let minus = fc.frequencies(q_at(-fraction)).unwrap();
        let worst = plus
            .iter()
            .zip(&minus)
            .fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()));
        eprintln!("    q = {fraction:>5}: omega(q) vs omega(-q) differ by {worst:.3e} cm^-1");
        assert!(worst < 1.0e-6, "the band is not even in q at {fraction}");
    }

    // Continuity: the zone boundary is a stationary point of a band with inversion symmetry, so
    // approaching it from inside must not jump.
    let inside = fc.frequencies(q_at(0.495)).unwrap();
    let boundary = fc.frequencies(q_at(0.5)).unwrap();
    let jump = inside
        .iter()
        .zip(&boundary)
        .fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()));
    eprintln!("    q = 0.495 -> 0.5 moves the band by {jump:.2} cm^-1");
    assert!(
        jump < 20.0,
        "the band jumps at the zone boundary: {jump} cm^-1"
    );
}

fn rounded(v: &[f64]) -> Vec<f64> {
    v.iter().map(|f| (f * 10.0).round() / 10.0).collect()
}

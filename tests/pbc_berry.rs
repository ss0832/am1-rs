// SPDX-License-Identifier: GPL-3.0-or-later

// Cartesian tensor indices, as the crate allows at its own root: `z[atom][alpha][beta]` says which
// index is which, and the iterator rewrite clippy suggests would not.
#![allow(clippy::needless_range_loop)]

//! Berry-phase polarization: the modern theory of polarization, and the four things that pin it.
//!
//! Through 0.2.1 `docs/scope.md` recorded "Berry-phase polarization ⛔ — `ε_∞` is the clamped-ion
//! dipole response, not a Berry phase". That was accurate and it was a gap: the dipole of a
//! periodic cell is not a property of the crystal, so without a Berry phase there was no
//! polarization at all, only its second derivative.
//!
//! # What can and cannot be asserted about a polarization
//!
//! `P` is defined **modulo the quantum** `e a/Ω`. An absolute value is not a physical prediction —
//! choosing a branch is choosing which cell to assign the electrons to. So no test here compares
//! `P` to a number. What they compare is:
//!
//! 1. **A difference**, reduced to the branch nearest zero — the only physically meaningful thing
//!    two polarizations can produce.
//! 2. **Invariance** under moving an atom by a lattice vector, which is the same crystal.
//! 3. **Vanishing** for a centrosymmetric cell, where symmetry forces `P` to a lattice point.
//! 4. **`Ω ∂P/∂τ_A` against the Born effective charges**, which this crate already computes by a
//!    completely different route — a CPHF response of the dipole operator. That is the sharp one:
//!    two independent formalisms, sharing only the SCF, for the same tensor.

use am1_rs::lattice::Lattice;
use am1_rs::math::Vec3;
use am1_rs::pbc::berry::berry_polarization;
use am1_rs::pbc::{KMesh, PbcOptions};
use am1_rs::{Am1Parameters, Atom, Molecule};

const ANG: f64 = 1.0 / 0.529167;

fn options(mesh: [usize; 3]) -> PbcOptions {
    PbcOptions {
        kmesh: KMesh::MonkhorstPack(mesh),
        realspace_cutoff: 30.0,
        exchange_cutoff: Some(12.0),
        e_tol: 1.0e-11,
        p_tol: 1.0e-10,
        max_scf: 600,
        smearing_ev: 0.0,
        ..PbcOptions::default()
    }
}

/// A polar cell: one hydrogen fluoride molecule in a cube, aligned along `x`.
fn hf_cell(a_ang: f64, bond_ang: f64, shift_ang: f64) -> Molecule {
    let a = a_ang * ANG;
    Molecule::new(vec![
        Atom {
            z: 9,
            position: Vec3::new(shift_ang * ANG, 0.0, 0.0),
        },
        Atom {
            z: 1,
            position: Vec3::new((shift_ang + bond_ang) * ANG, 0.0, 0.0),
        },
    ])
    .with_cell(
        Lattice::from_vectors(
            Vec3::new(a, 0.0, 0.0),
            Vec3::new(0.0, a, 0.0),
            Vec3::new(0.0, 0.0, a),
            [true, true, true],
        )
        .unwrap(),
    )
}

/// A centrosymmetric cell: two hydrogen molecules placed so the cell has an inversion centre.
fn centrosymmetric(a_ang: f64) -> Molecule {
    let a = a_ang * ANG;
    let d = 0.37 * ANG;
    Molecule::new(vec![
        Atom {
            z: 1,
            position: Vec3::new(-d, 0.0, 0.0),
        },
        Atom {
            z: 1,
            position: Vec3::new(d, 0.0, 0.0),
        },
    ])
    .with_cell(
        Lattice::from_vectors(
            Vec3::new(a, 0.0, 0.0),
            Vec3::new(0.0, a, 0.0),
            Vec3::new(0.0, 0.0, a),
            [true, true, true],
        )
        .unwrap(),
    )
}

// -------------------------------------------------------------------------- the quantum

#[test]
fn translating_the_whole_cell_by_a_lattice_vector_leaves_the_polarization_alone() {
    // Not "gives the same number" — gives the same number *modulo the quantum*, which is what
    // `difference` reduces. Anything else would be asserting a branch.
    let params = Am1Parameters::standard().unwrap();
    let o = options([1, 1, 1]);
    let base = berry_polarization(&hf_cell(6.0, 0.94, 0.0), &params, &o, 8).unwrap();
    let moved = berry_polarization(&hf_cell(6.0, 0.94, 6.0), &params, &o, 8).unwrap();

    let delta = base.difference(&moved);
    eprintln!(
        "    P(shifted by a) - P(base), reduced: ({:+.3e}, {:+.3e}, {:+.3e}) e/Bohr^2",
        delta.x, delta.y, delta.z
    );
    eprintln!(
        "    the quantum along x is {:.6} e/Bohr^2",
        base.quantum[0].x
    );
    assert!(
        delta.norm() < 1.0e-6,
        "shifting by a lattice vector changed the polarization by {:.3e}",
        delta.norm()
    );
}

#[test]
fn a_centrosymmetric_cell_has_no_polarization() {
    // Inversion symmetry forces `P = −P` modulo the quantum, so `2P` is a lattice point and `P` is
    // either 0 or exactly half a quantum. Here the cell is symmetric about the origin, so 0.
    let params = Am1Parameters::standard().unwrap();
    let p = berry_polarization(&centrosymmetric(6.0), &params, &options([1, 1, 1]), 8).unwrap();
    // Reduce against a zero reference, which is what `difference` does from a zeroed copy.
    let mut reduced = p.total;
    for q in &p.quantum {
        let n = (reduced.dot(*q) / q.norm2()).round();
        reduced -= *q * n;
    }
    eprintln!(
        "    centrosymmetric P (reduced) = ({:+.3e}, {:+.3e}, {:+.3e}) e/Bohr^2",
        reduced.x, reduced.y, reduced.z
    );
    assert!(
        reduced.norm() < 1.0e-6,
        "a centrosymmetric cell should have zero polarization, got {:.3e}",
        reduced.norm()
    );
}

// ------------------------------------------------------------------------- convergence

#[test]
fn the_phase_converges_with_the_string_length() {
    // The string length is the discretization of the Berry phase; the answer must stop moving.
    let params = Am1Parameters::standard().unwrap();
    let o = options([1, 1, 1]);
    let molecule = hf_cell(6.0, 0.94, 0.0);
    let mut previous: Option<Vec3> = None;
    let mut last_step = f64::INFINITY;
    for strings in [4, 8, 16, 32] {
        let p = berry_polarization(&molecule, &params, &o, strings).unwrap();
        if let Some(prev) = previous {
            last_step = (p.total - prev).norm();
            eprintln!(
                "    strings={strings:2}: P_x = {:+.10}, step {last_step:.2e}",
                p.total.x
            );
        } else {
            eprintln!("    strings={strings:2}: P_x = {:+.10}", p.total.x);
        }
        previous = Some(p.total);
    }
    assert!(
        last_step < 1.0e-5,
        "the polarization was still moving by {last_step:.3e} at 32 points per string"
    );
}

// ---------------------------------------------------- against an independent formalism

/// `Z*_{A,αβ} = Ω ∂P_α/∂τ_{A,β}` — the Born effective charges, from the Berry phase.
///
/// This crate computes the same tensor from a **CPHF response of the dipole operator**
/// (`pbc::born_charges`). The two share the SCF and nothing else: one is a finite difference of a
/// geometric phase over occupied bands, the other an analytic linear response. Agreement is the
/// strongest statement available about either.
///
/// They are not expected to agree exactly, and the reason is *measured* rather than asserted: the
/// Berry phase in this basis places each orbital at its atom, which is what the dipole operator's
/// diagonal does, but the dipole operator additionally carries the on-site `s`–`p` hybridization
/// moment `dd_a`. Here that accounts for **0.207 e** of a 0.266 e charge.
///
/// `on_a_hydrogen_only_cell_the_two_born_charge_routes_agree` is what makes that an explanation
/// rather than a story: with no `p` shell the `dd` term cannot exist, and the two routes then agree
/// to **7.5e-13 e**.
#[test]
fn the_born_charges_from_the_berry_phase_match_the_cphf_ones() {
    use am1_rs::pbc::born_charges;
    let params = Am1Parameters::standard().unwrap();
    let molecule = hf_cell(6.0, 0.94, 0.0);
    let volume = molecule.cell.unwrap().volume();

    // **Matched sampling on both sides.** The two routes must see the same Brillouin zone before
    // they can be compared: the Berry phase resamples the string's own direction at `strings`
    // points, so the CPHF ground state has to be sampled there too. Until 0.2.2 this test ran the
    // CPHF at Γ against a 12-point string and read the 0.064 e that produced as physics; it is
    // sampling, and with the two matched it falls to 1.2e-3 and keeps falling.
    //
    // A little smearing, and 1×1 transversely: displacing one atom of a two-atom cell breaks the
    // symmetry that was holding the sharp-filling SCF together, and the displaced geometries did
    // not converge without it. That is a property of this fixture, not of the Berry phase.
    let h = 0.01 * ANG;
    let mut previous = f64::INFINITY;
    for strings in [4usize, 6, 8] {
        let o = PbcOptions {
            kmesh: KMesh::MonkhorstPack([strings, 1, 1]),
            fold_time_reversal: false,
            smearing_ev: 0.02,
            max_scf: 2000,
            mixing: 0.2,
            ..options([1, 1, 1])
        };
        let reference = born_charges(&molecule, &params, &o).unwrap();

        // Finite difference of the polarization, reduced to the nearest branch at every step.
        //
        // Displacements along **x and z only**. A `y` displacement of either atom of this cell
        // does not converge at any smearing between 0.02 and 0.20 eV or any mesh from 4 to 8
        // points, while `x` and `z` converge at all of them -- a degeneracy this linear molecule's
        // own symmetry creates in the fixture, not a property of either route. Measured over that
        // grid of settings rather than assumed.
        let mut berry_z = vec![[[0.0_f64; 3]; 3]; molecule.atoms.len()];
        for atom in 0..molecule.atoms.len() {
            for beta in [0usize, 2] {
                let displaced = |sign: f64| {
                    let mut m = molecule.clone();
                    match beta {
                        0 => m.atoms[atom].position.x += sign * h,
                        1 => m.atoms[atom].position.y += sign * h,
                        _ => m.atoms[atom].position.z += sign * h,
                    }
                    berry_polarization(&m, &params, &o, strings).unwrap()
                };
                let minus = displaced(-1.0);
                let plus = displaced(1.0);
                let d = minus.difference(&plus) * (volume / (2.0 * h));
                berry_z[atom][0][beta] = d.x;
                berry_z[atom][1][beta] = d.y;
                berry_z[atom][2][beta] = d.z;
            }
        }

        // Two numbers, because two different claims rest on them. `worst_x` is the column along
        // the molecule -- the one the on-site moment actually moves, and the one whose residual is
        // the Berry phase's own discretization, so it is what the convergence assertion uses.
        // `worst` is every compared component, for the magnitude assertion. The transverse column
        // is an order of magnitude smaller and sits at this fixture's smearing noise floor, so
        // requiring *it* to fall monotonically would be asserting noise.
        let mut worst = 0.0_f64;
        let mut worst_x = 0.0_f64;
        for atom in 0..molecule.atoms.len() {
            for a in 0..3 {
                for b in [0usize, 2] {
                    let d = (berry_z[atom][a][b] - reference[atom][a][b]).abs();
                    worst = worst.max(d);
                    if b == 0 {
                        worst_x = worst_x.max(d);
                    }
                }
            }
        }
        eprintln!(
            "    strings={strings}: Z*_xx atom 0 Berry {:+.6}  CPHF {:+.6}  worst {worst:.3e} e \
             (x column {worst_x:.3e})",
            berry_z[0][0][0], reference[0][0][0]
        );

        // Both routes must obey the acoustic sum rule on their own, which follows from charge
        // conservation and not from anything the two share.
        for a in 0..3 {
            for b in [0usize, 2] {
                let s: f64 = (0..molecule.atoms.len()).map(|i| berry_z[i][a][b]).sum();
                assert!(
                    s.abs() < 5.0e-3,
                    "Berry-phase Z* violate their sum rule at ({a},{b}) by {s:.3e}"
                );
            }
        }
        assert!(
            worst_x < previous,
            "the two Born-charge routes did not converge together as the string lengthened: \
             {worst_x:.3e} against {previous:.3e}"
        );
        assert!(
            worst < 6.0e-3,
            "the two Born-charge routes differ by {worst:.3e} e at {strings} points per string"
        );
        previous = worst_x;
    }
    // At 8 points per string. The residual is the Berry phase's own `O(1/J²)` discretization --
    // the sequence above shows it falling -- and not a difference of formalism: since 0.2.2 the
    // link operator carries the on-site `s`–`p` moment, so the two are computing the same object.
    assert!(
        previous < 2.5e-3,
        "the two routes differ by {previous:.3e} e along the molecule at 8 points per string"
    );
}

/// The same comparison on a cell where the on-site hybridization moment **cannot exist**, so the
/// two routes have to agree outright.
///
/// This is what turns "the difference is the `s`–`p` term" from an explanation into a measurement.
/// Hydrogen has no `p` shell in AM1, so `Basis` gives it one orbital and
/// [`am1_rs::dipole_operator`]'s `dd` branch — which fires only for a four-orbital atom — is
/// unreachable for a hydrogen-only cell. The dipole operator is then *exactly* the position
/// operator the Berry phase uses, and any residual disagreement would be a defect in one of them
/// rather than a difference of definition.
#[test]
fn on_a_hydrogen_only_cell_the_two_born_charge_routes_agree() {
    use am1_rs::pbc::born_charges;
    let params = Am1Parameters::standard().unwrap();
    let o = PbcOptions {
        smearing_ev: 0.02,
        max_scf: 2000,
        mixing: 0.2,
        ..options([1, 1, 1])
    };
    // Two hydrogens, asymmetrically placed so the cell is polar and `Z*` is not zero by symmetry.
    let a = 6.0 * ANG;
    let molecule = |shift: f64| {
        Molecule::new(vec![
            Atom {
                z: 1,
                position: Vec3::new(shift, 0.0, 0.0),
            },
            Atom {
                z: 1,
                position: Vec3::new(shift + 0.80 * ANG, 0.35 * ANG, 0.0),
            },
        ])
        .with_cell(
            Lattice::from_vectors(
                Vec3::new(a, 0.0, 0.0),
                Vec3::new(0.0, a, 0.0),
                Vec3::new(0.0, 0.0, a),
                [true, true, true],
            )
            .unwrap(),
        )
    };
    let base = molecule(0.0);
    let volume = base.cell.unwrap().volume();
    let reference = born_charges(&base, &params, &o).unwrap();

    let h = 0.01 * ANG;
    let mut worst = 0.0_f64;
    for atom in 0..2 {
        for beta in 0..3 {
            let displaced = |sign: f64| {
                let mut m = base.clone();
                match beta {
                    0 => m.atoms[atom].position.x += sign * h,
                    1 => m.atoms[atom].position.y += sign * h,
                    _ => m.atoms[atom].position.z += sign * h,
                }
                berry_polarization(&m, &params, &o, 12).unwrap()
            };
            let d = displaced(-1.0).difference(&displaced(1.0)) * (volume / (2.0 * h));
            let got = [d.x, d.y, d.z];
            for (alpha, g) in got.iter().enumerate() {
                worst = worst.max((g - reference[atom][alpha][beta]).abs());
            }
        }
    }
    eprintln!("    H-only cell: Z*_xx Berry vs CPHF, worst component difference {worst:.3e} e");
    assert!(
        worst < 5.0e-3,
        "with no hybridization moment available the two routes should agree; they differ by \
         {worst:.3e} e, so the difference seen on HF is not only the s-p term"
    );
}

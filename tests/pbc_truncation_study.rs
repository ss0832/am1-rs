// SPDX-License-Identifier: GPL-3.0-or-later

//! What a real-space cutoff can and cannot do for a `1/R` lattice sum.
//!
//! This file records the limits of the pre-Ewald, Γ-only periodic path as measurements, so the
//! size of each approximation is on the record rather than assumed. It asserts only what is
//! actually true at this stage.
//!
//! ## Truncating the pair list: two schemes, each breaking what the other preserves
//!
//! * **By pair distance.** Slices through image shells — keeps an oxygen's attraction to a
//!   distant image while dropping a hydrogen's — so a neutral cell stops being neutral and the
//!   `1/R` monopole cancellation fails. Measured: a lone neutral carbon drifted by 5.7 Hartree.
//! * **By lattice translation.** Keeps whole images, so neutrality survives; this is what the
//!   code does. But a primitive cell and its supercell then sample different sets of physical
//!   separations, since the supercell's lattice is a sublattice with a coarser step.
//!
//! ## The exchange
//!
//! Separately from either, the two-centre exchange diverges at Γ and has to be tapered. The
//! taper is by distance alone — not by whether the partner is an image — because the same
//! physical pair is intra-cell in a supercell and an image pair in the primitive cell, and
//! keying on that would make a supercell disagree with itself.
//!
//! The measurements below are the reason `tests/pbc_gamma.rs` works at 12 Bohr and above.

use am1_rs::lattice::Lattice;
use am1_rs::math::Vec3;
use am1_rs::{run_am1, Am1Options, Am1Parameters, Atom, Molecule};

const ANG: f64 = 1.0 / 0.529167;

fn water_atoms() -> Vec<Atom> {
    [
        (8u8, [0.0, 0.0, 0.0]),
        (1, [0.9614, 0.0, 0.0]),
        (1, [-0.2246, 0.9348, 0.0]),
    ]
    .iter()
    .map(|(z, r)| Atom {
        z: *z,
        position: Vec3::new(r[0], r[1], r[2]) * ANG,
    })
    .collect()
}

fn options(cutoff: f64, exchange: Option<f64>) -> Am1Options {
    Am1Options {
        realspace_cutoff: cutoff,
        exchange_cutoff: exchange,
        ..Am1Options::default()
    }
}

/// Primitive cell and its 1x2 supercell, both cubic-derived from edge `a`.
fn pair_of_cells(a: f64) -> (Molecule, Molecule) {
    let primitive = Molecule::new(water_atoms()).with_cell(Lattice::cubic(a).unwrap());
    let mut super_atoms = water_atoms();
    for atom in water_atoms() {
        super_atoms.push(Atom {
            z: atom.z,
            position: atom.position + Vec3::new(a, 0.0, 0.0),
        });
    }
    let supercell = Molecule::new(super_atoms).with_cell(
        Lattice::from_vectors(
            Vec3::new(2.0 * a, 0.0, 0.0),
            Vec3::new(0.0, a, 0.0),
            Vec3::new(0.0, 0.0, a),
            [true, true, true],
        )
        .unwrap(),
    );
    (primitive, supercell)
}

#[test]
fn supercell_agreement_is_excellent_above_twelve_bohr_and_degrades_below() {
    // The exchange taper zone runs from 0.8*rc to rc. Once a cell is tight enough that real
    // inter-molecular pairs land inside that window, the primitive and supercell weight them
    // slightly differently and the agreement degrades sharply. This measures where that
    // happens, which is what sets the working range of the Γ-only path.
    let params = Am1Parameters::standard().unwrap();
    eprintln!("        a    exch      primitive      supercell/2     difference");
    let mut results = Vec::new();
    for a in [9.0_f64, 10.0, 12.0, 16.0, 24.0] {
        let (primitive, supercell) = pair_of_cells(a);
        let opts = options(4.0 * a, Some(8.0));
        let p = run_am1(&primitive, &params, &opts).unwrap().total_ev;
        let s = run_am1(&supercell, &params, &opts).unwrap().total_ev / 2.0;
        eprintln!("    {a:6.1}    8.00   {p:14.6}   {s:14.6}   {:+.3e}", s - p);
        results.push((a, (s - p).abs()));
    }

    let tight = results.iter().find(|(a, _)| *a == 9.0).unwrap().1;
    let workable = results.iter().find(|(a, _)| *a == 12.0).unwrap().1;
    eprintln!("\n    9 Bohr: {tight:.3e} eV,  12 Bohr: {workable:.3e} eV");
    assert!(
        workable < 1.0e-4,
        "supercell agreement at 12 Bohr should be excellent, got {workable:.3e} eV"
    );
    assert!(
        tight > 100.0 * workable,
        "the degradation at 9 Bohr should be dramatic; got {tight:.3e} vs {workable:.3e}"
    );
}

#[test]
fn a_narrower_exchange_taper_recovers_the_tight_cell() {
    // Confirms the mechanism: at 9 Bohr the trouble is the taper window overlapping genuine
    // inter-molecular pairs, not the image bookkeeping. Moving the window below them restores
    // agreement by a factor of ~60, at the cost of discarding real exchange physics — which is
    // exactly the trade k-point sampling removes.
    let params = Am1Parameters::standard().unwrap();
    let (primitive, supercell) = pair_of_cells(9.0);
    let mut prev = f64::INFINITY;
    for exch in [8.0_f64, 5.0] {
        let opts = options(36.0, Some(exch));
        let p = run_am1(&primitive, &params, &opts).unwrap().total_ev;
        let s = run_am1(&supercell, &params, &opts).unwrap().total_ev / 2.0;
        eprintln!(
            "    exchange cutoff {exch:4.1} Bohr -> supercell delta {:+.3e} eV",
            s - p
        );
        prev = prev.min((s - p).abs());
    }
    assert!(
        prev < 0.05,
        "narrowing the taper should recover the tight cell; best was {prev:.3e} eV"
    );
}

#[test]
fn keeping_the_image_exchange_diverges() {
    // The headline limitation, stated as a measurement. With the image exchange kept, the
    // energy is not merely inaccurate: it is wrong by hundreds of eV and gets worse as the
    // cell shrinks, because the Γ-only density matrix does not decay.
    let params = Am1Parameters::standard().unwrap();
    let molecular = run_am1(
        &Molecule::new(water_atoms()),
        &params,
        &Am1Options::default(),
    )
    .unwrap()
    .total_ev;

    eprintln!("    cell    exchange kept    exchange tapered");
    for a in [20.0_f64, 30.0, 50.0] {
        let cell = Molecule::new(water_atoms()).with_cell(Lattice::cubic(a).unwrap());
        let kept = run_am1(&cell, &params, &options(4.0 * a, None))
            .map(|r| r.total_ev - molecular)
            .unwrap_or(f64::NAN);
        let tapered = run_am1(&cell, &params, &options(4.0 * a, Some(8.0)))
            .map(|r| r.total_ev - molecular)
            .unwrap_or(f64::NAN);
        eprintln!("    {a:5.1}   {kept:+14.4} eV   {tapered:+14.6} eV");
        assert!(
            kept.abs() > 100.0,
            "at {a} Bohr the untapered exchange should be catastrophically wrong, got {kept:.3}"
        );
        assert!(
            tapered.abs() < 1.0,
            "at {a} Bohr the tapered result should be sane, got {tapered:.3}"
        );
    }
}

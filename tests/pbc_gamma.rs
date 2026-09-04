// SPDX-License-Identifier: GPL-3.0-or-later

//! Γ-point periodic energies.
//!
//! The molecular assembly run over an image-aware pair list. At `k = 0` the Bloch phase
//! `e^{ik·T}` is 1, so the Γ-point Hamiltonian *is* that sum, and these tests check the image
//! bookkeeping — does each physical pair count exactly once, does an atom see its own images.
//!
//! The long-range monopole sum is done by Ewald summation on top of that (see
//! [`am1_rs::pbc::ewald`]); [`Am1Options::ewald`] turns it off to recover the pure real-space
//! behaviour these tests were originally written against.
//!
//! ## Two things this stage requires, both established by measurement
//!
//! **The pair list is truncated by lattice translation, not by pair distance.** The NDDO
//! electrostatics is three `1/R` pieces that cancel to `Σ_ab Q_a Q_b γ` with `Σ_a Q_a = 0` for
//! a neutral cell. That cancellation needs every atom pair of a given image, or none of them:
//! a distance cutoff keeps an oxygen's attraction to a distant image while dropping a
//! hydrogen's, and the energy runs away. Measured before the fix: a lone neutral carbon drifted
//! by 5.7 Hartree.
//!
//! **The two-centre exchange must be tapered off.** NDDO carries a genuine exchange term whose
//! integral decays as `1/R`; it is finite only because the density matrix element it contracts
//! against decays with separation. At Γ-only sampling that element does not decay at all, so
//! the image sum diverges — the standard Hartree–Fock exchange divergence at Γ. Measured on a
//! lone neutral carbon in a 15 Bohr cell: −154 eV with the image exchange kept, +0.0003 eV with
//! it tapered. See `tests/pbc_exchange_diagnosis.rs`.
//!
//! Both are approximations that k-point sampling and Ewald remove. What this stage does
//! establish, to 1e-5 eV, is that the image bookkeeping itself is right.

use am1_rs::lattice::Lattice;
use am1_rs::math::Vec3;
use am1_rs::{run_am1, Am1Options, Am1Parameters, Atom, Molecule};

const ANG: f64 = 1.0 / 0.529167;

/// Smallest cell these tests use.
///
/// Below about 12 Bohr the exchange taper zone starts to cut through genuinely interacting
/// molecular pairs, and a primitive cell and its supercell then disagree at the 0.9 eV level
/// even though the bookkeeping is identical — the taper interacts with the translation
/// truncation. That is a real limitation of the pre-Ewald, Γ-only path, recorded in
/// `tests/pbc_truncation_study.rs` rather than hidden by choosing a tolerance to fit it.
const MIN_CELL_BOHR: f64 = 12.0;

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

fn options(cutoff: f64) -> Am1Options {
    Am1Options {
        realspace_cutoff: cutoff,
        exchange_cutoff: Some(8.0),
        ..Am1Options::default()
    }
}

/// [`options`] with the Klopman-Ohno `R^-3` tail switched off.
///
/// For the tests below that check a coefficient against a **closed form**. Those closed forms --
/// the image-dipole term `-2*pi*|p|^2/(3V)` above all -- are properties of the `1/R` kernel, and
/// the tail is a different channel: it is the lattice sum of `gamma_eta(R) - 1/R`, which is also
/// `O(L^-3)` and therefore lands in the same coefficient. With it on, the measured coefficient is
/// `-7.6` against a predicted `-10.6`, and the 28 % gap is not an error but the second channel.
///
/// Leaving it on and loosening the tolerance would have made the test agree with anything. Turning
/// it off is what keeps it a measurement of the monopole correction, which is what it is for.
fn options_untailed(cutoff: f64) -> Am1Options {
    Am1Options {
        klopman_ohno_tail: false,
        ..options(cutoff)
    }
}

#[test]
fn a_molecule_in_a_large_box_approaches_the_molecular_energy_as_the_cube_of_the_cell() {
    // A **polar** molecule in a periodic box does not reach its gas-phase energy however large
    // the box is, and it should not: under the tin-foil boundary condition the cell's dipole
    // interacts with the infinite lattice of its own images. That interaction is
    // `2π|p|²/3V` — the surface term proved exactly in `tests/pbc_ewald.rs` — so it falls as
    // `L⁻³` rather than vanishing at any finite size.
    //
    // Asserting `L⁻³` is a sharper statement than asserting a small number: a residual from a
    // real mistake (a dropped image, a mis-signed reciprocal term) would not have this power.
    // Before the Ewald correction this test asserted exact agreement, which held only because
    // the truncated real-space sum silently imposed the *spherical* convention instead.
    let params = Am1Parameters::standard().unwrap();
    let molecular = run_am1(
        &Molecule::new(water_atoms()),
        &params,
        &Am1Options::default(),
    )
    .unwrap();

    let mut deltas = Vec::new();
    for l in [45.0_f64, 60.0, 90.0] {
        let boxed = Molecule::new(water_atoms()).with_cell(Lattice::cubic(l).unwrap());
        let periodic = run_am1(&boxed, &params, &options_untailed(l / 2.0)).unwrap();
        assert!(periodic.converged, "L = {l}: SCF did not converge");
        let d = periodic.total_ev - molecular.total_ev;
        eprintln!(
            "    L = {l:5.1} Bohr: {:.9} eV, delta from molecular {d:+.4e} eV, delta * L^3 = {:.4}",
            periodic.total_ev,
            d * l.powi(3)
        );
        deltas.push((l, d));
    }
    eprintln!("    molecular reference {:.9} eV", molecular.total_ev);

    // Every residual must be an *attraction*: the image lattice of a dipole under tin-foil
    // lowers the energy. A positive residual would mean the sign of the reciprocal sum is wrong.
    for (l, d) in &deltas {
        assert!(
            *d < 0.0,
            "L = {l}: the image-dipole interaction must lower the energy, got {d:+.3e} eV"
        );
    }

    // `d · L³` constant is the `L⁻³` law.
    let scaled: Vec<f64> = deltas.iter().map(|(l, d)| d * l.powi(3)).collect();
    let hi = scaled.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let lo = scaled.iter().copied().fold(f64::INFINITY, f64::min);
    assert!(
        (hi - lo).abs() < 0.25 * lo.abs(),
        "the residual does not fall as L^-3, so it is not the image-dipole term: {scaled:?}"
    );

    // And the coefficient is not free either: it must be `−2π|p|²/3`.
    //
    // The dipole that matters is the **point-charge** one built from the net atomic charges,
    // not the reported AM1 dipole — the monopole correction never sees the sp hybridization
    // contribution that makes up the rest of the latter. Getting this right is what pins down
    // that the correction is applied through the net charges and nothing else.
    let mut p = am1_rs::math::Vec3::zero();
    for (atom, q) in molecular
        .charges
        .iter()
        .enumerate()
        .map(|(i, q)| (Molecule::new(water_atoms()).atoms[i].position, *q))
    {
        p += atom * q;
    }
    let predicted =
        -2.0 * std::f64::consts::PI * (p.norm() * p.norm()) / 3.0 * am1_rs::constants::AM1_EV;
    let measured = scaled.iter().sum::<f64>() / scaled.len() as f64;
    eprintln!(
        "    point-charge dipole {:.4} a.u.  =>  predicted -2*pi*|p|^2/3 = {predicted:.3} eV*Bohr^3, \
         measured {measured:.3}",
        p.norm()
    );
    assert!(
        (measured - predicted).abs() < 0.05 * predicted.abs(),
        "the image-dipole coefficient is {measured:.3} eV*Bohr^3 but -2*pi*|p|^2/3 = {predicted:.3}"
    );
}

#[test]
fn a_supercell_energy_is_twice_the_primitive_cell() {
    // The bookkeeping test: doubling the cell along one axis and duplicating its contents must
    // double the energy per cell. Double counting, a missing mirror pair, or a mishandled
    // self-image all break this, and none of them break the isolation test above.
    let params = Am1Parameters::standard().unwrap();
    let a = MIN_CELL_BOHR;

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

    let cutoff = 48.0;
    let p = run_am1(&primitive, &params, &options(cutoff)).unwrap();
    let s = run_am1(&supercell, &params, &options(cutoff)).unwrap();
    assert!(p.converged && s.converged);

    let per_cell = s.total_ev / 2.0;
    let d = per_cell - p.total_ev;
    eprintln!(
        "    primitive {:.9} eV, supercell/2 {per_cell:.9} eV, delta {d:+.3e}",
        p.total_ev
    );
    // Worth printing, and worth understanding: the *components* disagree by hundreds of eV
    // while the total agrees to 1e-5. A primitive cell and its supercell include different
    // lattice translations, so each one's core-core sum and electron-core attraction are
    // individually truncation-dependent — but both are driven by the same pair list, so the
    // (Z_a − P_a)(Z_b − P_b) cancellation holds pair by pair and the residual cancels in the
    // total. The upshot: for a periodic system the reported core-core and electronic
    // components are not separately meaningful; the total is.
    eprintln!(
        "    core-core: primitive {:.6}, supercell/2 {:.6}, delta {:+.3e}  \
         (components are truncation-dependent; only the total is not)",
        p.core_ev,
        s.core_ev / 2.0,
        s.core_ev / 2.0 - p.core_ev
    );
    assert!(
        d.abs() < 1.0e-4,
        "a supercell disagreed with its own primitive cell by {d:.3e} eV"
    );
}

#[test]
fn an_atom_interacts_with_its_own_images() {
    // A water in a moderate cell must not give the molecular energy: if it did, the image sum
    // would be doing nothing at all.
    let params = Am1Parameters::standard().unwrap();
    let molecular = run_am1(
        &Molecule::new(water_atoms()),
        &params,
        &Am1Options::default(),
    )
    .unwrap();
    let tight = Molecule::new(water_atoms()).with_cell(Lattice::cubic(MIN_CELL_BOHR).unwrap());
    let periodic = run_am1(&tight, &params, &options(48.0)).unwrap();

    let d = periodic.total_ev - molecular.total_ev;
    eprintln!(
        "    molecular {:.6} eV, in a {MIN_CELL_BOHR}-Bohr cell {:.6} eV, delta {d:.6} eV",
        molecular.total_ev, periodic.total_ev
    );
    assert!(
        d.abs() > 1.0e-3,
        "a {MIN_CELL_BOHR}-Bohr cell gave essentially the molecular energy (delta {d:.3e} eV); \
         the periodic images are not being seen"
    );
}

#[test]
fn every_dimensionality_runs_and_gives_a_distinct_answer() {
    // A chain, a slab and a crystal from the same cell. Deliberately *not* asserting that
    // energy falls monotonically with the number of periodic directions: these water molecules
    // are all identically oriented, so aligned dipoles stack head-to-tail along one axis and
    // side-by-side along another, and side-by-side aligned dipoles repel. "More neighbours
    // means more binding" is not a theorem here, and asserting it would be asserting a guess.
    let params = Am1Parameters::standard().unwrap();
    let a = MIN_CELL_BOHR;
    let vac = 90.0_f64;

    let make = |periodic: [bool; 3]| {
        let cell = Lattice::from_vectors(
            Vec3::new(a, 0.0, 0.0),
            Vec3::new(0.0, if periodic[1] { a } else { vac }, 0.0),
            Vec3::new(0.0, 0.0, if periodic[2] { a } else { vac }),
            periodic,
        )
        .unwrap();
        Molecule::new(water_atoms()).with_cell(cell)
    };

    let molecular = run_am1(
        &Molecule::new(water_atoms()),
        &params,
        &Am1Options::default(),
    )
    .unwrap()
    .total_ev;
    let cutoff = 60.0;
    let chain = run_am1(&make([true, false, false]), &params, &options(cutoff)).unwrap();
    let slab = run_am1(&make([true, true, false]), &params, &options(cutoff)).unwrap();
    let bulk = run_am1(&make([true, true, true]), &params, &options(cutoff)).unwrap();

    eprintln!("    molecular {molecular:.6} eV");
    for (name, r) in [("chain", &chain), ("slab", &slab), ("bulk", &bulk)] {
        eprintln!(
            "    {name:5}    {:.6} eV   ({:+.6} vs molecular)",
            r.total_ev,
            r.total_ev - molecular
        );
        assert!(r.converged, "{name} did not converge");
        assert!(
            (r.total_ev - molecular).abs() > 1.0e-4,
            "{name} returned the molecular energy; its periodic directions did nothing"
        );
    }
    // Each dimensionality sees a different neighbour set, so no two may coincide.
    assert!((chain.total_ev - slab.total_ev).abs() > 1.0e-4);
    assert!((slab.total_ev - bulk.total_ev).abs() > 1.0e-4);
}

// SPDX-License-Identifier: GPL-3.0-or-later

// The loops below index by atom and Cartesian axis, and the index *is* the quantity being
// checked -- `Z*_{a,alpha,beta}`, `alpha_ab` against its transpose. Rewriting them as
// iterators would hide which axis is which, so the lint is declined here rather than obeyed.
#![allow(clippy::needless_range_loop)]

//! Born effective charges, and the LO–TO splitting they carry.
//!
//! `Z*_{a,αβ} = ∂(V P_α)/∂u_{a,β}` — the dipole a cell acquires per unit displacement of one
//! atom. It is what makes a polar crystal's longitudinal and transverse optical branches
//! separate at `q → 0`; without it they stay degenerate, which is wrong by an amount that is not
//! small.
//!
//! Three properties are checked, in order of how hard they are to satisfy by accident:
//!
//! 1. **The acoustic sum rule `Σ_a Z*_a = 0`.** Translating the whole crystal produces no
//!    dipole. This follows from charge conservation and nothing else, so a violation is a defect
//!    in the response rather than a physical effect — and it is exact, not approximate.
//! 2. **A non-polar system has `Z* = Q δ`.** With no charge transfer to respond, the Born charge
//!    collapses to the static point charge.
//! 3. **`Z*` matches a finite difference of the dipole.** The direct definition, computed
//!    without any of the response machinery.

use am1_rs::lattice::Lattice;
use am1_rs::math::Vec3;
use am1_rs::pbc::{born_charges, KMesh, PbcOptions};
use am1_rs::{Am1Parameters, Atom, Molecule};

const ANG: f64 = 1.0 / 0.529167;

fn water_chain(spacing_ang: f64) -> Molecule {
    let l = spacing_ang * ANG;
    let atoms: Vec<Atom> = [
        (8u8, [0.0, 0.0, 0.0]),
        (1, [0.9614, 0.0, 0.0]),
        (1, [-0.2246, 0.9348, 0.0]),
    ]
    .iter()
    .map(|(z, r)| Atom {
        z: *z,
        position: Vec3::new(r[0], r[1], r[2]) * ANG,
    })
    .collect();
    Molecule::new(atoms).with_cell(
        Lattice::from_vectors(
            Vec3::new(l, 0.0, 0.0),
            Vec3::new(0.0, 40.0, 0.0),
            Vec3::new(0.0, 0.0, 40.0),
            [true, false, false],
        )
        .unwrap(),
    )
}

fn h2_chain(bond_ang: f64, spacing_ang: f64) -> Molecule {
    let l = spacing_ang * ANG;
    Molecule::new(vec![
        Atom {
            z: 1,
            position: Vec3::zero(),
        },
        Atom {
            z: 1,
            position: Vec3::new(bond_ang * ANG, 0.0, 0.0),
        },
    ])
    .with_cell(
        Lattice::from_vectors(
            Vec3::new(l, 0.0, 0.0),
            Vec3::new(0.0, 40.0, 0.0),
            Vec3::new(0.0, 0.0, 40.0),
            [true, false, false],
        )
        .unwrap(),
    )
}

fn options(mesh: [usize; 3]) -> PbcOptions {
    PbcOptions {
        kmesh: KMesh::MonkhorstPack(mesh),
        realspace_cutoff: 30.0,
        exchange_cutoff: Some(10.0),
        smearing_ev: 0.0,
        e_tol: 1.0e-11,
        p_tol: 1.0e-10,
        max_scf: 600,
        ..PbcOptions::default()
    }
}

#[test]
fn the_born_charges_obey_their_acoustic_sum_rule() {
    // `Σ_a Z*_a = 0`, exactly. This is the sharpest available check on the charge response,
    // because it is an identity rather than a comparison: it holds for any system, at any
    // geometry, whatever the physics happens to be, so a residual is unambiguously a defect.
    let params = Am1Parameters::standard().unwrap();
    for (label, molecule, mesh) in [
        ("water chain, Gamma", water_chain(3.4), [1usize, 1, 1]),
        ("water chain, 3 k-points", water_chain(3.4), [3, 1, 1]),
        ("H2 chain, 3 k-points", h2_chain(0.7, 3.6), [3, 1, 1]),
    ] {
        let z = born_charges(&molecule, &params, &options(mesh)).unwrap();
        let mut worst = 0.0_f64;
        for alpha in 0..3 {
            for beta in 0..3 {
                let sum: f64 = z.iter().map(|t| t[alpha][beta]).sum();
                worst = worst.max(sum.abs());
            }
        }
        eprintln!("    {label}: max |Σ_a Z*_a| = {worst:.3e} e");
        assert!(
            worst < 1.0e-6,
            "{label}: the Born-charge acoustic sum rule is violated by {worst:.3e}"
        );
    }
}

#[test]
fn a_symmetric_chain_has_born_charges_that_are_just_its_point_charges() {
    // A homonuclear chain has no charge transfer to respond with, so `Z*` collapses onto the
    // static charge — which is itself zero by symmetry. This separates "the response is right"
    // from "the response is absent": a code that returned only the `Q_a δ_αβ` term would pass
    // the sum rule above just as happily, and would fail the water case below.
    let params = Am1Parameters::standard().unwrap();
    let z = born_charges(&h2_chain(0.7, 3.6), &params, &options([3, 1, 1])).unwrap();
    let mut worst = 0.0_f64;
    for t in &z {
        for row in t {
            for v in row {
                worst = worst.max(v.abs());
            }
        }
    }
    eprintln!("    symmetric H2 chain: max |Z*| = {worst:.3e} e");
    assert!(
        worst < 1.0e-3,
        "a symmetric chain should have vanishing Z*, got {worst:.3e}"
    );
}

#[test]
fn the_born_charges_match_a_finite_difference_of_the_cell_dipole() {
    // The direct definition, with none of the response machinery in it: displace one atom, see
    // how the cell dipole moves.
    //
    // The dipole here is the model's own — net atomic charges times positions, plus the on-site
    // sp hybridization moments — which is exactly what `born_charges` differentiates. Comparing
    // against a point-charge-only dipole would be comparing against a different quantity.
    use am1_rs::pbc::run_pbc_scf;
    let params = Am1Parameters::standard().unwrap();
    let molecule = water_chain(3.4);
    let o = options([3, 1, 1]);
    let analytic = born_charges(&molecule, &params, &o).unwrap();

    let dipole = |m: &Molecule| -> Vec3 {
        let scf = run_pbc_scf(m, &params, &o).unwrap();
        let basis = am1_rs::basis::Basis::build(m, &params).unwrap();
        let p0 = scf
            .density
            .get(am1_rs::lattice::ImageOffset::origin())
            .unwrap();
        let mut p = Vec3::zero();
        for (b, atom) in m.atoms.iter().enumerate() {
            let elem = params.element(atom.z).unwrap();
            let off = basis.atom_offset[b];
            let norb = basis.atom_norb[b];
            let mut pop = 0.0;
            for k in 0..norb {
                pop += p0[(off + k, off + k)];
            }
            p += atom.position * (elem.core_charge - pop);
            if norb == 4 {
                p += Vec3::new(
                    -2.0 * elem.dd * p0[(off, off + 1)],
                    -2.0 * elem.dd * p0[(off, off + 2)],
                    -2.0 * elem.dd * p0[(off, off + 3)],
                );
            }
        }
        p
    };

    let step = 1.0e-4;
    let mut worst = 0.0_f64;
    for a in 0..molecule.atoms.len() {
        for beta in 0..3 {
            let shifted = |d: f64| {
                let mut m = molecule.clone();
                let q = &mut m.atoms[a].position;
                match beta {
                    0 => q.x += d,
                    1 => q.y += d,
                    _ => q.z += d,
                }
                dipole(&m)
            };
            let fd = (shifted(step) - shifted(-step)) / (2.0 * step);
            for (alpha, f) in [fd.x, fd.y, fd.z].iter().enumerate() {
                worst = worst.max((analytic[a][alpha][beta] - f).abs());
            }
        }
    }
    eprintln!("    water chain: max |Z*_analytic - Z*_finite difference| = {worst:.3e} e");
    assert!(
        worst < 1.0e-4,
        "Born charges disagree with a dipole finite difference by {worst:.3e}"
    );
}

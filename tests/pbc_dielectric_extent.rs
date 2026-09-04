// SPDX-License-Identifier: GPL-3.0-or-later

//! `ε_∞` for a slab and a chain, once the caller says how thick the material is.
//!
//! # What changed in 0.2.2, and what did not
//!
//! `ε_∞ = 1 + 4πα/Ω` needs `Ω` to be a volume. 0.2.0 fed it an *area* for a slab and a *length*
//! for a chain and returned numbers that were not dielectric constants; 0.2.1 refused. Neither is
//! the same as the quantity being undefined: what is missing is a **thickness**, and a thickness is
//! a claim about where the material stops. 0.2.2 takes it as a required argument.
//!
//! So the tests here divide in two, and the division is the point:
//!
//! - the **conversion** is arithmetic on a declared body, and its unit tests live next to it in
//!   `src/pbc/extent.rs` where they can use synthetic `α`;
//! - what needs an SCF is whether `α` is the quantity the conversion assumes — the response to the
//!   **external** field, with the depolarizing field already inside it. That is a physical claim
//!   about the periodic CPHF, it is what makes the out-of-plane law `1/(1 − 4πχ)` instead of
//!   `1 + 4πχ`, and getting it backwards produces a plausible number. It is measured here.
//!
//! Plus the invariants that hold whatever thickness is chosen, because those are what a slab
//! calculation can honestly report, and they must survive the SCF as well as the algebra.

#![allow(clippy::needless_range_loop)]

use am1_rs::lattice::Lattice;
use am1_rs::math::Vec3;
use am1_rs::pbc::{
    dielectric_function, dielectric_tensor, dielectric_tensor_with_extent,
    epsilon_from_polarizability, extent_axis_mixing, polarizability, ExtentConvention, KMesh,
    PbcOptions,
};
use am1_rs::{Am1Parameters, Atom, Molecule};

use std::f64::consts::PI;

const ANG: f64 = 1.0 / 0.529167;

fn options(mesh: [usize; 3]) -> PbcOptions {
    PbcOptions {
        kmesh: KMesh::MonkhorstPack(mesh),
        fold_time_reversal: false,
        realspace_cutoff: 30.0,
        exchange_cutoff: Some(12.0),
        smearing_ev: 0.0,
        e_tol: 1.0e-11,
        p_tol: 1.0e-10,
        max_scf: 800,
        ..PbcOptions::default()
    }
}

/// An HF slab: two periodic directions with `spacing`, vacuum `height` along `z`.
fn hf_slab(spacing_ang: f64, height_bohr: f64) -> Molecule {
    let step = spacing_ang * ANG;
    Molecule::new(vec![
        Atom {
            z: 9,
            position: Vec3::zero(),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.94 * ANG, 0.2 * ANG, 0.0),
        },
    ])
    .with_cell(
        Lattice::from_vectors(
            Vec3::new(step, 0.0, 0.0),
            Vec3::new(0.0, step, 0.0),
            Vec3::new(0.0, 0.0, height_bohr),
            [true, true, false],
        )
        .unwrap(),
    )
}

fn hf_chain(spacing_ang: f64) -> Molecule {
    let step = spacing_ang * ANG;
    Molecule::new(vec![
        Atom {
            z: 9,
            position: Vec3::zero(),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.94 * ANG, 0.0, 0.0),
        },
    ])
    .with_cell(
        Lattice::from_vectors(
            Vec3::new(step, 0.0, 0.0),
            Vec3::new(0.0, 30.0, 0.0),
            Vec3::new(0.0, 0.0, 30.0),
            [true, false, false],
        )
        .unwrap(),
    )
}

/// Methane, which is non-polar and nearly isotropic — the local-field test wants a molecule whose
/// neighbours pull on it electrostatically and not through a hydrogen bond.
fn methane_atoms() -> Vec<Atom> {
    let d = 1.087 * ANG / 3.0_f64.sqrt();
    [
        (6u8, [0.0, 0.0, 0.0]),
        (1, [d, d, d]),
        (1, [d, -d, -d]),
        (1, [-d, d, -d]),
        (1, [-d, -d, d]),
    ]
    .iter()
    .map(|(z, r)| Atom {
        z: *z,
        position: Vec3::new(r[0], r[1], r[2]),
    })
    .collect()
}

fn methane_slab(spacing_ang: f64) -> Molecule {
    let step = spacing_ang * ANG;
    Molecule::new(methane_atoms()).with_cell(
        Lattice::from_vectors(
            Vec3::new(step, 0.0, 0.0),
            Vec3::new(0.0, step, 0.0),
            Vec3::new(0.0, 0.0, 40.0),
            [true, true, false],
        )
        .unwrap(),
    )
}

// ------------------------------------------------------------------ the 0.2.0 failure mode

/// **The answer must not depend on the vacuum**, which is exactly what 0.2.0 got wrong.
///
/// 0.2.0 divided `α` by `Lattice::measure`. For a slab that is an area, so the "dielectric
/// constant" it reported was a length — and worse, adding padding along `z` changed nothing in it
/// while changing everything about the cell, so there was no way to notice from the number alone.
///
/// With a *declared* thickness the vacuum drops out completely: `α` is a property of the layer, the
/// thickness is a property of the layer, and the cell height enters neither. Anything left is the
/// SCF's own convergence, so the tolerance is tight on purpose.
#[test]
fn the_conversion_does_not_depend_on_the_vacuum() {
    let params = Am1Parameters::standard().unwrap();
    let d = 6.0; // Bohr, the assigned thickness — the same layer in every cell.
    let mut seen: Vec<[[f64; 3]; 3]> = Vec::new();
    for height in [24.0, 30.0, 40.0] {
        let m = hf_slab(4.0, height);
        let (alpha, eps) = dielectric_tensor_with_extent(
            &m,
            &params,
            &options([4, 4, 1]),
            ExtentConvention::SlabThickness(d),
        )
        .unwrap();
        eprintln!(
            "    height {height:5.1} Bohr: alpha_zz {:9.5}  eps_xx {:9.6}  eps_zz {:9.6}",
            alpha[2][2], eps[0][0], eps[2][2]
        );
        seen.push(eps);
    }
    for e in &seen[1..] {
        for a in 0..3 {
            for b in 0..3 {
                let drift = (e[a][b] - seen[0][a][b]).abs();
                assert!(
                    drift < 2.0e-6,
                    "eps[{a}][{b}] moved by {drift:.2e} when only the vacuum changed"
                );
            }
        }
    }
}

// ------------------------------------------------------------------ what survives the choice

/// The two combinations of `ε` and `d` that are **thickness-free**, measured on a real `α`.
///
/// ```text
/// (ε_∥ − 1) d = 4π α_∥/A            (1 − 1/ε_⊥) d = 4π α_⊥/A
/// ```
///
/// These are what a slab calculation can report without adopting a convention; everything else in
/// `ε` is the convention. `α` is solved once and the sweep is pure arithmetic, so a failure here is
/// the conversion and not the SCF.
#[test]
fn the_sheet_invariants_survive_a_real_polarizability() {
    let params = Am1Parameters::standard().unwrap();
    let m = hf_slab(4.0, 30.0);
    let cell = m.cell.unwrap();
    let alpha = polarizability(&m, &params, &options([4, 4, 1])).unwrap();
    let area = cell.measure();
    let n = Vec3::new(0.0, 0.0, 1.0);

    // The slab normal has to be a principal axis of `α` for the split to be exact; say so.
    let mixing = extent_axis_mixing(&alpha, n);
    eprintln!("    axis mixing = {mixing:.3e}");
    assert!(
        mixing < 1.0e-10,
        "the normal is not a principal axis: {mixing:.3e}"
    );

    let want_in = 4.0 * PI * alpha[0][0] / area;
    let want_out = 4.0 * PI * alpha[2][2] / area;
    eprintln!("    4*pi*alpha_xx/A = {want_in:.6} Bohr,  4*pi*alpha_zz/A = {want_out:.6} Bohr");

    for d in [3.0, 6.0, 12.0, 24.0] {
        let e = epsilon_from_polarizability(&alpha, n, area, ExtentConvention::SlabThickness(d))
            .unwrap();
        let got_in = (e[0][0] - 1.0) * d;
        let got_out = (1.0 - 1.0 / e[2][2]) * d;
        eprintln!(
            "    d = {d:5.1}: eps = ({:.6}, {:.6})  invariants = ({got_in:.6}, {got_out:.6})",
            e[0][0], e[2][2]
        );
        assert!((got_in - want_in).abs() < 1.0e-11, "in-plane at d={d}");
        assert!(
            (got_out - want_out).abs() < 1.0e-11,
            "out-of-plane at d={d}"
        );
        // And `ε` itself really does move with `d`, so the invariance above is not trivial.
        assert!(e[0][0] > 1.0 && e[2][2] > 1.0);
    }
    // And the choice really does matter: `ε − 1` is inversely proportional to the thickness, so
    // an eightfold thicker layer screens exactly an eighth as much. Comparing `ε` itself would
    // understate it — 1.30 against 1.04 — which is the whole reason the *invariants* above are
    // what gets reported rather than `ε`.
    let thin =
        epsilon_from_polarizability(&alpha, n, area, ExtentConvention::SlabThickness(3.0)).unwrap();
    let thick = epsilon_from_polarizability(&alpha, n, area, ExtentConvention::SlabThickness(24.0))
        .unwrap();
    let ratio = (thin[0][0] - 1.0) / (thick[0][0] - 1.0);
    eprintln!("    (eps(3)-1)/(eps(24)-1) = {ratio:.10}, expected 8");
    assert!(
        (ratio - 8.0).abs() < 1.0e-9,
        "eps-1 should scale as 1/d: {ratio}"
    );
}

/// The in-plane invariant is **the same quantity** `dielectric_function` is built on, reached by a
/// different route.
///
/// `ε(q) = 1 + 2π (q̂·α·q̂)|q|/A` for a slab, so `(ε(q) − 1)/|q| → 2π α_∥/A`, while the thickness
/// conversion gives `(ε_∥ − 1)d = 4π α_∥/A`. The ratio is exactly **2**, and nothing in either
/// derivation was allowed to know about the other: one is a Coulomb kernel in reciprocal space,
/// the other a capacitor in real space.
///
/// This is the check that would have caught the `2π` versus `4π` slip, and the one that ties the
/// thickness-dependent number back to the thickness-free one.
#[test]
fn the_in_plane_invariant_agrees_with_the_dielectric_function() {
    let params = Am1Parameters::standard().unwrap();
    let m = hf_slab(4.0, 30.0);
    let cell = m.cell.unwrap();
    let o = options([4, 4, 1]);
    let alpha = polarizability(&m, &params, &o).unwrap();
    let area = cell.measure();

    let d = 5.0;
    let eps = epsilon_from_polarizability(
        &alpha,
        Vec3::new(0.0, 0.0, 1.0),
        area,
        ExtentConvention::SlabThickness(d),
    )
    .unwrap();
    // `x` is in plane, and the in-plane law is `I + 4πχ` on the whole block, so the Cartesian
    // component may be read directly without worrying about principal axes.
    let from_extent = (eps[0][0] - 1.0) * d;

    let recip = cell.reciprocal_vectors_2pi();
    let qhat = recip[0] / recip[0].norm();
    let mut slopes = Vec::new();
    for scale in [0.03, 0.015, 0.0075] {
        let q = qhat * (recip[0].norm() * scale);
        let e = dielectric_function(&m, &params, &o, q, None).unwrap();
        slopes.push((e - 1.0) / q.norm());
    }
    eprintln!("    (eps(q)-1)/|q| = {slopes:?}");
    // `ε(q)` is linear in `|q|` for a slab, so the slope is already the limit.
    for s in &slopes[1..] {
        assert!(
            (s - slopes[0]).abs() < 1.0e-6 * slopes[0].abs().max(1.0),
            "the slab's eps(q) should be linear in |q|: {slopes:?}"
        );
    }

    // `q` is along `x` only if the cell is orthogonal, which it is here — assert it rather than
    // trust it, since a sheared cell would make this compare two different components.
    assert!(qhat.y.abs() < 1.0e-12 && qhat.z.abs() < 1.0e-12, "{qhat:?}");

    let ratio = from_extent / slopes[0];
    eprintln!(
        "    (eps_par-1)d = {from_extent:.8},  slope = {:.8},  ratio = {ratio:.10}",
        slopes[0]
    );
    assert!(
        (ratio - 2.0).abs() < 1.0e-7,
        "the two routes to the sheet susceptibility disagree: ratio {ratio}"
    );
}

// ------------------------------------------------------------ is alpha the external response?

/// **The depolarizing field is already inside `α`**, which is the physical premise the whole
/// conversion rests on — and it is asserted nowhere else in the crate.
///
/// The periodic CPHF lets the induced charges interact through the same Coulomb operator the SCF
/// uses, and for a 2D-periodic cell the in-plane lattice sum of those induced charges is a *sheet*.
/// A sheet polarized along its normal produces `−4πP` inside itself; polarized in its own plane it
/// produces nothing macroscopic, and the near-field lattice sum instead **enhances**. So bringing
/// the molecules closer must move the two components in **opposite directions**:
///
/// | | isolated → dense |
/// |---|---|
/// | `α_zz` (normal) | falls — depolarization opposes |
/// | `α_xx` (in plane) | rises — the neighbours' fields add |
///
/// If `α` were the response to the *internal* macroscopic field there would be no such asymmetry,
/// and the out-of-plane conversion would need `1 + 4πχ` rather than `1/(1 − 4πχ)`. Both laws are
/// positive, both are monotonic, both look right: the sign asymmetry is what separates them.
///
/// Methane, because the effect wanted is electrostatic and a hydrogen-bonding molecule would answer
/// with chemistry instead.
#[test]
fn the_polarizability_carries_the_depolarizing_field() {
    let params = Am1Parameters::standard().unwrap();

    // The dilute end stands in for the isolated molecule: at 14 Å the neighbours' fields are
    // three orders of magnitude down on the on-site one, so the trend from there is the lattice's
    // doing and not the molecule's.
    let mut rows = Vec::new();
    for spacing in [14.0, 9.0, 6.5] {
        let m = methane_slab(spacing);
        let a = polarizability(&m, &params, &options([4, 4, 1])).unwrap();
        eprintln!(
            "    spacing {spacing:5.2} A: alpha_xx {:9.5}  alpha_zz {:9.5} Bohr^3",
            a[0][0], a[2][2]
        );
        rows.push((spacing, a[0][0], a[2][2]));
    }

    // Monotone in opposite directions as the lattice tightens.
    for w in rows.windows(2) {
        let (s0, xx0, zz0) = w[0];
        let (s1, xx1, zz1) = w[1];
        assert!(
            xx1 > xx0,
            "in-plane alpha should rise from {s0} A to {s1} A: {xx0} -> {xx1}"
        );
        assert!(
            zz1 < zz0,
            "out-of-plane alpha should fall from {s0} A to {s1} A: {zz0} -> {zz1}"
        );
    }

    // And the split is a real one, not two numbers that happen to straddle: at the tightest
    // spacing the anisotropy is well outside anything the SCF tolerance could produce.
    let (_, xx, zz) = *rows.last().unwrap();
    let (_, xx0, zz0) = rows[0];
    let split = (xx / xx0 - 1.0) - (zz / zz0 - 1.0);
    eprintln!("    fractional split between the two channels: {split:.4}");
    assert!(
        split > 1.0e-3,
        "the two channels barely separated ({split:.2e}); depolarization is not showing"
    );
}

// ------------------------------------------------------------------ the three-dimensional tie

/// The new conversion **contains** the old one: fed a crystal's `α` with the depolarization factor
/// its tin-foil boundary conditions imply (`N = 0`), it reproduces `dielectric_tensor` exactly.
///
/// Three-dimensional tin-foil summation removes the macroscopic depolarizing field, so there `α`
/// is already the response to the internal macroscopic field and `1 + 4πα/Ω` is right. The in-plane
/// branch of the slab law is the same `N = 0` arithmetic, so pointing it at a volume must give back
/// the same tensor — to round-off, not to a tolerance.
#[test]
fn the_zero_depolarization_branch_reproduces_the_crystal_tensor() {
    let params = Am1Parameters::standard().unwrap();
    let a = 4.5 * ANG;
    let crystal = Molecule::new(vec![
        Atom {
            z: 9,
            position: Vec3::zero(),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.94 * ANG, 0.0, 0.0),
        },
    ])
    .with_cell(Lattice::cubic(a).unwrap());

    let (alpha, eps) = dielectric_tensor(&crystal, &params, &options([2, 2, 2])).unwrap();
    // Read the volume as "area × height" and take the in-plane (`N = 0`) branch.
    let area = a * a;
    let via_extent = epsilon_from_polarizability(
        &alpha,
        Vec3::new(0.0, 0.0, 1.0),
        area,
        ExtentConvention::SlabThickness(a),
    )
    .unwrap();
    for i in 0..2 {
        for j in 0..2 {
            let drift = (via_extent[i][j] - eps[i][j]).abs();
            eprintln!(
                "    eps[{i}][{j}]: {:.12} vs {:.12}",
                via_extent[i][j], eps[i][j]
            );
            assert!(drift < 1.0e-13, "({i},{j}) differs by {drift:.2e}");
        }
    }
}

// ------------------------------------------------------------------ the chain

/// A wire gets the **circular cylinder**'s transverse factor `N = 1/2`, and its own
/// cross-section-free invariant `S(ε_⊥ − 1)/(ε_⊥ + 1) = 2π α_⊥/L`.
///
/// The axial direction has no depolarization — an infinite wire polarized along itself has no ends
/// — so it takes the same `1 + 4πχ` the slab's plane does. Two different factors in one tensor,
/// which is why the conversion diagonalizes rather than dividing.
#[test]
fn a_wire_gets_the_cylinder_factor_transverse_and_none_along_its_axis() {
    let params = Am1Parameters::standard().unwrap();
    let m = hf_chain(4.0);
    let cell = m.cell.unwrap();
    let length = cell.measure();
    let alpha = polarizability(&m, &params, &options([4, 1, 1])).unwrap();
    eprintln!(
        "    chain alpha = ({:.5}, {:.5}, {:.5}) Bohr^3, L = {length:.4} Bohr",
        alpha[0][0], alpha[1][1], alpha[2][2]
    );

    let axis = Vec3::new(1.0, 0.0, 0.0);
    let want_axial = 4.0 * PI * alpha[0][0] / length;
    let want_trans = 2.0 * PI * alpha[1][1] / length;
    for s in [20.0, 60.0, 150.0] {
        let e = epsilon_from_polarizability(
            &alpha,
            axis,
            length,
            ExtentConvention::WireCrossSection(s),
        )
        .unwrap();
        let axial = (e[0][0] - 1.0) * s;
        let trans = s * (e[1][1] - 1.0) / (e[1][1] + 1.0);
        eprintln!(
            "    S = {s:6.1}: eps = ({:.6}, {:.6})  invariants = ({axial:.6}, {trans:.6})",
            e[0][0], e[1][1]
        );
        assert!((axial - want_axial).abs() < 1.0e-10, "axial at S={s}");
        assert!((trans - want_trans).abs() < 1.0e-10, "transverse at S={s}");
        // The transverse law is *not* the axial one; if it were, this would be `4π α/L`.
        assert!((trans - want_axial * 0.5).abs() > 1.0e-6 || alpha[0][0] == alpha[1][1]);
    }

    // End to end, through the cell.
    let (_, eps) = dielectric_tensor_with_extent(
        &m,
        &params,
        &options([4, 1, 1]),
        ExtentConvention::WireCrossSection(60.0),
    )
    .unwrap();
    let direct = epsilon_from_polarizability(
        &alpha,
        axis,
        length,
        ExtentConvention::WireCrossSection(60.0),
    )
    .unwrap();
    for i in 0..3 {
        for j in 0..3 {
            assert!((eps[i][j] - direct[i][j]).abs() < 1.0e-12);
        }
    }
}

// ------------------------------------------------------------------ what it still refuses

/// Every refusal is about something the caller has to decide, and each names what to do instead.
/// None of them runs an SCF first — the cell alone settles them, so a mistake costs no time.
#[test]
fn the_conversion_refuses_what_the_caller_has_to_settle() {
    let params = Am1Parameters::standard().unwrap();
    let o = options([2, 2, 1]);

    // A crystal does not need an assigned extent.
    let crystal = Molecule::new(vec![Atom {
        z: 9,
        position: Vec3::zero(),
    }])
    .with_cell(Lattice::cubic(4.0 * ANG).unwrap());
    let err =
        dielectric_tensor_with_extent(&crystal, &params, &o, ExtentConvention::SlabThickness(3.0))
            .unwrap_err()
            .to_string();
    assert!(err.contains("pbc::dielectric_tensor"), "{err}");

    // A chain is not a slab: different units, different depolarization factor.
    let err = dielectric_tensor_with_extent(
        &hf_chain(4.0),
        &params,
        &o,
        ExtentConvention::SlabThickness(3.0),
    )
    .unwrap_err()
    .to_string();
    eprintln!("    {err}");
    assert!(
        err.contains("SlabThickness") && err.contains("chain"),
        "{err}"
    );

    // A body larger than its own cell would overlap its images.
    let err = dielectric_tensor_with_extent(
        &hf_slab(4.0, 30.0),
        &params,
        &o,
        ExtentConvention::SlabThickness(31.0),
    )
    .unwrap_err()
    .to_string();
    eprintln!("    {err}");
    assert!(err.contains("overlap"), "{err}");

    // Exactly the cell height is allowed: that is the no-vacuum limit, not an error.
    assert!(dielectric_tensor_with_extent(
        &hf_slab(4.0, 30.0),
        &params,
        &options([4, 4, 1]),
        ExtentConvention::SlabThickness(30.0)
    )
    .is_ok());

    // And `dielectric_tensor` still refuses a slab, now naming both alternatives.
    let err = dielectric_tensor(&hf_slab(4.0, 30.0), &params, &o)
        .unwrap_err()
        .to_string();
    assert!(err.contains("pbc::polarizability"), "{err}");
    assert!(err.contains("dielectric_tensor_with_extent"), "{err}");
}

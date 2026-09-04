// SPDX-License-Identifier: GPL-3.0-or-later

//! How the periodic SCF converges, and the three things that stopped it converging through 0.2.2.
//!
//! A two-dimensional lattice of **methane** is the system throughout. It is about as easy as a
//! periodic calculation gets — closed shell, nine electronvolts of gap, no hydrogen bonds, no
//! magnetism, no metal — and through 0.2.2 it could not be converged to `p_tol = 1e-10` at *any*
//! iteration count, mesh, cutoff, mixing fraction, or smearing width. What made it hard is the one
//! thing it does have: a **threefold degenerate HOMO**. The isolated molecule converged fine, so
//! nothing in the parameterization or the integrals was at fault.
//!
//! Three separate defects, found in that order by tracing:
//!
//! 1. **`pbc::complex::hermitian_eigen` lost `√ε` on a degenerate level.** The real embedding
//!    doubles every eigenvalue; picking one complex vector per pair used a single classical
//!    Gram–Schmidt pass and accepted any residual over `1e-8`. For a degenerate level the
//!    duplicate is genuinely in the span already, so what survived the subtraction was
//!    cancellation noise, renormalized to unit length and accepted. The density rebuilt from it
//!    carried `3e-8`, which is exactly the floor the SCF stalled at. Fixed by projecting twice and
//!    cutting at `0.1`; `src/pbc/complex.rs` checks the projector against its closed form.
//! 2. **The periodic SCF had no convergence acceleration at all** — plain linear mixing, while the
//!    molecular path had A-DIIS→CDIIS. It did not show because the test systems were stiff:
//!    hydrogen fluoride reached `1e-10` in about 140 passes, which looks like a slow system rather
//!    than a missing feature. Fixed with Pulay mixing on the real-space density.
//! 3. **The energy was contracted from the mixed density against the unmixed Fock**, so `de` — half
//!    the convergence test — measured the mixer as much as the iteration.
//!
//! The lesson those three share is in the ordering: a defect that only costs iterations hides a
//! defect that costs correctness, because both present as "it needs more iterations".

use am1_rs::lattice::Lattice;
use am1_rs::math::Vec3;
use am1_rs::pbc::{run_pbc_scf, KMesh, PbcOptions};
use am1_rs::{Am1Parameters, Atom, Molecule};

const ANG: f64 = 1.0 / 0.529167;

/// Methane: `T_d`, so the HOMO is a threefold degenerate `t₂`.
fn methane() -> Vec<Atom> {
    let d = 1.087 * ANG / 3.0_f64.sqrt();
    [
        (6u8, [1.0, 1.0, 1.0]),
        (1, [1.0, 1.0, 1.0]),
        (1, [1.0, -1.0, -1.0]),
        (1, [-1.0, 1.0, -1.0]),
        (1, [-1.0, -1.0, 1.0]),
    ]
    .iter()
    .enumerate()
    .map(|(i, (z, s))| Atom {
        z: *z,
        position: if i == 0 {
            Vec3::zero()
        } else {
            Vec3::new(s[0] * d, s[1] * d, s[2] * d)
        },
    })
    .collect()
}

fn methane_slab(spacing_ang: f64) -> Molecule {
    let step = spacing_ang * ANG;
    Molecule::new(methane()).with_cell(
        Lattice::from_vectors(
            Vec3::new(step, 0.0, 0.0),
            Vec3::new(0.0, step, 0.0),
            Vec3::new(0.0, 0.0, 40.0),
            [true, true, false],
        )
        .unwrap(),
    )
}

fn tight(mesh: [usize; 3]) -> PbcOptions {
    PbcOptions {
        kmesh: KMesh::MonkhorstPack(mesh),
        fold_time_reversal: false,
        realspace_cutoff: 30.0,
        exchange_cutoff: Some(12.0),
        smearing_ev: 0.0,
        e_tol: 1.0e-11,
        p_tol: 1.0e-10,
        max_scf: 300,
        ..PbcOptions::default()
    }
}

/// **A degenerate closed-shell insulator converges**, at every mesh and spacing.
///
/// The iteration cap is 300 and the assertion is 60, so this fails loudly rather than by taking a
/// long time — the failure mode being guarded against is a stall, and a stall is invisible if the
/// only thing checked is `converged`.
#[test]
fn a_degenerate_molecular_slab_converges_at_every_mesh() {
    let params = Am1Parameters::standard().unwrap();
    let mut worst = 0usize;
    for spacing in [6.5, 9.0, 14.0] {
        for mesh in [[1, 1, 1], [2, 2, 1], [3, 3, 1], [4, 4, 1]] {
            let r = run_pbc_scf(&methane_slab(spacing), &params, &tight(mesh)).unwrap();
            eprintln!(
                "    {spacing:5.1} A mesh {mesh:?}: converged={} in {:3} iterations, E = {:.9} eV",
                r.converged, r.iterations, r.total_ev
            );
            assert!(
                r.converged,
                "{spacing} A, mesh {mesh:?}: not converged in {} iterations",
                r.iterations
            );
            worst = worst.max(r.iterations);
        }
    }
    eprintln!("    worst case: {worst} iterations");
    assert!(worst <= 60, "convergence has regressed: {worst} iterations");
}

/// The dilute limit is the **isolated molecule**, and must agree with it.
///
/// At 14 Å the neighbours contribute a few micro-electronvolts, so the periodic total energy has
/// to reproduce the molecular one. This is the cross-check that the periodic path converged to the
/// *right* state and not merely to a stationary one: through 0.2.2 the unconverged run reported
/// energies that differed by up to 0.6 eV between meshes, all of them plausible.
#[test]
fn the_dilute_slab_reproduces_the_isolated_molecule() {
    use am1_rs::scf::{run_am1, Am1Options};
    let params = Am1Parameters::standard().unwrap();

    let free = run_am1(
        &Molecule::new(methane()),
        &params,
        &Am1Options {
            e_tol: 1.0e-12,
            p_tol: 1.0e-11,
            max_scf: 800,
            ..Am1Options::default()
        },
    )
    .unwrap();
    assert!(free.converged);

    let mut last = 0.0;
    for spacing in [9.0, 14.0, 20.0] {
        let r = run_pbc_scf(&methane_slab(spacing), &params, &tight([2, 2, 1])).unwrap();
        assert!(r.converged, "{spacing} A did not converge");
        let gap = r.total_ev - free.total_ev;
        eprintln!("    {spacing:5.1} A: periodic - molecular = {gap:+.3e} eV");
        last = gap.abs();
    }
    assert!(
        last < 1.0e-5,
        "at 20 A the lattice should be invisible, and it is off by {last:.2e} eV"
    );
}

/// **The reported energy belongs to the reported density**, which is what moving the energy above
/// the mixing step bought.
///
/// Checked by consistency across settings that cannot change the answer: the real-space cutoff, the
/// mesh, and time-reversal folding all leave a dilute molecular slab's energy alone to well under a
/// micro-electronvolt once it is genuinely converged. Through 0.2.2 these disagreed by up to
/// 0.42 eV, because each stopped at a different point in a mix that had not settled.
#[test]
fn settings_that_cannot_change_the_answer_do_not() {
    let params = Am1Parameters::standard().unwrap();
    let m = methane_slab(9.0);
    let mut energies = Vec::new();
    for (tag, o) in [
        ("mesh 2x2, rc 30", tight([2, 2, 1])),
        ("mesh 4x4, rc 30", tight([4, 4, 1])),
        (
            "mesh 4x4, rc 45",
            PbcOptions {
                realspace_cutoff: 45.0,
                ..tight([4, 4, 1])
            },
        ),
        (
            "mesh 4x4, folded",
            PbcOptions {
                fold_time_reversal: true,
                ..tight([4, 4, 1])
            },
        ),
        (
            "mesh 4x4, mixing 0.6",
            PbcOptions {
                mixing: 0.6,
                ..tight([4, 4, 1])
            },
        ),
        (
            "mesh 4x4, no DIIS",
            PbcOptions {
                diis_history: 0,
                max_scf: 3000,
                ..tight([4, 4, 1])
            },
        ),
    ] {
        let r = run_pbc_scf(&m, &params, &o).unwrap();
        eprintln!(
            "    {tag:22}: converged={} in {:4}, E = {:.9} eV",
            r.converged, r.iterations, r.total_ev
        );
        assert!(r.converged, "{tag} did not converge");
        energies.push((tag, r.total_ev));
    }
    let base = energies[0].1;
    for (tag, e) in &energies[1..] {
        let d = (e - base).abs();
        assert!(d < 1.0e-6, "{tag} differs from the reference by {d:.2e} eV");
    }
}

/// **The converged energy is the variational energy of the converged density**, which is only true
/// if it is evaluated somewhere idempotent.
///
/// `E[P] = ½Tr[P(H + F(P))]` is stationary on the idempotent manifold and nowhere else. The SCF's
/// working density is the *mixed* input, which is not on that manifold — and a Pulay step, being a
/// signed combination of past densities, can sit further off it than its distance to the fixed
/// point suggests. Evaluating there leaves a first-order error that the energy itself hides and
/// anything differenced does not.
///
/// So it is measured by differencing: an analytic gradient against a central difference of the
/// total energy, with and without the mixer. Both must agree with the analytic gradient to the same
/// order, because the mixer chooses the path and not the answer. Before the fix — one extra pass at
/// the output density once the tolerances are met — the Pulay run was **1.2e-6** eV/Bohr off while
/// the linear one was 3.5e-7, at identical stated tolerances. That is what a first-order error in
/// a quantity divided by `2h = 2e-4` looks like.
#[test]
fn the_converged_energy_is_differentiable_to_the_analytic_gradient() {
    use am1_rs::pbc::pbc_energy_and_gradient;

    let params = Am1Parameters::standard().unwrap();
    // Two waters in a 6 Å cube: dense enough that the image pairs matter, small enough to difference.
    let w = |o: Vec3| {
        [
            o,
            o + Vec3::new(0.9584, 0.0, 0.0) * ANG,
            o + Vec3::new(-0.2400, 0.9279, 0.0) * ANG,
        ]
    };
    let a = w(Vec3::new(0.2, 0.1, 0.0) * ANG);
    let b = w(Vec3::new(0.3, 0.4, 3.1) * ANG);
    let system = Molecule::new(
        [
            (8u8, a[0]),
            (1, a[1]),
            (1, a[2]),
            (8, b[0]),
            (1, b[1]),
            (1, b[2]),
        ]
        .into_iter()
        .map(|(z, position)| Atom { z, position })
        .collect(),
    )
    .with_cell(Lattice::cubic(6.0 * ANG).unwrap());

    let opts = |depth: usize| PbcOptions {
        kmesh: KMesh::Gamma,
        fold_time_reversal: false,
        realspace_cutoff: 40.0,
        exchange_cutoff: Some(10.0),
        smearing_ev: 0.0,
        e_tol: 1.0e-12,
        p_tol: 1.0e-11,
        max_scf: 4000,
        mixing: 0.3,
        diis_history: depth,
        ..PbcOptions::default()
    };

    // `h = 1e-4`: large enough that the `O(h²)` truncation is about 6e-8, small enough that it
    // does not mask a first-order energy error of the size being guarded against.
    let h = 1.0e-4;
    // The oxygen of the second molecule, whose three components carried the worst disagreement.
    let atom = 3usize;
    for depth in [0usize, 8] {
        let o = opts(depth);
        let (scf, grad) = pbc_energy_and_gradient(&system, &params, &o).unwrap();
        assert!(scf.converged);
        let mut worst = 0.0_f64;
        for axis in 0..3 {
            let mut plus = system.clone();
            let mut minus = system.clone();
            let shift = |v: &mut Vec3, d: f64| match axis {
                0 => v.x += d,
                1 => v.y += d,
                _ => v.z += d,
            };
            shift(&mut plus.atoms[atom].position, h);
            shift(&mut minus.atoms[atom].position, -h);
            let ep = run_pbc_scf(&plus, &params, &o).unwrap();
            let em = run_pbc_scf(&minus, &params, &o).unwrap();
            assert!(ep.converged && em.converged);
            let fd = (ep.total_ev - em.total_ev) / (2.0 * h);
            let ana = match axis {
                0 => grad.gradient[atom].x,
                1 => grad.gradient[atom].y,
                _ => grad.gradient[atom].z,
            };
            worst = worst.max((ana - fd).abs());
        }
        eprintln!(
            "    diis_history {depth}: {:3} iterations, max |analytic - FD| = {worst:.3e} eV/Bohr",
            scf.iterations
        );
        assert!(
            worst < 5.0e-7,
            "with diis_history {depth} the differenced energy is {worst:.3e} from the analytic \
             gradient; the energy is being evaluated somewhere the functional is not stationary"
        );
    }
}

/// Pulay mixing is **measured**, not assumed: it must beat plain linear mixing on the same system.
///
/// Both reach the same energy — the mixer chooses the path, not the fixed point — so the claim
/// this makes is about iteration count alone, which is the whole cost of a periodic SCF.
#[test]
fn the_pulay_mixer_earns_its_memory() {
    let params = Am1Parameters::standard().unwrap();
    let m = methane_slab(9.0);
    let run = |depth: usize| {
        let o = PbcOptions {
            diis_history: depth,
            max_scf: 3000,
            ..tight([4, 4, 1])
        };
        let r = run_pbc_scf(&m, &params, &o).unwrap();
        assert!(r.converged, "depth {depth} did not converge");
        (r.iterations, r.total_ev)
    };
    let (plain, e_plain) = run(0);
    let (pulay, e_pulay) = run(8);
    eprintln!("    linear mixing: {plain:4} iterations,  Pulay depth 8: {pulay:4} iterations");
    assert!(
        (e_plain - e_pulay).abs() < 1.0e-8,
        "the two mixers found different states: {e_plain} vs {e_pulay}"
    );
    assert!(
        pulay * 2 <= plain,
        "Pulay should at least halve the count: {pulay} against {plain}"
    );
}

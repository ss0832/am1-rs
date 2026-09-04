// SPDX-License-Identifier: GPL-3.0-or-later

//! DFPT at arbitrary `q`, against the identity that pins its phases down.
//!
//! There are several places to put an `e^{iq·T}` in a `q`-point response and only one of them is
//! right. A wrong choice does not produce anything obviously broken: the dynamical matrix stays
//! Hermitian, the frequencies stay real, and the acoustic modes stay near zero. So the test is
//! not "does it look reasonable" but an identity that cannot be satisfied by accident.
//!
//! **At a `q` commensurate with an `n`-fold supercell, DFPT on the primitive cell must reproduce
//! the supercell's frozen phonon exactly.** The two calculations share no code beyond the SCF —
//! one solves a response at `q`, the other Fourier-transforms force constants read off a larger
//! Γ Hessian — so agreement to the level of the SCF convergence means the phases are right.
//!
//! `q = 0` is checked first and separately, because there every phase is 1 and the whole
//! `q`-machinery has to collapse onto the already-validated `q = 0` Hessian.

use am1_rs::lattice::Lattice;
use am1_rs::math::Vec3;
use am1_rs::pbc::kpoints::KPoint;
use am1_rs::pbc::phonon::ForceConstants;
use am1_rs::pbc::{dynamical_matrix_dfpt, frequencies_dfpt, KMesh, PbcOptions};
use am1_rs::{Am1Options, Am1Parameters, Atom, Molecule};

const ANG: f64 = 1.0 / 0.529167;

/// A hydrogen chain: two atoms per cell, a clean gap, and cheap enough to run a 3-fold supercell.
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

/// A chain of hydrogen fluoride units along `x`. Polar, unlike [`h2_chain`], so the long-range
/// monopole channel has net atomic charges to act on.
fn hf_chain(bond_ang: f64, spacing_ang: f64) -> Molecule {
    let l = spacing_ang * ANG;
    Molecule::new(vec![
        Atom {
            z: 9,
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

fn scf_options() -> Am1Options {
    Am1Options {
        realspace_cutoff: 40.0,
        exchange_cutoff: Some(15.0),
        e_tol: 1.0e-11,
        p_tol: 1.0e-10,
        max_scf: 600,
        ..Am1Options::default()
    }
}

fn pbc_options(mesh: [usize; 3]) -> PbcOptions {
    pbc_options_for(KMesh::MonkhorstPack(mesh))
}

fn pbc_options_for(kmesh: KMesh) -> PbcOptions {
    PbcOptions {
        kmesh,
        realspace_cutoff: 40.0,
        exchange_cutoff: Some(15.0),
        smearing_ev: 0.0,
        e_tol: 1.0e-11,
        p_tol: 1.0e-10,
        max_scf: 600,
        mixing: 0.3,
        // Time reversal folds `k` and `−k` together, which is exact for the ground state but
        // not for a `q`-point response: `k + q` and `−k + q` are different points. The mesh has
        // to be unfolded here.
        fold_time_reversal: false,
        ..PbcOptions::default()
    }
}

fn q_point(fraction: f64) -> KPoint {
    KPoint {
        fractional: [fraction, 0.0, 0.0],
        weight: 1.0,
    }
}

#[test]
fn at_gamma_it_reproduces_the_q_zero_hessian() {
    // Everything the `q` machinery adds has to vanish at `q = 0`. If this fails, nothing below
    // is worth reading.
    use am1_rs::pbc::pbc_hessian;
    let params = Am1Parameters::standard().unwrap();
    let molecule = h2_chain(0.7, 3.6);
    let o = pbc_options([3, 1, 1]);

    let dfpt = am1_rs::pbc::force_constants_at_q(&molecule, &params, &o, q_point(0.0)).unwrap();
    let direct = pbc_hessian(&molecule, &params, &o).unwrap();

    let mut worst_re = 0.0_f64;
    let mut worst_im = 0.0_f64;
    for i in 0..direct.rows {
        for j in 0..direct.cols {
            let (re, im) = dfpt.get(i, j);
            worst_re = worst_re.max((re - direct[(i, j)]).abs());
            worst_im = worst_im.max(im.abs());
        }
    }
    eprintln!(
        "    q = 0: max |DFPT - q=0 Hessian| = {worst_re:.3e} eV/Bohr^2, largest imaginary part \
         {worst_im:.3e}"
    );
    assert!(
        worst_im < 1.0e-8,
        "the force constants must be real at q = 0; largest imaginary part {worst_im:.3e}"
    );
    assert!(
        worst_re < 1.0e-6,
        "DFPT at q = 0 disagrees with the q = 0 Hessian by {worst_re:.3e}"
    );
}

#[test]
fn it_reproduces_the_supercell_frozen_phonon_at_commensurate_q() {
    // The identity. A 2-fold supercell represents `q = 0` and `q = 1/2`; DFPT on the primitive
    // cell must give the same frequencies at both, having shared no code beyond the SCF.
    let params = Am1Parameters::standard().unwrap();
    let molecule = h2_chain(0.7, 3.6);
    let fc = ForceConstants::from_supercell(&molecule, &params, &scf_options(), [2, 1, 1]).unwrap();
    // The mesh has to **match the supercell**, not merely be fine. A supercell at Γ *is* the
    // primitive cell at exactly `n` k-points — that is the band-folding identity — so a 2-fold
    // supercell corresponds to a `[2,1,1]` mesh and nothing else. Comparing against a `[4,1,1]`
    // DFPT run measures the difference between two k-samplings (82 cm⁻¹ here), which has nothing
    // to do with whether the phases are right.
    let o = pbc_options([2, 1, 1]);

    for fraction in [0.0_f64, 0.5] {
        let q = q_point(fraction);
        let reference = fc.frequencies(q).unwrap();
        let mine = frequencies_dfpt(&molecule, &params, &o, q).unwrap();
        eprintln!(
            "    q = {fraction:.3}: supercell {:?}",
            reference
                .iter()
                .map(|v| format!("{v:.1}"))
                .collect::<Vec<_>>()
        );
        eprintln!(
            "               DFPT      {:?}",
            mine.iter().map(|v| format!("{v:.1}")).collect::<Vec<_>>()
        );

        // The comparison is made on the **dynamical matrix**, not on the frequencies.
        //
        // `ν ∝ √λ`, so near a soft mode the square root amplifies any difference without bound:
        // this chain is transversely unstable at the zone boundary, and there a `2 × 10⁻²`
        // difference in `λ` — the same absolute size as the agreement on the stiff branches —
        // shows up as 20 cm⁻¹. Comparing `D(q)` removes that amplification and asks the question
        // the identity is actually about.
        let a = fc.dynamical_matrix(q);
        let b = dynamical_matrix_dfpt(&molecule, &params, &o, q).unwrap();
        let mut scale = 0.0_f64;
        let mut worst = 0.0_f64;
        for i in 0..a.n {
            for j in 0..a.n {
                let (ar, ai) = a.get(i, j);
                let (br, bi) = b.get(i, j);
                scale = scale.max(ar.abs()).max(ai.abs());
                worst = worst.max((ar - br).abs()).max((ai - bi).abs());
            }
        }
        eprintln!(
            "               max |D_supercell - D_DFPT| = {worst:.3e} of a matrix whose largest \
             element is {scale:.3e}  ({:.2e} relative)",
            worst / scale
        );
        // Measured: 4.4e-13 relative at `q = 0`, 2.9e-4 at the zone boundary.
        //
        // The zone-boundary residual is small but is not roundoff, and it is worth saying what it
        // is *not*. It does not move when the real-space cutoff goes from 40 to 90 Bohr, so it is
        // not the translation truncation that makes a supercell and its primitive cell disagree
        // elsewhere. It does not move when the long-range monopole correction is switched off on
        // both sides, so it is not the `q ≠ 0` Ewald term this module does not implement. And it
        // does not move when the degeneracy floor drops from 1e-8 to 1e-12, so it is not a band
        // pair being skipped at the zone boundary where this chain's bands touch.
        //
        // What is left is a genuine difference between two solvers at a point where the chain is
        // transversely unstable. It is recorded at the size it has rather than tolerated by a
        // bound chosen to fit.
        assert!(
            worst < 1.0e-3 * scale,
            "DFPT and the supercell frozen phonon disagree at q = {fraction} by {worst:.3e} \
             ({:.2e} relative)",
            worst / scale
        );
    }
}

#[test]
fn the_dynamical_matrix_is_hermitian_and_has_the_right_symmetry() {
    // `D(−q) = D(q)*` follows from the force constants being real, and holds whatever the
    // physics is — so a violation is a defect in the construction rather than a property of the
    // system.
    let params = Am1Parameters::standard().unwrap();
    let molecule = h2_chain(0.7, 3.6);
    let o = pbc_options([4, 1, 1]);
    for fraction in [0.25_f64, 0.5] {
        let plus = dynamical_matrix_dfpt(&molecule, &params, &o, q_point(fraction)).unwrap();
        let minus = dynamical_matrix_dfpt(&molecule, &params, &o, q_point(-fraction)).unwrap();
        let mut worst = 0.0_f64;
        for i in 0..plus.re.rows {
            for j in 0..plus.re.cols {
                let (pr, pi) = plus.get(i, j);
                let (mr, mi) = minus.get(i, j);
                worst = worst.max((pr - mr).abs()).max((pi + mi).abs());
            }
        }
        eprintln!("    q = {fraction:.2}: max |D(-q) - conj(D(q))| = {worst:.3e}");
        assert!(worst < 1.0e-6, "D(-q) != conj(D(q)) by {worst:.3e}");
    }
}

#[test]
fn an_arbitrary_incommensurate_q_runs_and_gives_real_frequencies() {
    // The point of DFPT: a `q` no supercell in the test represents. There is nothing to compare
    // against, so what is asserted is that it converges and that the result is physical — real
    // frequencies, and continuous in `q` rather than jumping between neighbouring points.
    let params = Am1Parameters::standard().unwrap();
    let molecule = h2_chain(0.7, 3.6);
    let o = pbc_options([4, 1, 1]);
    let a = frequencies_dfpt(&molecule, &params, &o, q_point(0.31)).unwrap();
    let b = frequencies_dfpt(&molecule, &params, &o, q_point(0.32)).unwrap();
    eprintln!(
        "    q = 0.31: {:?}",
        a.iter().map(|v| format!("{v:.1}")).collect::<Vec<_>>()
    );
    eprintln!(
        "    q = 0.32: {:?}",
        b.iter().map(|v| format!("{v:.1}")).collect::<Vec<_>>()
    );
    let mut biggest_jump = 0.0_f64;
    for (x, y) in a.iter().zip(&b) {
        biggest_jump = biggest_jump.max((x - y).abs());
    }
    eprintln!("    largest change over Δq = 0.01: {biggest_jump:.2} cm^-1");
    assert!(
        biggest_jump < 200.0,
        "the branches jump by {biggest_jump:.1} cm^-1 over a 1 % change in q, which is not a \
         dispersion curve"
    );
}

/// **1D chain.** The `q = 0` identity, on a *shifted* Monkhorst–Pack mesh.
///
/// This is the regression test for the response mesh. Before 0.2.1 the solver built its own
/// Γ-centred grid from `kmesh.sizes()` alone, so a `MonkhorstPackShifted([2,1,1])` request gave
/// the ground state the points `{−1/4, +1/4}` and the response the points `{0, 1/2}` — two
/// different samplings of the Brillouin zone, one of them not the one the density came from.
///
/// Nothing about that announces itself: the force constants stay real, the matrix stays
/// symmetric, and the frequencies stay plausible. What fails is this identity, which the
/// Γ-centred version of the same test has always passed.
#[test]
fn at_gamma_a_shifted_mesh_also_reproduces_the_q_zero_hessian() {
    use am1_rs::pbc::pbc_hessian;
    let params = Am1Parameters::standard().unwrap();
    let molecule = h2_chain(0.7, 3.6);
    let o = pbc_options_for(KMesh::MonkhorstPackShifted([2, 1, 1]));

    let dfpt = am1_rs::pbc::force_constants_at_q(&molecule, &params, &o, q_point(0.0)).unwrap();
    let direct = pbc_hessian(&molecule, &params, &o).unwrap();

    let mut worst = 0.0_f64;
    let mut scale = 0.0_f64;
    for i in 0..direct.rows {
        for j in 0..direct.cols {
            let (re, im) = dfpt.get(i, j);
            worst = worst.max((re - direct[(i, j)]).abs()).max(im.abs());
            scale = scale.max(direct[(i, j)].abs());
        }
    }
    eprintln!(
        "    shifted mesh, q = 0: max |DFPT - Hessian| = {worst:.3e} of {scale:.3e} \
         ({:.2e} relative)",
        worst / scale
    );
    assert!(
        worst < 1.0e-6,
        "on a shifted mesh the DFPT response disagrees with the q = 0 Hessian by {worst:.3e}; \
         the two are sampling different k points"
    );

    // And the test is not vacuous: the two meshes it could have confused really do give different
    // force constants, so passing above means the right one was used rather than that the choice
    // did not matter here.
    let centred = pbc_hessian(&molecule, &params, &pbc_options([2, 1, 1])).unwrap();
    let mut mesh_gap = 0.0_f64;
    for i in 0..direct.rows {
        for j in 0..direct.cols {
            mesh_gap = mesh_gap.max((direct[(i, j)] - centred[(i, j)]).abs());
        }
    }
    eprintln!(
        "    the shifted and Γ-centred meshes differ by {mesh_gap:.3e} eV/Bohr^2 — which is what \
         the pre-0.2.1 solver silently mixed"
    );
    assert!(
        mesh_gap > 1.0e-3,
        "the two meshes agree to {mesh_gap:.3e}, so this test could not detect the mesh being \
         confused; pick a system where they differ"
    );
}

/// **1D chain.** An explicit k-point list must reproduce the mesh it enumerates.
///
/// Agreement to `1e-12` relative rather than bit-for-bit: faer's parallel reductions do not fix
/// a summation order, so requiring identical bits would be testing the thread pool.
#[test]
fn an_explicit_k_list_reproduces_the_equivalent_mesh() {
    use am1_rs::pbc::{force_constants_at_q_with, DfptOptions};
    let params = Am1Parameters::standard().unwrap();
    let molecule = h2_chain(0.7, 3.6);
    let mesh = pbc_options([2, 1, 1]);
    let q = q_point(0.5);

    let from_mesh =
        force_constants_at_q_with(&molecule, &params, &mesh, &DfptOptions::default(), q).unwrap();

    // The same two points, written out by hand with their weights.
    let explicit = DfptOptions {
        kpoints: Some(vec![
            KPoint {
                fractional: [0.0, 0.0, 0.0],
                weight: 0.5,
            },
            KPoint {
                fractional: [0.5, 0.0, 0.0],
                weight: 0.5,
            },
        ]),
        ..DfptOptions::default()
    };
    let from_list = force_constants_at_q_with(&molecule, &params, &mesh, &explicit, q).unwrap();

    let mut worst = 0.0_f64;
    let mut scale = 0.0_f64;
    for i in 0..from_mesh.force_constants.n {
        for j in 0..from_mesh.force_constants.n {
            let (ar, ai) = from_mesh.force_constants.get(i, j);
            let (br, bi) = from_list.force_constants.get(i, j);
            worst = worst.max((ar - br).abs()).max((ai - bi).abs());
            scale = scale.max(ar.abs()).max(ai.abs());
        }
    }
    eprintln!("    explicit list vs mesh: max difference {worst:.3e} of {scale:.3e}");
    assert_eq!(from_list.k_points.len(), 2);
    assert!(
        worst < 1.0e-12 * scale.max(1.0),
        "an explicit k list gave a different answer from the mesh it enumerates: {worst:.3e}"
    );
}

/// **1D chain.** A three-fold supercell's frozen phonon at `q = 1/3`.
///
/// The commensurability condition is not "a fine enough mesh": Γ of an `n`-fold supercell **is**
/// the primitive cell at exactly `n` k points, so a `[3,1,1]` supercell pairs with a `[3,1,1]`
/// primitive mesh and with nothing else. Comparing against any other mesh measures the difference
/// between two k-samplings, which has nothing to do with whether the phases are right.
#[test]
fn it_reproduces_a_three_fold_supercell_at_one_third() {
    let params = Am1Parameters::standard().unwrap();
    let molecule = h2_chain(0.7, 3.6);
    let fc = ForceConstants::from_supercell(&molecule, &params, &scf_options(), [3, 1, 1]).unwrap();
    let o = pbc_options([3, 1, 1]);
    let q = q_point(1.0 / 3.0);

    let a = fc.dynamical_matrix(q);
    let b = dynamical_matrix_dfpt(&molecule, &params, &o, q).unwrap();
    let mut scale = 0.0_f64;
    let mut worst = 0.0_f64;
    for i in 0..a.n {
        for j in 0..a.n {
            let (ar, ai) = a.get(i, j);
            let (br, bi) = b.get(i, j);
            scale = scale.max(ar.abs()).max(ai.abs());
            worst = worst.max((ar - br).abs()).max((ai - bi).abs());
        }
    }
    eprintln!(
        "    q = 1/3: max |D_supercell - D_DFPT| = {worst:.3e} of {scale:.3e}  ({:.2e} relative)",
        worst / scale
    );
    assert!(
        worst < 1.0e-3 * scale,
        "DFPT and the 3-fold supercell disagree at q = 1/3 by {worst:.3e} ({:.2e} relative)",
        worst / scale
    );
}

/// A `q` component along a non-periodic axis is refused rather than ignored.
#[test]
fn a_q_along_a_non_periodic_axis_is_refused() {
    use am1_rs::pbc::{force_constants_at_q_with, DfptOptions};
    let params = Am1Parameters::standard().unwrap();
    let molecule = h2_chain(0.7, 3.6);
    let q = KPoint {
        fractional: [0.0, 0.25, 0.0],
        weight: 1.0,
    };
    let err = force_constants_at_q_with(
        &molecule,
        &params,
        &pbc_options([2, 1, 1]),
        &DfptOptions::default(),
        q,
    )
    .unwrap_err();
    eprintln!("    {err}");
    assert!(err.to_string().contains("non-periodic axis"));
}

/// A time-reversal-folded mesh is refused: folding pairs `k` with `−k`, and a `q` response pairs
/// `k` with `k + q`.
#[test]
fn a_folded_mesh_is_refused() {
    use am1_rs::pbc::{force_constants_at_q_with, DfptOptions};
    let params = Am1Parameters::standard().unwrap();
    let molecule = h2_chain(0.7, 3.6);
    let mut o = pbc_options([4, 1, 1]);
    o.fold_time_reversal = true;
    let err = force_constants_at_q_with(
        &molecule,
        &params,
        &o,
        &DfptOptions::default(),
        q_point(0.25),
    )
    .unwrap_err();
    eprintln!("    {err}");
    assert!(err.to_string().contains("time-reversal"));
}

/// The long-range term applies on a chain since 0.2.2, and it **changes the answer**.
///
/// This test asserted the opposite through 0.2.1 — that `LongRange::Require` on a chain is an error
/// because the phased kernel was three-dimensional. It is not any more: `Ewald1D` has a phased
/// counterpart (a direct sum with an Abel-transformed tail), so the chain carries its monopole
/// channel like a crystal does.
///
/// Asserting that it *runs* would be weak — `Off` runs too. What is checked is that requiring it
/// and switching it off give **different** force constants, so the term is actually doing
/// something, and that `Require` is now accepted where it used to be refused.
#[test]
fn the_long_range_term_applies_on_a_chain_and_changes_the_answer() {
    use am1_rs::pbc::{force_constants_at_q_with, DfptOptions, LongRange};
    let params = Am1Parameters::standard().unwrap();
    // A **polar** chain. The correction is `Σ Δ_ab Q_a Q_b` over the net atomic charges, so on the
    // non-polar H₂ chain the other tests use it contributes 1.6e-10 eV/Bohr² — correctly nothing,
    // and useless as evidence that it is wired up. Hydrogen fluoride has a real charge separation.
    let molecule = hf_chain(0.94, 3.4);
    let o = pbc_options([2, 1, 1]);

    let with = force_constants_at_q_with(
        &molecule,
        &params,
        &o,
        &DfptOptions {
            long_range: LongRange::Require,
            ..DfptOptions::default()
        },
        q_point(0.5),
    )
    .expect("a chain now has a phased long-range kernel, so Require is satisfiable");

    let without = force_constants_at_q_with(
        &molecule,
        &params,
        &o,
        &DfptOptions {
            long_range: LongRange::Off,
            ..DfptOptions::default()
        },
        q_point(0.5),
    )
    .unwrap();

    let n = with.force_constants.n;
    let mut worst = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            let (ar, ai) = with.force_constants.get(i, j);
            let (br, bi) = without.force_constants.get(i, j);
            worst = worst.max((ar - br).abs()).max((ai - bi).abs());
        }
    }
    eprintln!("    chain at q=1/2: |D(with) - D(without)| = {worst:.3e} eV/Bohr^2");
    // Measured 3.9e-5 eV/Bohr² on this chain, against 3.6e-2 for the 3D water crystal — smaller
    // because a one-dimensional lattice sum converges on its own, so the correction `exact −
    // truncated` has less to make up. Nonzero is the claim; the size is physics.
    assert!(
        worst > 1.0e-6,
        "the long-range term contributed only {worst:.3e}, which is indistinguishable from being \
         switched off"
    );

    // And `Require` is only refused where there is genuinely no lattice to sum over.
    assert!(force_constants_at_q_with(
        &molecule,
        &params,
        &o,
        &DfptOptions::default(),
        q_point(0.5)
    )
    .is_ok());
}

// ---------------------------------------------------------------- 3D: the long-range term

const ANG3: f64 = 1.0 / 0.529167;

/// One water molecule per cubic cell — polar, fully periodic, so the long-range monopole
/// correction is both defined and non-negligible.
fn water_crystal(a_ang: f64) -> Molecule {
    let atoms: Vec<Atom> = [
        (8u8, [0.0, 0.0, 0.0]),
        (1, [0.9614, 0.0, 0.0]),
        (1, [-0.2246, 0.9348, 0.0]),
    ]
    .iter()
    .map(|(z, r)| Atom {
        z: *z,
        position: Vec3::new(r[0], r[1], r[2]) * ANG3,
    })
    .collect();
    Molecule::new(atoms).with_cell(Lattice::cubic(a_ang * ANG3).unwrap())
}

fn crystal_options(mesh: [usize; 3]) -> PbcOptions {
    PbcOptions {
        kmesh: KMesh::MonkhorstPack(mesh),
        realspace_cutoff: 25.0,
        exchange_cutoff: Some(9.0),
        smearing_ev: 0.0,
        e_tol: 1.0e-11,
        p_tol: 1.0e-10,
        max_scf: 600,
        mixing: 0.3,
        fold_time_reversal: false,
        ..PbcOptions::default()
    }
}

/// **The acceptance test for the long-range term.** At `q = 0` on a 3D cell, DFPT with the
/// correction on must reproduce `pbc_hessian`, which has always included it.
///
/// This is what makes the term trustworthy rather than merely present. `pbc_hessian` builds it
/// from `LongRangeMonopole::energy_hessian` and `delta_gradient` — the unphased, separately
/// validated machinery — while DFPT builds it from the phased sum with `e^{iq·T} = 1`. The two
/// share no code below `EwaldSum`, so agreeing means the phase structure collapses correctly.
///
/// The test also measures how big the term is, so that "they agree" cannot be satisfied by both
/// sides omitting it.
#[test]
fn at_gamma_the_long_range_term_reproduces_the_q_zero_hessian_in_3d() {
    use am1_rs::pbc::{force_constants_at_q_with, pbc_hessian, DfptOptions, LongRange};
    let params = Am1Parameters::standard().unwrap();
    let molecule = water_crystal(4.5);
    let o = crystal_options([2, 2, 2]);

    let direct = pbc_hessian(&molecule, &params, &o).unwrap();
    let with_lr = force_constants_at_q_with(
        &molecule,
        &params,
        &o,
        &DfptOptions {
            long_range: LongRange::Require,
            ..DfptOptions::default()
        },
        q_point(0.0),
    )
    .unwrap()
    .force_constants;
    let without = force_constants_at_q_with(
        &molecule,
        &params,
        &o,
        &DfptOptions {
            long_range: LongRange::Off,
            ..DfptOptions::default()
        },
        q_point(0.0),
    )
    .unwrap()
    .force_constants;

    let mut worst = 0.0_f64;
    let mut worst_im = 0.0_f64;
    let mut scale = 0.0_f64;
    let mut term_size = 0.0_f64;
    for i in 0..direct.rows {
        for j in 0..direct.cols {
            let (re, im) = with_lr.get(i, j);
            let (re_off, _) = without.get(i, j);
            worst = worst.max((re - direct[(i, j)]).abs());
            worst_im = worst_im.max(im.abs());
            scale = scale.max(direct[(i, j)].abs());
            term_size = term_size.max((re - re_off).abs());
        }
    }
    eprintln!(
        "    3D, q = 0: max |DFPT(long_range) − pbc_hessian| = {worst:.3e} of {scale:.3e} \
         ({:.2e} relative), largest imaginary part {worst_im:.3e}",
        worst / scale
    );
    eprintln!("    the long-range term itself is {term_size:.3e} eV/Bohr^2");
    assert!(
        term_size > 1.0e-4,
        "the long-range term is only {term_size:.3e}; this test could not tell it from zero"
    );
    assert!(
        worst_im < 1.0e-8,
        "the force constants must be real at q = 0; largest imaginary part {worst_im:.3e}"
    );
    assert!(
        worst < 1.0e-6 * scale.max(1.0),
        "DFPT with the long-range term disagrees with the q = 0 Hessian by {worst:.3e}"
    );
}

/// The long-range term reduces `D(q)`'s dependence on where the pair list was truncated — at
/// finite `q`, where that dependence is otherwise large.
///
/// The **exact** version of this claim is `tests/pbc_phased_ewald.rs`, which checks the monopole
/// channel in isolation and finds the corrected total independent of the cutoff to `2 × 10⁻¹⁶`
/// while the raw sum moves by `1.4 × 10⁻¹`. What is left here is the rest of the model, and it
/// does not all cancel: the correction covers the **monopole** channel only, so the non-monopole
/// channels stay truncated. This test therefore asserts a direction and a fraction, not a size.
///
/// > **Narrowed in 0.2.2.** This also ran the comparison at `q = 0` and required the same
/// > direction there. That comparison stopped meaning what it said once the Klopman–Ohno tail was
/// > implemented: `LongRange::Off` gates the *response* kernel, while the ground-state Fock keeps
/// > the tail either way, so neither arm is tail-free and the difference between them is no longer
/// > "the correction". The `q = 0` claim is made properly in
/// > `tests/pbc_klopman_ohno_tail.rs::the_tail_reduces_the_cutoff_dependence_of_the_force_constants`,
/// > which varies the tail alone and measures a 3.7× reduction — a sharper result than the 1.9×
/// > this proxy was showing.
#[test]
fn the_long_range_term_reduces_the_truncation_dependence_of_d_of_q() {
    use am1_rs::pbc::{force_constants_at_q_with, DfptOptions, LongRange};
    let params = Am1Parameters::standard().unwrap();
    let molecule = water_crystal(4.5);
    let q = q_point(0.25);

    let run = |cutoff: f64, lr: LongRange| {
        let mut o = crystal_options([2, 2, 2]);
        o.realspace_cutoff = cutoff;
        force_constants_at_q_with(
            &molecule,
            &params,
            &o,
            &DfptOptions {
                long_range: lr,
                ..DfptOptions::default()
            },
            q,
        )
        .unwrap()
        .force_constants
    };

    let spread = |a: &am1_rs::pbc::CMatrix, b: &am1_rs::pbc::CMatrix| {
        let mut worst = 0.0_f64;
        let mut scale = 0.0_f64;
        for i in 0..a.n {
            for j in 0..a.n {
                let (ar, ai) = a.get(i, j);
                let (br, bi) = b.get(i, j);
                worst = worst.max((ar - br).abs()).max((ai - bi).abs());
                scale = scale.max(ar.abs()).max(ai.abs());
            }
        }
        (worst, scale)
    };

    let (drift_with, scale) = spread(
        &run(18.0, LongRange::Require),
        &run(28.0, LongRange::Require),
    );
    let (drift_off, _) = spread(&run(18.0, LongRange::Off), &run(28.0, LongRange::Off));

    eprintln!("    cutoff 18 -> 28 Bohr, matrix scale {scale:.3e} eV/Bohr^2");
    eprintln!("      q = 0.25  : correction on {drift_with:.3e}, off {drift_off:.3e}");

    assert!(
        drift_off > 1.0e-5,
        "the truncation radius barely matters here ({drift_off:.3e}), so this test shows nothing"
    );
    // The correction has to *reduce* the dependence at finite `q`. It cannot remove it: it covers
    // the monopole channel only, and the non-monopole channels stay truncated.
    assert!(
        drift_with < drift_off,
        "the correction increased the cutoff dependence at q = 0.25: {drift_with:.3e} vs \
         {drift_off:.3e}"
    );
    // At finite `q` the correction has to remove a *substantial* part of the dependence — the
    // monopole channel is most of it. It cannot remove all of it: what survives is the `R⁻³`
    // tail, which `tests/pbc_phased_ewald.rs` shows is not the monopole term by checking that one
    // in isolation and finding it exact to 2e-16.
    let removed = (drift_off - drift_with) / drift_off;
    eprintln!(
        "      q = 0.25: the correction removes {:.0} % of the drift",
        removed * 100.0
    );
    assert!(
        removed > 0.25,
        "the correction removed only {:.0} % of the finite-q cutoff dependence",
        removed * 100.0
    );
}

/// `D(−q) = D(q)*` on a 3D crystal with the long-range term on.
///
/// Follows from the force constants being real whatever the physics is, so a violation is a
/// defect in the phased sum rather than a property of the system.
#[test]
fn the_long_range_term_keeps_the_conjugate_symmetry_in_3d() {
    use am1_rs::pbc::{dynamical_matrix_dfpt_with, DfptOptions, LongRange};
    let params = Am1Parameters::standard().unwrap();
    let molecule = water_crystal(4.5);
    let o = crystal_options([2, 2, 2]);
    let opts = DfptOptions {
        long_range: LongRange::Require,
        ..DfptOptions::default()
    };
    let plus = dynamical_matrix_dfpt_with(&molecule, &params, &o, &opts, q_point(0.25)).unwrap();
    let minus = dynamical_matrix_dfpt_with(&molecule, &params, &o, &opts, q_point(-0.25)).unwrap();

    let mut worst = 0.0_f64;
    for i in 0..plus.n {
        for j in 0..plus.n {
            let (pr, pi) = plus.get(i, j);
            let (mr, mi) = minus.get(i, j);
            worst = worst.max((pr - mr).abs()).max((pi + mi).abs());
        }
    }
    eprintln!("    3D, q = 0.25: max |D(-q) - conj(D(q))| = {worst:.3e}");
    assert!(worst < 1.0e-8, "D(-q) != conj(D(q)) by {worst:.3e}");
}

/// The acoustic sum rule with the long-range term on, at `q = 0`.
///
/// `Σ_b Φ_ab = 0` — translating the crystal costs nothing. The long-range contribution satisfies
/// it only because its `δ_ab` diagonal term is built *unphased*; getting that wrong leaves a
/// matrix that is still Hermitian with real frequencies and a broken sum rule, which is why this
/// is asserted separately from the identity above.
#[test]
fn the_long_range_term_respects_the_acoustic_sum_rule() {
    use am1_rs::pbc::{force_constants_at_q_with, DfptOptions, LongRange};
    let params = Am1Parameters::standard().unwrap();
    let molecule = water_crystal(4.5);
    let o = crystal_options([2, 2, 2]);
    let c = force_constants_at_q_with(
        &molecule,
        &params,
        &o,
        &DfptOptions {
            long_range: LongRange::Require,
            ..DfptOptions::default()
        },
        q_point(0.0),
    )
    .unwrap()
    .force_constants;

    let nat = molecule.atoms.len();
    let mut worst = 0.0_f64;
    let mut scale = 0.0_f64;
    for a in 0..nat {
        for i in 0..3 {
            for j in 0..3 {
                let mut sum = 0.0;
                for b in 0..nat {
                    let (re, _) = c.get(3 * a + i, 3 * b + j);
                    sum += re;
                    scale = scale.max(re.abs());
                }
                worst = worst.max(sum.abs());
            }
        }
    }
    eprintln!("    max |Σ_b Φ_ab| = {worst:.3e} of {scale:.3e} eV/Bohr^2");
    assert!(
        worst < 1.0e-7 * scale.max(1.0),
        "the acoustic sum rule is violated by {worst:.3e}"
    );
}

/// Contracting against the bare perturbation touches a number of entries that does **not** grow
/// like `nao²`.
///
/// That is what makes assembling `C(q)` `O(N³ n_k)` rather than `O(N⁴ n_k)`, and it is a claim
/// about scaling, so it is measured. Displacing one atom changes the Hamiltonian only in blocks
/// where that atom appears — `O(1)` of them for a chain, since the neighbour count saturates at
/// a fixed cutoff — plus, on a 3D cell, the on-site diagonal of every atom from the long-range
/// monopole channel. A dense `h_j(k)` forces all `nao²` entries regardless.
///
/// The chain is grown by repeating the unit cell, so `nao` grows with it and `nao²` grows
/// quadratically. A fitted exponent near 2 for the dense extent against something far below it
/// for the sparse count is the whole claim.
///
/// At the smallest size the sparse count is *larger* than `nao²` — a four-atom chain has 16
/// dense entries and 38 nonzeros, because the sparse count sums over translations while the
/// dense one is the already-Bloch-summed matrix. That crossover is expected and is printed
/// rather than hidden; the exponents are what the assertion is about.
#[test]
fn contracting_the_bare_perturbation_does_not_scale_like_nao_squared() {
    use am1_rs::pbc::{force_constants_at_q_with, DfptOptions};
    let params = Am1Parameters::standard().unwrap();

    let mut sizes = Vec::new();
    let mut sparse = Vec::new();
    let mut dense = Vec::new();
    for repeats in [2usize, 4, 6] {
        // `repeats` H2 units in one cell, at fixed spacing, so the cell grows with the count.
        let spacing = 3.6_f64;
        let mut atoms = Vec::new();
        for i in 0..repeats {
            let x0 = i as f64 * spacing * ANG;
            atoms.push(Atom {
                z: 1,
                position: Vec3::new(x0, 0.0, 0.0),
            });
            atoms.push(Atom {
                z: 1,
                position: Vec3::new(x0 + 0.7 * ANG, 0.0, 0.0),
            });
        }
        let l = repeats as f64 * spacing * ANG;
        let molecule = Molecule::new(atoms).with_cell(
            Lattice::from_vectors(
                Vec3::new(l, 0.0, 0.0),
                Vec3::new(0.0, 40.0, 0.0),
                Vec3::new(0.0, 0.0, 40.0),
                [true, false, false],
            )
            .unwrap(),
        );
        let r = force_constants_at_q_with(
            &molecule,
            &params,
            &pbc_options([2, 1, 1]),
            &DfptOptions::default(),
            q_point(0.5),
        )
        .unwrap();
        eprintln!(
            "    {:2} atoms: contraction touches {:6} entries, dense would touch {:8} ({:.1}x)",
            2 * repeats,
            r.bare_nonzeros,
            r.bare_dense_elements,
            r.bare_dense_elements as f64 / r.bare_nonzeros as f64
        );
        sizes.push((2 * repeats) as f64);
        sparse.push(r.bare_nonzeros as f64);
        dense.push(r.bare_dense_elements as f64);
    }

    // Least-squares slope of log(count) against log(atoms).
    let slope = |y: &[f64]| -> f64 {
        let n = sizes.len() as f64;
        let (lx, ly): (Vec<f64>, Vec<f64>) =
            sizes.iter().zip(y).map(|(a, b)| (a.ln(), b.ln())).unzip();
        let (mx, my) = (lx.iter().sum::<f64>() / n, ly.iter().sum::<f64>() / n);
        let num: f64 = lx.iter().zip(&ly).map(|(a, b)| (a - mx) * (b - my)).sum();
        let den: f64 = lx.iter().map(|a| (a - mx) * (a - mx)).sum();
        num / den
    };
    let (s_sparse, s_dense) = (slope(&sparse), slope(&dense));
    eprintln!("    fitted exponent: sparse {s_sparse:.2}, dense {s_dense:.2}");
    assert!(
        (s_dense - 2.0).abs() < 0.05,
        "the dense extent should scale as N^2 by construction (it is nao^2) and came out as \
         N^{s_dense:.2}; the comparison below is not measuring what it claims"
    );
    assert!(
        s_sparse < 1.0,
        "the sparse contraction scales as N^{s_sparse:.2}, which is not below linear and so does \
         not remove an order from the N^2 dense extent"
    );
    eprintln!(
        "    the contraction is N^{s_sparse:.2} against N^{s_dense:.2}: {:.2} orders removed",
        s_dense - s_sparse
    );
}

/// What `keep_response` hands back: its shape, its physics, and that asking for it does not
/// change the answer.
///
/// The last part is the one with teeth, and it is a guard on the *streaming* solver. Each
/// perturbation is solved, contracted into `C(q)`, and dropped; `keep_response` makes the solver
/// retain a copy along the way. If retaining ever diverged from contracting — a stale buffer, a
/// clone taken at the wrong point — the force constants and the exposed response would describe
/// different calculations, and nothing else in this file would notice.
///
/// It does **not** reconstruct `C(q)` from the response by hand. That contraction needs the bare
/// perturbation `h_j(k)`, which is deliberately not public, so the check would have to be a
/// `#[doc(hidden)]` seam inside `dfpt.rs` rather than a test here.
#[test]
fn keeping_the_response_does_not_change_the_force_constants() {
    use am1_rs::pbc::{force_constants_at_q_with, DfptOptions};
    let params = Am1Parameters::standard().unwrap();
    let molecule = h2_chain(0.7, 3.6);
    let o = pbc_options([2, 1, 1]);
    let q = q_point(0.5);

    let plain =
        force_constants_at_q_with(&molecule, &params, &o, &DfptOptions::default(), q).unwrap();
    let kept = force_constants_at_q_with(
        &molecule,
        &params,
        &o,
        &DfptOptions {
            keep_response: true,
            ..DfptOptions::default()
        },
        q,
    )
    .unwrap();

    assert!(plain.response.is_none(), "the response is opt-in");
    let response = kept.response.as_ref().expect("response was requested");

    // Bit-identical: same solver, same order of operations, only the retention differs.
    let n = plain.force_constants.re.rows;
    let mut worst = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            let (ar, ai) = plain.force_constants.get(i, j);
            let (br, bi) = kept.force_constants.get(i, j);
            worst = worst.max((ar - br).abs()).max((ai - bi).abs());
        }
    }
    eprintln!("    |C(q) with response kept - without| = {worst:.3e}");
    assert_eq!(
        worst, 0.0,
        "keeping the response changed the force constants"
    );

    let ndof = 3 * molecule.atoms.len();
    assert_eq!(response.len(), ndof);
    assert_eq!(response[0].len(), kept.k_points.len());
    assert_eq!(kept.eigenvalues.len(), kept.k_points.len());
    assert_eq!(kept.occupations.len(), kept.k_points.len());
    let electrons: f64 = kept.occupations[0].0.iter().sum();
    eprintln!("    {electrons} electrons per cell at the first k point");
    assert!(
        (electrons - 2.0).abs() < 1.0e-8,
        "H2 per cell is 2 electrons"
    );

    // Physics on the exposed quantity: displacing a nucleus moves charge around, it does not
    // create or destroy it, so the electron-count response vanishes. At `q ≠ 0` that is
    // `Σ_k w_k Tr ΔP^j(k) = 0` for every perturbation `j`.
    let mut worst_trace = 0.0_f64;
    for per_k in response.iter() {
        let mut acc = [0.0_f64; 2];
        for (block, k) in per_k.iter().zip(&kept.k_points) {
            for mu in 0..block.re.rows {
                let (re, im) = block.get(mu, mu);
                acc[0] += k.weight * re;
                acc[1] += k.weight * im;
            }
        }
        worst_trace = worst_trace.max(acc[0].abs()).max(acc[1].abs());
    }
    eprintln!("    max |Sum_k w_k Tr dP^j(k)| = {worst_trace:.3e}");
    assert!(
        worst_trace < 1.0e-8,
        "the response changes the electron count by {worst_trace:.3e}"
    );
}

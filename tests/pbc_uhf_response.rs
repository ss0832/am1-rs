// SPDX-License-Identifier: GPL-3.0-or-later

//! The **open-shell** k-point periodic response.
//!
//! Refused through 0.2.1 with "the k-point periodic response is restricted-only; run the Gamma
//! path for an open-shell system". No sibling crate had it either -- pm6-rs and pm7-rs refuse in
//! the same place -- so this is the one item of the 0.2.2 list with no port to work from.
//!
//! # What had to change
//!
//! The restricted response solves one CPHF. The unrestricted one solves two, **coupled**: the
//! kernel is
//!
//! ```text
//! G^s(dP) = J(dP_total) - K(dP_s)
//! ```
//!
//! so alpha reads beta's response density through the Coulomb half. Solving the two channels
//! independently would silently drop `J(dP_beta)` from `G^alpha` and return a plausible number.
//!
//! Three factor conventions had to move with it, and each is a place a wrong answer would look
//! right:
//!
//! * one orbital holds 2 electrons restricted and 1 per channel unrestricted, which scales every
//!   response density;
//! * the exchange contracts `P` at half weight restricted and `P^s` at full weight unrestricted,
//!   in **both** the skeleton and the perturbed Fock;
//! * the relaxation term's factor 4 -- two spins times the occupied-virtual/virtual-occupied pair
//!   -- becomes 2 per channel.
//!
//! # What is asserted
//!
//! **Forcing UHF on a closed shell must reproduce the restricted answer.** That is the sharp test:
//! the two go through different code from the ground-state Fock onward, and on a closed shell
//! `P^alpha = P^beta = P/2` makes them algebraically identical. Any one of the three factors above
//! being wrong breaks it, and breaks it by a large amount rather than a subtle one.
//!
//! A genuine open shell against a finite difference is the second test, because the first cannot
//! see a term that vanishes when `P^alpha = P^beta`.

use am1_rs::lattice::Lattice;
use am1_rs::math::Vec3;
use am1_rs::pbc::{born_charges, pbc_hessian, KMesh, PbcOptions};
use am1_rs::{Am1Parameters, Atom, Molecule};

const ANG: f64 = 1.0 / 0.529167;

/// A hydrogen-fluoride chain: closed shell, polar, and small enough to differentiate.
fn hf_chain(n: usize, spacing_ang: f64) -> Molecule {
    let step = spacing_ang * ANG;
    let mut atoms = Vec::new();
    for cell in 0..n {
        let x = step * cell as f64;
        atoms.push(Atom {
            z: 9,
            position: Vec3::new(x, 0.0, 0.0),
        });
        atoms.push(Atom {
            z: 1,
            position: Vec3::new(x + 0.94 * ANG, 0.1, 0.0),
        });
    }
    Molecule::new(atoms).with_cell(
        Lattice::from_vectors(
            Vec3::new(step * n as f64, 0.0, 0.0),
            Vec3::new(0.0, 30.0, 0.0),
            Vec3::new(0.0, 0.0, 30.0),
            [true, false, false],
        )
        .unwrap(),
    )
}

/// A methyl-radical chain: a doublet with one well-separated singly-occupied orbital.
fn methyl_chain(n: usize, spacing_ang: f64) -> Molecule {
    let step = spacing_ang * ANG;
    let mut atoms = Vec::new();
    for cell in 0..n {
        let shift = Vec3::new(step * cell as f64, 0.0, 0.0);
        for (z, r) in [
            (6u8, [0.0, 0.0, 0.0]),
            (1, [1.0790, 0.0, 0.0]),
            (1, [-0.5395, 0.9344, 0.0]),
            (1, [-0.5395, -0.9344, 0.0]),
        ] {
            atoms.push(Atom {
                z,
                position: Vec3::new(r[0], r[1], r[2]) * ANG + shift,
            });
        }
    }
    Molecule::new(atoms).with_cell(
        Lattice::from_vectors(
            Vec3::new(step * n as f64, 0.0, 0.0),
            Vec3::new(0.0, 30.0, 0.0),
            Vec3::new(0.0, 0.0, 30.0),
            [true, false, false],
        )
        .unwrap(),
    )
}

fn options(grid: usize) -> PbcOptions {
    PbcOptions {
        kmesh: KMesh::MonkhorstPack([grid, 1, 1]),
        fold_time_reversal: false,
        realspace_cutoff: 30.0,
        exchange_cutoff: Some(12.0),
        smearing_ev: 0.0,
        e_tol: 1.0e-11,
        p_tol: 1.0e-10,
        max_scf: 800,
        mixing: 0.3,
        ..PbcOptions::default()
    }
}

fn worst(a: &am1_rs::linalg::Matrix, b: &am1_rs::linalg::Matrix) -> f64 {
    let mut w = 0.0_f64;
    for i in 0..a.rows {
        for j in 0..a.cols {
            w = w.max((a[(i, j)] - b[(i, j)]).abs());
        }
    }
    w
}

/// The sharp one: forcing UHF on a closed shell must give the restricted force constants back.
///
/// With a **k-point mesh**, not just at Gamma, because half the ways to get the spin bookkeeping
/// wrong survive at Gamma -- where every block carries phase 1 -- and only show up once the two
/// channels' k-resolved densities have to be combined with their own phases.
#[test]
fn forcing_uhf_on_a_closed_shell_reproduces_the_restricted_hessian() {
    let params = Am1Parameters::standard().unwrap();
    let molecule = hf_chain(2, 3.4);

    let restricted = pbc_hessian(&molecule, &params, &options(3)).unwrap();
    let unrestricted = pbc_hessian(
        &molecule,
        &params,
        &PbcOptions {
            unrestricted: true,
            ..options(3)
        },
    )
    .unwrap();

    let scale = restricted
        .as_slice()
        .iter()
        .fold(0.0_f64, |m, v| m.max(v.abs()));
    let diff = worst(&restricted, &unrestricted);
    eprintln!(
        "    3 k-points: max |RHF - forced UHF| = {diff:.3e} eV/Bohr^2 of {scale:.3e} \
         ({:.1e} relative)",
        diff / scale
    );
    assert!(
        scale > 1.0,
        "the force constants are only {scale:.3e}, so this comparison shows nothing"
    );
    assert!(
        diff < 1.0e-7 * scale,
        "forced UHF disagrees with RHF by {diff:.3e} eV/Bohr^2"
    );
}

/// The same identity for the Born charges, which read the response density rather than `U` and
/// therefore exercise a different factor.
#[test]
fn forcing_uhf_on_a_closed_shell_reproduces_the_restricted_born_charges() {
    let params = Am1Parameters::standard().unwrap();
    let molecule = hf_chain(2, 3.4);

    let restricted = born_charges(&molecule, &params, &options(3)).unwrap();
    let unrestricted = born_charges(
        &molecule,
        &params,
        &PbcOptions {
            unrestricted: true,
            ..options(3)
        },
    )
    .unwrap();

    let mut diff = 0.0_f64;
    let mut scale = 0.0_f64;
    for (r, u) in restricted.iter().zip(&unrestricted) {
        for i in 0..3 {
            for j in 0..3 {
                diff = diff.max((r[i][j] - u[i][j]).abs());
                scale = scale.max(r[i][j].abs());
            }
        }
    }
    eprintln!("    max |Z*(RHF) - Z*(forced UHF)| = {diff:.3e} e of {scale:.3e}");
    assert!(scale > 0.1, "the Born charges are too small to compare");
    assert!(
        diff < 1.0e-7 * scale,
        "forced UHF disagrees with RHF on Z* by {diff:.3e} e"
    );
}

/// A **genuine** open shell against a finite difference of the analytic gradient.
///
/// The forced-UHF identity above cannot see any term that vanishes when `P^alpha = P^beta` -- the
/// whole spin-polarized part of the kernel. This can.
#[test]
fn the_open_shell_hessian_matches_finite_differences() {
    use am1_rs::pbc::{pbc_energy_and_gradient, pbc_gradient, run_pbc_scf};
    let params = Am1Parameters::standard().unwrap();
    let molecule = methyl_chain(1, 4.2);
    let opts = PbcOptions {
        multiplicity: 2,
        unrestricted: true,
        ..options(3)
    };

    let scf = run_pbc_scf(&molecule, &params, &opts).unwrap();
    assert!(
        scf.spin_density.is_some(),
        "this must be an open shell, or the test is the restricted one again"
    );
    // Not needed for the comparison, only to confirm the gradient path agrees this is open-shell.
    let _ = pbc_gradient(&molecule, &params, &opts, &scf).unwrap();

    let analytic = pbc_hessian(&molecule, &params, &opts).unwrap();

    let nat = molecule.atoms.len();
    let h = 1.0e-4;
    let mut numeric = am1_rs::linalg::Matrix::zeros(3 * nat, 3 * nat);
    for atom in 0..nat {
        for axis in 0..3 {
            let shifted = |d: f64| {
                let mut m = molecule.clone();
                let p = &mut m.atoms[atom].position;
                match axis {
                    0 => p.x += d,
                    1 => p.y += d,
                    _ => p.z += d,
                }
                m
            };
            let (_, gp) = pbc_energy_and_gradient(&shifted(h), &params, &opts).unwrap();
            let (_, gm) = pbc_energy_and_gradient(&shifted(-h), &params, &opts).unwrap();
            for other in 0..nat {
                for k in 0..3 {
                    let c = |v: &Vec3| match k {
                        0 => v.x,
                        1 => v.y,
                        _ => v.z,
                    };
                    numeric[(3 * other + k, 3 * atom + axis)] =
                        (c(&gp.gradient[other]) - c(&gm.gradient[other])) / (2.0 * h);
                }
            }
        }
    }

    let scale = analytic
        .as_slice()
        .iter()
        .fold(0.0_f64, |m, v| m.max(v.abs()));
    let diff = worst(&analytic, &numeric);
    eprintln!(
        "    open-shell chain: max |analytic - finite difference| = {diff:.3e} eV/Bohr^2 \
         of {scale:.3e}"
    );
    assert!(
        diff < 2.0e-3 * scale,
        "the open-shell periodic Hessian is off by {diff:.3e} eV/Bohr^2"
    );
}

/// An open-shell response must not silently answer with the restricted equations, and the way to
/// show it does not is that the two **disagree** on a system where they should.
///
/// Without this, every assertion above would still pass if `unrestricted` were quietly ignored.
#[test]
fn the_open_shell_answer_is_not_the_restricted_one() {
    let params = Am1Parameters::standard().unwrap();
    let molecule = methyl_chain(1, 4.2);

    let doublet = pbc_hessian(
        &molecule,
        &params,
        &PbcOptions {
            multiplicity: 2,
            unrestricted: true,
            ..options(3)
        },
    )
    .unwrap();
    // The same nuclei solved as a closed shell: a different physical state, so a different
    // Hessian. If the open-shell path were the restricted one in disguise these would agree.
    let closed = pbc_hessian(
        &molecule,
        &params,
        &PbcOptions {
            charge: -1.0,
            ..options(3)
        },
    )
    .unwrap();

    let scale = doublet
        .as_slice()
        .iter()
        .fold(0.0_f64, |m, v| m.max(v.abs()));
    let diff = worst(&doublet, &closed);
    eprintln!("    doublet vs closed-shell anion: max difference {diff:.3e} of {scale:.3e}");
    assert!(
        diff > 0.01 * scale,
        "the open-shell and closed-shell force constants differ by only {diff:.3e}, which \
         suggests the unrestricted path is not being taken"
    );
}

/// A **cubic** cell holding one methyl radical: fully periodic, so `epsilon_infinity` applies.
fn methyl_box(a_ang: f64) -> Molecule {
    let a = a_ang * ANG;
    let atoms: Vec<Atom> = [
        (6u8, [0.0, 0.0, 0.0]),
        (1, [1.0790, 0.0, 0.0]),
        (1, [-0.5395, 0.9344, 0.0]),
        (1, [-0.5395, -0.9344, 0.0]),
    ]
    .iter()
    .map(|(z, r)| Atom {
        z: *z,
        position: Vec3::new(r[0], r[1], r[2]) * ANG,
    })
    .collect();
    Molecule::new(atoms).with_cell(Lattice::cubic(a).unwrap())
}

fn box_options() -> PbcOptions {
    PbcOptions {
        kmesh: KMesh::Gamma,
        realspace_cutoff: 30.0,
        exchange_cutoff: Some(12.0),
        smearing_ev: 0.0,
        e_tol: 1.0e-12,
        p_tol: 1.0e-11,
        max_scf: 800,
        ..PbcOptions::default()
    }
}

/// Forcing UHF on a closed shell must give the restricted polarizability and `epsilon_infinity`
/// back.
///
/// The field response goes through the same two-channel CPHF as the phonon response since 0.2.2,
/// but it reaches it by a different bare perturbation -- the dipole operator rather than
/// `dF/dR` -- and it reads the response density rather than `U`. So this is a distinct check, not
/// a restatement of the Hessian one.
#[test]
fn forcing_uhf_on_a_closed_shell_reproduces_the_restricted_dielectric_tensor() {
    use am1_rs::pbc::dielectric_tensor;
    let params = Am1Parameters::standard().unwrap();
    // A water in a small-ish cube, so the polarizability is not vanishingly small.
    let cell = {
        let a = 7.0 * ANG;
        Molecule::new(vec![
            Atom {
                z: 8,
                position: Vec3::zero(),
            },
            Atom {
                z: 1,
                position: Vec3::new(0.9614, 0.0, 0.0) * ANG,
            },
            Atom {
                z: 1,
                position: Vec3::new(-0.2246, 0.9348, 0.0) * ANG,
            },
        ])
        .with_cell(Lattice::cubic(a).unwrap())
    };

    let restricted = dielectric_tensor(&cell, &params, &box_options()).unwrap();
    let unrestricted = dielectric_tensor(
        &cell,
        &params,
        &PbcOptions {
            unrestricted: true,
            ..box_options()
        },
    )
    .unwrap();

    let mut da = 0.0_f64;
    let mut de = 0.0_f64;
    let mut scale = 0.0_f64;
    for i in 0..3 {
        for j in 0..3 {
            da = da.max((restricted.0[i][j] - unrestricted.0[i][j]).abs());
            de = de.max((restricted.1[i][j] - unrestricted.1[i][j]).abs());
            scale = scale.max(restricted.0[i][j].abs());
        }
    }
    eprintln!(
        "    alpha: max |RHF - forced UHF| = {da:.3e} Bohr^3 of {scale:.3e};  epsilon: {de:.3e}"
    );
    assert!(scale > 1.0, "the polarizability is too small to compare");
    assert!(
        da < 1.0e-8 * scale && de < 1.0e-8,
        "forced UHF disagrees with RHF on the dielectric response: alpha {da:.3e}, eps {de:.3e}"
    );
}

/// **The magnitude check for an open shell.** A radical alone in a large box must have the
/// isolated radical's polarizability.
///
/// The forced-UHF identity above cannot see anything that vanishes when `Palpha = Pbeta`. This
/// can, and it compares two genuinely independent routes: the periodic side is an analytic
/// two-channel CPHF under Bloch boundary conditions, the molecular side is finite differences of
/// six extra UHF solves. They share the SCF and the dipole operator and nothing else.
///
/// The shape tests elsewhere -- symmetry, positive diagonal, origin independence -- are all
/// satisfied by a polarizability that is wrong by a constant factor. This is the one that is not.
#[test]
fn an_open_shell_radical_in_a_box_has_the_isolated_radical_polarizability() {
    use am1_rs::constants::{AU_DIPOLE_TO_DEBYE, HARTREE_TO_EV};
    use am1_rs::pbc::dielectric_tensor;
    use am1_rs::scf::{run_am1, Am1Options, ScfReference};

    let params = Am1Parameters::standard().unwrap();
    let cell = methyl_box(12.0);
    let periodic = dielectric_tensor(
        &cell,
        &params,
        &PbcOptions {
            multiplicity: 2,
            unrestricted: true,
            ..box_options()
        },
    )
    .unwrap();

    // The same radical, isolated, by finite differences of the UHF dipole.
    let molecule = Molecule::new(cell.atoms.clone());
    let base = Am1Options {
        multiplicity: 2,
        reference: ScfReference::Unrestricted,
        e_tol: 1.0e-12,
        p_tol: 1.0e-11,
        max_scf: 800,
        ..Am1Options::default()
    };
    let h = 2.0e-4;
    let mut fd = [[0.0_f64; 3]; 3];
    #[allow(clippy::needless_range_loop)]
    for beta in 0..3 {
        let dipole = |sign: f64| -> [f64; 3] {
            let mut f = Vec3::zero();
            match beta {
                0 => f.x = sign * h,
                1 => f.y = sign * h,
                _ => f.z = sign * h,
            }
            let r = run_am1(
                &molecule,
                &params,
                &Am1Options {
                    electric_field: Some(f),
                    ..base.clone()
                },
            )
            .unwrap();
            let d = r.dipole_debye;
            [
                d.x / AU_DIPOLE_TO_DEBYE,
                d.y / AU_DIPOLE_TO_DEBYE,
                d.z / AU_DIPOLE_TO_DEBYE,
            ]
        };
        let (plus, minus) = (dipole(1.0), dipole(-1.0));
        for a in 0..3 {
            fd[a][beta] = (plus[a] - minus[a]) / (2.0 * h) * HARTREE_TO_EV;
        }
    }

    let trace_p = (0..3).map(|i| periodic.0[i][i]).sum::<f64>() / 3.0;
    let trace_f = (0..3).map(|i| fd[i][i]).sum::<f64>() / 3.0;
    eprintln!(
        "    open-shell CH3 in a 12 A box: periodic <alpha> = {trace_p:.4} Bohr^3, isolated \
         finite-field <alpha> = {trace_f:.4} Bohr^3 ({:.2} % apart)",
        100.0 * (trace_p - trace_f).abs() / trace_f.abs()
    );
    assert!(
        trace_f > 1.0,
        "the finite-field polarizability is {trace_f:.4}, too small to compare against"
    );
    // The residual is physical: the radical polarizing its own periodic images at 12 A. The
    // restricted version of this check converges to 0.17 % at the same box size.
    assert!(
        (trace_p - trace_f).abs() < 0.05 * trace_f.abs(),
        "the periodic open-shell polarizability {trace_p:.4} is not the isolated radical's \
         {trace_f:.4}"
    );
}

/// **DFPT at finite `q`, open shell.** Forcing UHF on a closed shell must give `D(q)` back.
///
/// At a `q` that is neither Gamma nor a zone boundary, so the band pairs `(k, k+q)` are genuinely
/// distinct and the phases are not all real. Half the ways to get the two-channel bookkeeping
/// wrong survive at Gamma and only appear once `k` and `k+q` carry different occupations.
#[test]
fn forcing_uhf_on_a_closed_shell_reproduces_the_restricted_dynamical_matrix() {
    use am1_rs::pbc::{force_constants_at_q_with, DfptOptions, KPoint, LongRange};
    let params = Am1Parameters::standard().unwrap();
    let molecule = hf_chain(1, 3.4);
    let q = KPoint {
        fractional: [0.3, 0.0, 0.0],
        weight: 1.0,
    };
    let dfpt = DfptOptions {
        long_range: LongRange::Require,
        ..DfptOptions::default()
    };

    let restricted = force_constants_at_q_with(&molecule, &params, &options(4), &dfpt, q)
        .unwrap()
        .force_constants;
    let unrestricted = force_constants_at_q_with(
        &molecule,
        &params,
        &PbcOptions {
            unrestricted: true,
            ..options(4)
        },
        &dfpt,
        q,
    )
    .unwrap()
    .force_constants;

    let mut diff = 0.0_f64;
    let mut scale = 0.0_f64;
    for i in 0..restricted.n {
        for j in 0..restricted.n {
            let (ar, ai) = restricted.get(i, j);
            let (br, bi) = unrestricted.get(i, j);
            diff = diff.max((ar - br).abs()).max((ai - bi).abs());
            scale = scale.max(ar.abs()).max(ai.abs());
        }
    }
    eprintln!(
        "    q = 0.3, 4 k-points: max |D(RHF) - D(forced UHF)| = {diff:.3e} eV/Bohr^2 of \
         {scale:.3e}"
    );
    assert!(scale > 1.0, "the force constants are too small to compare");
    assert!(
        diff < 1.0e-7 * scale,
        "forced UHF disagrees with RHF on D(q) by {diff:.3e} eV/Bohr^2"
    );
}

/// A **genuine** open shell: `D(q = 0)` from DFPT must equal the `q = 0` Hessian.
///
/// The two are the same number computed by different machinery -- DFPT solves band pairs across
/// `k` and `k + q` with occupation differences, the Hessian solves an occupied-virtual CPHF -- and
/// on an open shell each does it in two coupled channels. Anything mismatched between the
/// channels' bare perturbations, occupations or kernel weights separates them.
#[test]
fn the_open_shell_dfpt_reproduces_the_open_shell_hessian_at_gamma() {
    use am1_rs::pbc::{force_constants_at_q_with, DfptOptions, KPoint, LongRange};
    let params = Am1Parameters::standard().unwrap();
    let molecule = methyl_chain(1, 4.2);
    let opts = PbcOptions {
        multiplicity: 2,
        unrestricted: true,
        ..options(4)
    };

    let hessian = pbc_hessian(&molecule, &params, &opts).unwrap();
    let dfpt = force_constants_at_q_with(
        &molecule,
        &params,
        &opts,
        &DfptOptions {
            long_range: LongRange::Require,
            ..DfptOptions::default()
        },
        KPoint {
            fractional: [0.0; 3],
            weight: 1.0,
        },
    )
    .unwrap()
    .force_constants;

    let mut diff = 0.0_f64;
    let mut worst_im = 0.0_f64;
    let mut scale = 0.0_f64;
    for i in 0..hessian.rows {
        for j in 0..hessian.cols {
            let (re, im) = dfpt.get(i, j);
            diff = diff.max((re - hessian[(i, j)]).abs());
            worst_im = worst_im.max(im.abs());
            scale = scale.max(hessian[(i, j)].abs());
        }
    }
    eprintln!(
        "    open shell, q = 0: max |DFPT - pbc_hessian| = {diff:.3e} of {scale:.3e} \
         ({:.1e} relative), largest imaginary part {worst_im:.3e}",
        diff / scale
    );
    assert!(scale > 1.0, "the force constants are too small to compare");
    assert!(
        worst_im < 1.0e-8,
        "the force constants must be real at q = 0; largest imaginary part {worst_im:.3e}"
    );
    assert!(
        diff < 1.0e-5 * scale,
        "the open-shell DFPT disagrees with the open-shell Hessian by {diff:.3e}"
    );
}

/// The open-shell `D(q)` must not be the restricted one in disguise.
///
/// Without this, both assertions above would still pass if `unrestricted` were quietly ignored at
/// finite `q`.
#[test]
fn the_open_shell_dynamical_matrix_is_not_the_closed_shell_one() {
    use am1_rs::pbc::{force_constants_at_q_with, DfptOptions, KPoint, LongRange};
    let params = Am1Parameters::standard().unwrap();
    let molecule = methyl_chain(1, 4.2);
    let q = KPoint {
        fractional: [0.25, 0.0, 0.0],
        weight: 1.0,
    };
    let dfpt = DfptOptions {
        long_range: LongRange::Require,
        ..DfptOptions::default()
    };
    let run = |o: PbcOptions| {
        force_constants_at_q_with(&molecule, &params, &o, &dfpt, q)
            .unwrap()
            .force_constants
    };
    let doublet = run(PbcOptions {
        multiplicity: 2,
        unrestricted: true,
        ..options(4)
    });
    let closed = run(PbcOptions {
        charge: -1.0,
        ..options(4)
    });

    let mut diff = 0.0_f64;
    let mut scale = 0.0_f64;
    for i in 0..doublet.n {
        for j in 0..doublet.n {
            let (ar, ai) = doublet.get(i, j);
            let (br, bi) = closed.get(i, j);
            diff = diff.max((ar - br).abs()).max((ai - bi).abs());
            scale = scale.max(ar.abs()).max(ai.abs());
        }
    }
    eprintln!("    D(q) doublet vs closed-shell anion: max difference {diff:.3e} of {scale:.3e}");
    assert!(
        diff > 0.01 * scale,
        "the open-shell and closed-shell D(q) differ by only {diff:.3e}, which suggests the \
         unrestricted path is not being taken"
    );
}

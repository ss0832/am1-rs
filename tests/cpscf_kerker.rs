// SPDX-License-Identifier: GPL-3.0-or-later

//! Kerker preconditioning on the CPSCF, and whether it earns its place.
//!
//! The coupled-perturbed solve is a fixed point in the *response* density, and it fails the same
//! way the ground-state SCF does: on a dense polar cell the residual stops falling well before the
//! tolerance. Rutile GeO₂ stops at the 200-iteration cap with `8.2e-8` against `1e-10`, after
//! several minutes, and a phonon calculation that cannot solve its response has no answer at all.
//!
//! The hypothesis is that this is the same long-wavelength charge sloshing, in the derivative of
//! the density rather than the density — the response's charge transfer couples through the same
//! monopole Coulomb kernel `γ`, so `(1 + κγ)⁻¹` should damp it the same way. This file is where
//! that is *measured* rather than assumed, because the record for such hypotheses in this crate is
//! poor: two SCF fixes were added and removed in 0.2.3 after measurement showed one broke
//! formaldehyde and the other crippled a converging graphene supercell.

use am1_rs::lattice::Lattice;
use am1_rs::math::Vec3;
use am1_rs::params::Am1Parameters;
use am1_rs::pbc::dfpt::{force_constants_at_q_with, DfptOptions};
use am1_rs::pbc::kpoints::{KMesh, KPoint};
use am1_rs::pbc::PbcOptions;
use am1_rs::system::{Atom, Molecule};

const ANG: f64 = am1_rs::constants::ANGSTROM_TO_BOHR;

/// Rutile GeO₂ — the case whose CPSCF does not converge without help.
fn rutile() -> Molecule {
    let (a, c, u) = (4.3966 * ANG, 2.8624 * ANG, 0.3059);
    let cell = Lattice::from_vectors(
        Vec3::new(a, 0.0, 0.0),
        Vec3::new(0.0, a, 0.0),
        Vec3::new(0.0, 0.0, c),
        [true; 3],
    )
    .unwrap();
    let atoms = [
        (32, 0.0, 0.0, 0.0),
        (32, 0.5, 0.5, 0.5),
        (8, u, u, 0.0),
        (8, 1.0 - u, 1.0 - u, 0.0),
        (8, 0.5 + u, 0.5 - u, 0.5),
        (8, 0.5 - u, 0.5 + u, 0.5),
    ]
    .into_iter()
    .map(|(z, x, y, zc)| Atom {
        z,
        position: Vec3::new(x * a, y * a, zc * c),
    })
    .collect();
    Molecule::new(atoms).with_cell(cell)
}

/// A chain, which converges either way — the control.
fn chain() -> Molecule {
    let a = 3.0 * ANG;
    let cell = Lattice::from_vectors(
        Vec3::new(60.0 * ANG, 0.0, 0.0),
        Vec3::new(0.0, 60.0 * ANG, 0.0),
        Vec3::new(0.0, 0.0, a),
        [false, false, true],
    )
    .unwrap();
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
    .with_cell(cell)
}

fn options(mesh: [usize; 3]) -> PbcOptions {
    PbcOptions {
        kmesh: KMesh::MonkhorstPack(mesh),
        fold_time_reversal: false,
        max_scf: 400,
        ..Default::default()
    }
}

/// Preconditioning must not change the fixed point, only the path to it.
///
/// This is the property that makes the option safe to turn on: `(1 + κγ)⁻¹` multiplies the
/// *residual*, which is zero at the solution, so the solution is untouched. A preconditioner that
/// shifted the answer would still converge and still look plausible, and only a comparison against
/// the unpreconditioned solve would show it.
#[test]
fn preconditioning_does_not_move_the_answer() {
    let molecule = chain();
    let params = Am1Parameters::standard().unwrap();
    let options = options([1, 1, 4]);
    let plain = force_constants_at_q_with(
        &molecule,
        &params,
        &options,
        &DfptOptions::default(),
        KPoint::gamma(),
    )
    .unwrap();
    let kerker = force_constants_at_q_with(
        &molecule,
        &params,
        &options,
        &DfptOptions {
            cpscf_kerker_kappa: 0.2,
            ..Default::default()
        },
        KPoint::gamma(),
    )
    .unwrap();
    let n = plain.force_constants.re.rows;
    let mut worst = 0.0_f64;
    let mut scale = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            let (ar, ai) = plain.force_constants.get(i, j);
            let (br, bi) = kerker.force_constants.get(i, j);
            worst = worst.max((ar - br).abs()).max((ai - bi).abs());
            scale = scale.max(ar.abs()).max(ai.abs());
        }
    }
    eprintln!("chain: |C| ~ {scale:.3e}, preconditioned answer differs by {worst:.3e}");
    // The two solves stop at the same tolerance from the same fixed point, so they agree to about
    // that tolerance rather than to roundoff.
    assert!(
        worst < 1.0e-7 * scale.max(1.0),
        "preconditioning moved the answer by {worst:e} on a matrix of scale {scale:e}"
    );
}

/// The case it exists for: does it actually converge rutile GeO₂'s response?
///
/// Ignored because it is minutes long either way. It is the measurement that decides whether the
/// option is worth having, so it is written down rather than left in a scratch file.
#[test]
#[ignore]
fn it_converges_the_response_that_would_not() {
    let molecule = rutile();
    let params = Am1Parameters::standard().unwrap();
    let options = options([2, 2, 3]);
    for kappa in [0.0, 0.05, 0.2, 0.8] {
        let dfpt = DfptOptions {
            cpscf_kerker_kappa: kappa,
            ..Default::default()
        };
        let start = std::time::Instant::now();
        let outcome =
            force_constants_at_q_with(&molecule, &params, &options, &dfpt, KPoint::gamma());
        let wall = start.elapsed().as_secs_f64();
        match outcome {
            Ok(_) => eprintln!("kappa {kappa:>4}: converged in {wall:.0} s"),
            Err(e) => eprintln!("kappa {kappa:>4}: {} after {wall:.0} s", e),
        }
    }
}

/// Is the residual floor the response's, or the ground state's?
///
/// The CPSCF asked for `cpscf_tol = 1e-10` through 0.2.2. The ground state it is built on stops at
/// `PbcOptions::p_tol = 1e-7` — **three orders looser**. A response cannot be more converged than
/// the density it is the derivative of, so the floor may be nothing to do with the response solver
/// at all; it may be the ground state's stopping point, measured through it.
///
/// Kerker was the first hypothesis and the measurement did not support it: on rutile GeO2 a sweep
/// of `kappa` over 0, 0.05, 0.2 and 0.8 moved the worst residual between 1.8e-7 and 5.0e-7 and
/// converged none of them. Charge sloshing in the response is not what is stopping this.
///
/// This tightens the ground state instead and reports what the response then reaches. Ignored
/// because each arm is minutes long.
#[test]
#[ignore]
fn whose_floor_is_the_cpscf_residual() {
    let molecule = rutile();
    let params = Am1Parameters::standard().unwrap();
    for (e_tol, p_tol) in [(1.0e-8, 1.0e-7), (1.0e-10, 1.0e-9), (1.0e-12, 1.0e-11)] {
        let options = PbcOptions {
            e_tol,
            p_tol,
            ..options([2, 2, 3])
        };
        let start = std::time::Instant::now();
        let outcome = force_constants_at_q_with(
            &molecule,
            &params,
            &options,
            &DfptOptions::default(),
            KPoint::gamma(),
        );
        let wall = start.elapsed().as_secs_f64();
        match outcome {
            Ok(_) => eprintln!("ground state p_tol {p_tol:.0e}: response converged in {wall:.0} s"),
            Err(e) => eprintln!("ground state p_tol {p_tol:.0e}: {} [{wall:.0} s]", e),
        }
    }
}

/// Every layer agrees on the response tolerance.
///
/// `DfptOptions::default` is the Rust library's, `native.dfpt`'s signature is the Python
/// surface's, `python/am1_rs/native.py` is the wrapper in front of it, and `AM1.get_phonons` is
/// the ASE calculator's. Four hand-written copies of one number. The divide-and-conquer options
/// had the same shape and had drifted into three different partitions across five layers before
/// `tests/dc_optimize.rs` pinned them; this is the same guard for the same reason.
#[test]
fn every_layer_agrees_on_the_response_tolerance() {
    let want = DfptOptions::default().cpscf_tol;
    // `1e-8`, not `1e-10`: see the note on `DfptOptions::cpscf_tol`. Spinel ZnAl2O4's response
    // reaches 1.025e-9 and was refused by the old default.
    assert!(
        (want - 1.0e-8).abs() < 1.0e-20,
        "the library default moved to {want:e}"
    );
    for (file, text, needle) in [
        (
            "src/python.rs",
            include_str!("../src/python.rs"),
            "cpscf_tol=1.0e-8",
        ),
        (
            "python/am1_rs/native.py",
            include_str!("../python/am1_rs/native.py"),
            "cpscf_tol: float = 1.0e-8",
        ),
        (
            "python/am1_rs/ase.py",
            include_str!("../python/am1_rs/ase.py"),
            "cpscf_tol=1.0e-8",
        ),
    ] {
        assert!(
            text.contains(needle),
            "{file} does not carry `{needle}`; it and DfptOptions::default() have drifted apart"
        );
    }
}

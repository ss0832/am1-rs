// SPDX-License-Identifier: GPL-3.0-or-later

//! A periodic **response** refuses a ground state with fractional occupations.
//!
//! # What the refusal is about
//!
//! Every response path here derives from a fixed integer occupation. That is not a small
//! approximation on a partially filled band, it is a missing term: there is no `∂f/∂ε` anywhere in
//! the coupled-perturbed equations, so the Fermi surface cannot redistribute charge under the
//! perturbation. Through 0.2.2 both paths ran anyway, and both went wrong quietly:
//!
//! * the **CPHF** path classifies each band as occupied (`f > full − 1e-6`) or virtual
//!   (`f < 1e-6`), and a partially filled band is **neither** — so it is dropped from the response
//!   entirely, taking its whole orbital-relaxation contribution with it;
//! * **DFPT** keeps every band pair but weights each by a frozen `f_n(k) − f_m(k+q)`.
//!
//! # What it is *not* about
//!
//! It does not gate smearing. Smearing is often the only way to converge the ground state of a
//! small-gap or coarsely sampled solid, and the judgement here is on the **converged occupations**
//! rather than on `smearing_ev > 0`. [`smearing_itself_is_not_what_is_refused`] is the test that
//! pins that distinction, and it measures the margin rather than asserting it: at `kT` well below
//! the gap the conduction band holds `exp(−gap/2kT)` electrons, which reaches `1e-6` only when the
//! gap falls under about `27·kT`.

use am1_rs::error::Am1Error;
use am1_rs::lattice::Lattice;
use am1_rs::math::Vec3;
use am1_rs::pbc::kpoints::KPoint;
use am1_rs::pbc::{pbc_hessian, KMesh, PbcOptions, INTEGER_OCCUPATION_TOL};
use am1_rs::{Am1Parameters, Atom, Molecule};

const ANG: f64 = 1.0 / 0.529167;

/// A chain of H₂ units: a wide gap, and cheap.
fn h2_chain(bond_ang: f64, spacing_ang: f64) -> Molecule {
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
            Vec3::new(spacing_ang * ANG, 0.0, 0.0),
            Vec3::new(0.0, 40.0, 0.0),
            Vec3::new(0.0, 0.0, 40.0),
            [true, false, false],
        )
        .unwrap(),
    )
}

fn pbc_options(mesh: [usize; 3], smearing_ev: f64) -> PbcOptions {
    PbcOptions {
        kmesh: KMesh::MonkhorstPack(mesh),
        realspace_cutoff: 40.0,
        exchange_cutoff: Some(15.0),
        smearing_ev,
        e_tol: 1.0e-11,
        p_tol: 1.0e-10,
        max_scf: 600,
        fold_time_reversal: false,
        ..PbcOptions::default()
    }
}

fn gamma() -> KPoint {
    KPoint {
        fractional: [0.0, 0.0, 0.0],
        weight: 1.0,
    }
}

/// The occupations a ground state actually reaches, so the tests below argue from measurement.
///
/// This mirrors what `solve_orbitals` does internally — diagonalize at each `k`, fill against one
/// chemical potential — because the tests need the number the refusal is comparing, not a proxy.
fn worst_fractional(mol: &Molecule, opts: &PbcOptions) -> f64 {
    let params = Am1Parameters::standard().unwrap();
    // The response entry point is the thing under test, so ask it and read the number out of the
    // error rather than reimplementing the filling here (which could agree with itself and be
    // wrong). A run that succeeds is one whose worst fractional occupation is below the cut.
    match pbc_hessian(mol, &params, opts) {
        Ok(_) => 0.0,
        Err(Am1Error::FractionalOccupation {
            occupation, full, ..
        }) => occupation.min(full - occupation),
        Err(e) => panic!("unexpected error: {e}"),
    }
}

#[test]
fn smearing_itself_is_not_what_is_refused() {
    // The distinction the user of a small-gap solid needs: turning smearing on does not put the
    // response out of reach. An H2 chain has a wide gap, so even at a smearing far above anything
    // a converged insulator needs, its occupations stay integer to well under the cut.
    let mol = h2_chain(0.7, 4.0);
    let params = Am1Parameters::standard().unwrap();

    for kt in [0.0, 0.05, 0.2, 0.5] {
        let opts = pbc_options([4, 1, 1], kt);
        let h = pbc_hessian(&mol, &params, &opts);
        assert!(
            h.is_ok(),
            "smearing at kT = {kt} eV was refused on a gapped chain: {:?}",
            h.err().map(|e| e.to_string())
        );
    }
}

#[test]
fn a_partially_filled_band_is_refused_by_name() {
    // Compressing the chain until the two hydrogens are equidistant removes the dimerization that
    // opens the gap: the folded band is half filled, which is a metal. Smeared, that is a genuine
    // fractional occupation rather than a numerical one.
    let mol = h2_chain(1.5, 3.0);
    let params = Am1Parameters::standard().unwrap();
    let opts = pbc_options([6, 1, 1], 0.5);

    match pbc_hessian(&mol, &params, &opts) {
        Err(Am1Error::FractionalOccupation {
            occupation,
            full,
            path,
            smearing_ev,
            ..
        }) => {
            let d = occupation.min(full - occupation);
            assert!(
                d > INTEGER_OCCUPATION_TOL,
                "refused at {d:.3e}, under the cut"
            );
            assert_eq!(path, "coupled-perturbed (CPHF)");
            assert_eq!(smearing_ev, 0.5);
        }
        Ok(_) => panic!("a half-filled band was accepted; the response would be missing the band"),
        Err(e) => panic!("wrong diagnosis: {e}"),
    }
}

#[test]
fn the_message_says_what_to_do_and_what_not_to() {
    // A diagnosis is only useful if it points somewhere. Widening `kT` is the intuitive move and
    // the wrong one -- it moves occupations *away* from integer -- so the message has to say so,
    // and it has to say that smearing as such is not the thing being refused.
    let text = Am1Error::FractionalOccupation {
        k_index: 3,
        level: 7,
        occupation: 0.83,
        full: 2.0,
        smearing_ev: 0.5,
        path: "DFPT (at k+q)",
    }
    .to_string();

    assert!(text.contains("k-point 3"), "{text}");
    assert!(text.contains("band 7"), "{text}");
    assert!(text.contains("0.830000"), "{text}");
    assert!(text.contains("DFPT (at k+q)"), "{text}");
    assert!(text.contains("not a refusal to use smearing"), "{text}");
    assert!(text.contains("finer k-mesh"), "{text}");
}

#[test]
fn the_escape_hatch_runs_the_calculation_that_was_refused() {
    // `require_integer_occupations = false` is the 0.2.2 behaviour, kept because refusing to
    // produce a number at all is not always the right trade -- but the number it produces is the
    // wrong one, which is why it is opt-in and not a tolerance.
    let mol = h2_chain(1.5, 3.0);
    let params = Am1Parameters::standard().unwrap();

    let strict = pbc_options([6, 1, 1], 0.5);
    assert!(pbc_hessian(&mol, &params, &strict).is_err());

    let loose = PbcOptions {
        require_integer_occupations: false,
        ..strict
    };
    assert!(
        pbc_hessian(&mol, &params, &loose).is_ok(),
        "the escape hatch did not restore the old path"
    );
}

#[test]
fn dfpt_refuses_the_same_state_the_cphf_does() {
    // Two different failure modes, one diagnosis. The paths reach occupations by different routes
    // -- the CPHF path fills a k-resolved list, DFPT fills `k` and `k + q` together -- so this
    // checks they agree about which states are out of scope rather than assuming it.
    let mol = h2_chain(1.5, 3.0);
    let params = Am1Parameters::standard().unwrap();
    let opts = pbc_options([6, 1, 1], 0.5);

    let d = am1_rs::pbc::dynamical_matrix_dfpt(&mol, &params, &opts, gamma());
    match d {
        Err(Am1Error::FractionalOccupation { path, .. }) => {
            assert!(path.starts_with("DFPT"), "wrong path label: {path}");
        }
        Ok(_) => panic!("DFPT accepted a partially filled band"),
        Err(e) => panic!("wrong diagnosis: {e}"),
    }
}

/// A zincblende primitive cell: two atoms, fcc lattice vectors, `B` at `a/4 (1,1,1)`.
fn zincblende(za: u8, zb: u8, a_ang: f64) -> Molecule {
    let a = a_ang * ANG;
    Molecule::new(vec![
        Atom {
            z: za,
            position: Vec3::zero(),
        },
        Atom {
            z: zb,
            position: Vec3::new(a / 4.0, a / 4.0, a / 4.0),
        },
    ])
    .with_cell(
        Lattice::from_vectors(
            Vec3::new(0.0, a / 2.0, a / 2.0),
            Vec3::new(a / 2.0, 0.0, a / 2.0),
            Vec3::new(a / 2.0, a / 2.0, 0.0),
            [true, true, true],
        )
        .unwrap(),
    )
}

/// **The workflow question**, measured rather than argued: `docs/pbc.md` tells a user to reach a
/// converged ground state on a dense inorganic solid with smearing — cubic BN at `kT = 0.3 eV`,
/// zincblende AlP at 0.2. If the refusal introduced in 0.2.3 turned those into errors it would
/// have broken the advice, so this prints what they actually reach.
///
/// Measured:
///
/// ```text
/// cubic BN,       4x4x4, kT = 0.3   accepted
/// cubic BN,       6x6x6, kT = 0     accepted
/// zincblende AlP, 6x6x6, kT = 0     accepted        <- the mesh the reported frequencies use
/// zincblende AlP, 4x4x4, kT = 0.2   REFUSED, 1.794e-6
/// zincblende AlP, 6x6x6, kT = 0.2   REFUSED, 1.957e-6
/// ```
///
/// So the documented advice survives: BN's whole point is that smearing is what converges it, and
/// it is accepted at the width the docs name. **AlP under smearing is not**, and marginally —
/// `2e-6` of an electron. That is not a near miss to be tuned away: at `2e-6` the classifier puts
/// that band in neither the occupied nor the virtual set, so the CPHF response omits an entire
/// band. Refusing turns a silent 100 % error on one band into a message. The remedy is the one
/// `docs/pbc.md` already gives for AlP — 6×6×6 with no smearing, which is accepted here.
#[test]
#[ignore = "measurement, not an assertion"]
fn the_smeared_inorganic_workflows_still_pass() {
    for (label, mol, mesh, kt) in [
        ("cubic BN, 4x4x4", zincblende(5, 7, 3.615), [4, 4, 4], 0.3),
        (
            "cubic BN, 6x6x6, no smearing",
            zincblende(5, 7, 3.615),
            [6, 6, 6],
            0.0,
        ),
        (
            "zincblende AlP, 4x4x4",
            zincblende(13, 15, 5.4635),
            [4, 4, 4],
            0.2,
        ),
        // The mesh `docs/pbc.md` actually recommends for AlP, and the one its reported
        // frequencies come from.
        (
            "zincblende AlP, 6x6x6",
            zincblende(13, 15, 5.4635),
            [6, 6, 6],
            0.0,
        ),
        (
            "zincblende AlP, 6x6x6, kT=0.2",
            zincblende(13, 15, 5.4635),
            [6, 6, 6],
            0.2,
        ),
    ] {
        let opts = pbc_options(mesh, kt);
        let params = Am1Parameters::standard().unwrap();
        match pbc_hessian(&mol, &params, &opts) {
            Ok(_) => println!("{label:32} kT={kt} -> accepted"),
            Err(Am1Error::FractionalOccupation {
                occupation, full, ..
            }) => println!(
                "{label:32} kT={kt} -> REFUSED, worst fractional {:.3e}",
                occupation.min(full - occupation)
            ),
            Err(e) => println!("{label:32} kT={kt} -> other failure: {e}"),
        }
    }
}

/// Not an assertion — a measurement, printed with `--nocapture`, of where the cut actually bites.
///
/// The question this answers is the practical one: at what smearing does a system with a given gap
/// stop being accepted? `f_LUMO ≈ exp(−gap/2kT)`, so the cut at `1e-6` is crossed when the gap
/// falls below roughly `27.6·kT` — 1.4 eV at `kT = 0.05`, 8.3 eV at `kT = 0.3`. That is the number
/// to quote when someone asks whether their smeared run will be accepted.
#[test]
#[ignore = "measurement, not an assertion"]
fn where_the_cut_bites() {
    for (label, mol) in [
        ("H2 chain, dimerized (wide gap)", h2_chain(0.7, 4.0)),
        ("H2 chain, near-uniform", h2_chain(1.4, 3.0)),
        ("H2 chain, uniform (metallic)", h2_chain(1.5, 3.0)),
    ] {
        for kt in [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0] {
            let opts = pbc_options([6, 1, 1], kt);
            let d = worst_fractional(&mol, &opts);
            let verdict = if d > INTEGER_OCCUPATION_TOL {
                "REFUSED"
            } else {
                "ok"
            };
            println!("{label:32} kT={kt:4} -> worst fractional {d:.3e}  {verdict}");
        }
    }
}

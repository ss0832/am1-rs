// SPDX-License-Identifier: GPL-3.0-or-later

//! RM1 validation against MOPAC.
//!
//! RM1 (Rocha, Freire, Simas & Stewart, *J. Comput. Chem.* **27**, 1101 (2006)) is a
//! reparameterization of AM1 with an identical functional form, so it exercises the whole of
//! this crate's existing machinery with a different parameter table. That makes an external
//! reference essential: if the table were mis-parsed — a column swapped, a Gaussian dropped —
//! the calculation would still run and still look plausible.
//!
//! The reference is MOPAC's own regression case `tests/keywords/RM1.mop`, carbon dioxide,
//! whose shipped output gives three independently checkable numbers: the heat of formation at
//! the input geometry, the heat of formation and bond length after optimization, and the
//! Koopmans ionization potential.

use am1_rs::method::NddoMethod;
use am1_rs::{optimize, run_am1, Am1Options, Am1Parameters, Molecule, OptOptions};

/// MOPAC 22 `tests/keywords/RM1.out`, input geometry C-O = 1.16 A (first optimization cycle).
const MOPAC_HOF_AT_INPUT: f64 = -81.09054;
/// Same file, after eigenvector-following optimization.
const MOPAC_HOF_OPTIMIZED: f64 = -81.55398;
const MOPAC_CO_BOND_ANGSTROM: f64 = 1.173109199;
/// Also from the final (optimized) result, not the input geometry.
const MOPAC_IONIZATION_EV: f64 = 12.913991;

/// Heat-of-formation agreement to expect against **MOPAC 22**.
///
/// Not zero, and not because of RM1. MOPAC carries two sets of physical constants
/// (`src/.../conref_C`): the modern CODATA values, and the historical MOPAC7 ones
/// — `a0 = 0.529167`, `1 au = 27.21 eV`, `1 eV = 23.061 kcal/mol`. This crate deliberately
/// uses the historical set, because the AM1 parameters and the derived rho terms were fitted
/// against it (see `src/constants.rs`); modern MOPAC defaults to CODATA.
///
/// The resulting offset is a property of the crate, not of any one parameter table, which is
/// exactly what the measurements show — on this same CO2 case, against MOPAC's own reference
/// outputs:
///
/// ```text
///     AM1 optimized   -79.828954  vs MOPAC -79.86140   delta +0.0325 kcal/mol
///     RM1 optimized   -81.519150  vs MOPAC -81.55398   delta +0.0348 kcal/mol
/// ```
///
/// Same size, same sign, two different parameter sets. So this tolerance is a bound on a
/// known systematic, not slack hiding a transcription error — a mis-parsed column would move
/// the answer by whole kcal/mol, not by three hundredths.
const HOF_TOLERANCE_KCAL: f64 = 0.06;

fn co2(bond: f64) -> Molecule {
    Molecule::from_xyz_str(
        &format!("3\nCO2\nC 0.0 0.0 0.0\nO {bond} 0.0 0.0\nO -{bond} 0.0 0.0\n"),
        0.0,
    )
    .unwrap()
}

#[test]
fn rm1_parameter_table_loads() {
    let p = Am1Parameters::for_method(NddoMethod::Rm1).unwrap();
    let symbols = p.supported_symbols();
    assert_eq!(
        symbols,
        vec!["H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"],
        "RM1's published main-group set is exactly these ten elements"
    );
    // Gaussian counts per element, straight from the RM1 paper.
    for (z, want) in [
        (1u8, 3usize),
        (6, 4),
        (7, 3),
        (8, 2),
        (9, 2),
        (15, 3),
        (16, 3),
        (17, 2),
        (35, 2),
        (53, 2),
    ] {
        let n = p.element(z).unwrap().gauss.len();
        assert_eq!(
            n, want,
            "Z={z} should carry {want} RM1 Gaussians, found {n}"
        );
    }
    assert_eq!(p.method, NddoMethod::Rm1);
}

#[test]
fn rm1_is_not_am1() {
    // A guard against the parameter table silently falling back to AM1.
    let am1 = Am1Parameters::standard().unwrap();
    let rm1 = Am1Parameters::for_method(NddoMethod::Rm1).unwrap();
    let (a, r) = (am1.element(6).unwrap(), rm1.element(6).unwrap());
    // A reparameterization, not a redesign: carbon's U_ss moves by only ~0.30 eV
    // (-52.0287 -> -51.7256) and beta_s by ~0.26 eV. So this checks that the tables are
    // distinct, and leaves "distinct enough to matter" to the energy comparison below.
    for (name, x, y) in [
        ("U_ss", a.u_ss, r.u_ss),
        ("U_pp", a.u_pp, r.u_pp),
        ("beta_s", a.beta_s, r.beta_s),
        ("zeta_s", a.zeta_s, r.zeta_s),
        ("alpha", a.alpha, r.alpha),
    ] {
        assert!(
            (x - y).abs() > 1.0e-6,
            "AM1 and RM1 carbon {name} are identical ({x}); the RM1 table did not load"
        );
    }
    let e_am1 = run_am1(&co2(1.16), &am1, &Am1Options::default()).unwrap();
    let e_rm1 = run_am1(&co2(1.16), &rm1, &Am1Options::default()).unwrap();
    assert!(
        (e_am1.heat_of_formation_kcal - e_rm1.heat_of_formation_kcal).abs() > 1.0,
        "AM1 and RM1 CO2 heats of formation should differ"
    );
}

#[test]
fn rm1_co2_single_point_matches_mopac() {
    let params = Am1Parameters::for_method(NddoMethod::Rm1).unwrap();
    let r = run_am1(&co2(1.16), &params, &Am1Options::default()).unwrap();
    assert!(r.converged);
    let d = r.heat_of_formation_kcal - MOPAC_HOF_AT_INPUT;
    eprintln!(
        "    RM1 CO2 at C-O = 1.16 A: dHf = {:.5} kcal/mol (MOPAC {MOPAC_HOF_AT_INPUT:.5}, delta {d:+.5})",
        r.heat_of_formation_kcal
    );
    assert!(
        d.abs() < HOF_TOLERANCE_KCAL,
        "RM1 heat of formation off by {d:.5} kcal/mol"
    );
}

#[test]
fn rm1_co2_optimizes_to_the_mopac_minimum() {
    let params = Am1Parameters::for_method(NddoMethod::Rm1).unwrap();
    let opt = optimize(
        &co2(1.16),
        &params,
        &Am1Options::default(),
        &OptOptions::default(),
    )
    .unwrap();
    assert!(opt.converged, "RM1 CO2 optimization did not converge");

    let d = opt.scf.heat_of_formation_kcal - MOPAC_HOF_OPTIMIZED;
    eprintln!(
        "    RM1 CO2 optimized: dHf = {:.5} kcal/mol (MOPAC {MOPAC_HOF_OPTIMIZED:.5}, delta {d:+.5})",
        opt.scf.heat_of_formation_kcal
    );
    assert!(
        d.abs() < HOF_TOLERANCE_KCAL,
        "optimized RM1 heat of formation off by {d:.5} kcal/mol"
    );

    let bohr_to_ang = 0.529167_f64;
    let bond =
        (opt.molecule.atoms[1].position - opt.molecule.atoms[0].position).norm() * bohr_to_ang;
    let db = bond - MOPAC_CO_BOND_ANGSTROM;
    eprintln!("    RM1 CO2 optimized C-O = {bond:.6} A (MOPAC {MOPAC_CO_BOND_ANGSTROM:.6}, delta {db:+.6})");
    assert!(db.abs() < 0.002, "optimized RM1 C-O bond off by {db:.6} A");

    // MOPAC reports the ionization potential for its *final* structure, so this has to be
    // compared at the optimized geometry -- not, as it was first written here, at the input
    // geometry, which made a geometry difference look like a parameter error.
    let ip = -opt.scf.homo_ev.unwrap();
    let dip = ip - MOPAC_IONIZATION_EV;
    eprintln!("    RM1 CO2 optimized ionization potential = {ip:.6} eV (MOPAC {MOPAC_IONIZATION_EV:.6}, delta {dip:+.6})");
    assert!(
        dip.abs() < 0.01,
        "RM1 ionization potential off by {dip:.6} eV"
    );
}

#[test]
fn rm1_refuses_elements_it_does_not_parameterize_and_says_so() {
    // Silicon is fine under AM1 but is not in RM1's published set. The error should name the
    // method and what it does cover, rather than claiming the element is unknown.
    let params = Am1Parameters::for_method(NddoMethod::Rm1).unwrap();
    let silane = Molecule::from_xyz_str(
        "5\nSiH4\nSi 0.0 0.0 0.0\nH 0.8544 0.8544 0.8544\nH -0.8544 -0.8544 0.8544\nH -0.8544 0.8544 -0.8544\nH 0.8544 -0.8544 -0.8544\n",
        0.0,
    )
    .unwrap();
    let err = run_am1(&silane, &params, &Am1Options::default()).unwrap_err();
    let msg = err.to_string();
    eprintln!("    {msg}");
    assert!(msg.contains("RM1"), "error should name the method: {msg}");
    assert!(
        msg.contains("14"),
        "error should name the atomic number: {msg}"
    );
    assert!(
        msg.contains('H') && msg.contains("Br"),
        "error should list coverage: {msg}"
    );
}

#[test]
fn rm1_gradient_is_analytic_and_correct() {
    // The RM1 parameters flow through the same closed-form gradient as AM1; verify against a
    // full-SCF finite difference so a mis-parsed Gaussian would show up here too.
    let params = Am1Parameters::for_method(NddoMethod::Rm1).unwrap();
    let mol = Molecule::from_xyz_str(
        "4\nformaldehyde-ish\nC 0.0 0.0 0.0\nO 1.21 0.0 0.0\nH -0.58 0.94 0.0\nH -0.58 -0.94 0.0\n",
        0.0,
    )
    .unwrap();
    let opts = Am1Options::default();
    let ana = am1_rs::closed_form_gradient(&mol, &params, &opts).unwrap();
    let num = am1_rs::numerical_gradient(&mol, &params, &opts, 1.0e-4).unwrap();
    let mut worst = 0.0_f64;
    for (a, n) in ana.gradient.iter().zip(num.gradient.iter()) {
        for k in 0..3 {
            worst = worst.max((a.get(k) - n.get(k)).abs());
        }
    }
    eprintln!("    RM1 gradient vs finite difference: {worst:.3e} eV/Bohr");
    assert!(worst < 5.0e-5, "RM1 gradient off by {worst:.3e}");
}

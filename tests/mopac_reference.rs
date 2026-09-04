// SPDX-License-Identifier: GPL-3.0-or-later

//! Is this actually AM1? Checked against MOPAC's own regression outputs.
//!
//! The crate's existing validation was one number — the heat of formation of water, to within
//! 0.5 kcal/mol. That is loose enough to pass with a real defect in place, and it exercises
//! three elements. These tests compare against MOPAC 22's shipped reference outputs
//! (`tests/keywords/AM1.out`, `RM1.out`), which give several independent quantities per case:
//! the heat of formation at a stated geometry, the optimized heat of formation and bond
//! length, the Koopmans ionization potential, and the Mulliken charges.
//!
//! ## The residual disagreement, and why it is not a bug
//!
//! Agreement is ~0.03 kcal/mol on a −80 kcal/mol quantity, not zero. That is a constants
//! choice, not a model error. MOPAC carries two sets of physical constants side by side
//! (`src/.../conref_C.F90`):
//!
//! ```text
//!                        historical (MOPAC7)     modern (CODATA)
//!     a0 / Angstrom      0.529167                0.529177210903
//!     1 au / eV          27.21                   27.211386245988
//!     1 eV / kcal/mol    23.061                  23.060547830619029
//! ```
//!
//! This crate uses the historical set deliberately, because the AM1 parameters and the
//! derived `rho` terms were fitted against it (see `src/constants.rs`). Modern MOPAC defaults
//! to CODATA. The offset that produces is a property of the crate rather than of any one
//! parameter table, which is what the two cases below demonstrate: AM1 and RM1 — different
//! parameters, same code — disagree with MOPAC by the same amount in the same direction.
//!
//! ## What this covers, and what it does not
//!
//! Every number here is read off MOPAC's own shipped reference outputs
//! (`tests/keywords/AM1.out`, `RM1.out` in the openmopac/mopac repository), not recomputed and
//! not recalled. The comparison spans six independent observables per method:
//!
//! | observable | what it catches that the others do not |
//! |---|---|
//! | `ΔHf` at a fixed geometry | the energy expression, with the geometry held out of it |
//! | `ΔHf` optimized | the gradient, through where it stops |
//! | optimized C–O length | the gradient again, independently of the energy scale |
//! | **all twelve orbital energies** | every level of the converged Fock matrix at once |
//! | Koopmans IP | the frontier level specifically (a subset of the above, kept for its own bound) |
//! | Mulliken charges | whether the *density* is right, not just the energy |
//!
//! The orbital spectrum is the sharpest of them. `ΔHf` is a single number that a compensating
//! pair of errors survives, and the IP is a single eigenvalue; the full spectrum constrains the
//! occupied *and* virtual levels simultaneously, and it carries CO₂'s two degenerate pairs
//! (`π_u` at −18.34, `π_g` at −13.21), which a broken two-centre rotation would split while
//! leaving `ΔHf` and the HOMO almost unmoved. Measured worst case across all twelve: **0.0022 eV
//! for AM1, 0.0034 eV for RM1**, both at the deepest level, which is where the constants-set
//! offset is largest in absolute terms.
//!
//! # Why one molecule
//!
//! Because that is what MOPAC's public reference set contains for these two methods, and this
//! was checked rather than assumed: `tests/keywords` holds 61 cases, of which `AM1.mop` and
//! `RM1.mop` are the only ones selecting these methods, and both are CO₂ at C–O = 1.16 Å. The
//! rest use MOPAC's default (PM7) or explicitly select MNDO/PM3/PM6/PM7 — `CHARGE.mop` is a bare
//! carbon atom, `DOUBLET.mop` a cobalt atom, and so on. The `tests/` directory above it is
//! PM6/PM7 protein work. Widening the oracle therefore means *running* MOPAC, not reading it.
//!
//! Frequencies and infrared intensities are not compared for the same reason: the shipped AM1
//! case does not run `FORCE`, so there is nothing to compare against. Those are validated
//! instead by finite differences and sum rules in `tests/ir.rs`, which need no external oracle.

use am1_rs::method::NddoMethod;
use am1_rs::{optimize, run_am1, Am1Options, Am1Parameters, Molecule, OptOptions};

const BOHR_TO_ANGSTROM: f64 = 0.529167;

/// Bound on the constants-set offset described above. A mis-parsed parameter column moves a
/// heat of formation by whole kcal/mol, so this separates the two cleanly.
const HOF_TOLERANCE_KCAL: f64 = 0.06;

/// Per-observable tolerances against MOPAC.
///
/// The `0.06 kcal/mol` above is an argument about a **constants offset in ΔHf** — the crate uses
/// MOPAC7's `ev = 27.21` and `a0 = 0.529167` where modern MOPAC defaults to CODATA — and that
/// argument does not transfer to a charge or a dipole. Each observable gets its own bound, set
/// from what the comparison actually measures rather than from one number reused.
mod tolerance {
    /// Mulliken charge, e. A mis-assembled density moves these by hundredths.
    pub const CHARGE: f64 = 2.0e-3;
    /// Dipole magnitude, Debye.
    pub const DIPOLE: f64 = 1.0e-2;
    /// Koopmans ionization potential, eV.
    pub const IONIZATION: f64 = 1.0e-2;
    /// Any single molecular-orbital energy, eV. Looser than `IONIZATION` because the deep
    /// core-like levels near −40 eV carry the constants-set offset amplified by their own
    /// magnitude: the same relative discrepancy that is 0.0004 eV at the HOMO is 0.001 eV there.
    /// Set just above the measured worst case, so a real parameter error still fails it.
    pub const ORBITAL: f64 = 2.0e-2;
    /// Optimized bond length, Ångström.
    pub const BOND: f64 = 2.0e-3;
}

fn co2(bond_angstrom: f64) -> Molecule {
    Molecule::from_xyz_str(
        &format!("3\nCO2\nC 0.0 0.0 0.0\nO {bond_angstrom} 0.0 0.0\nO -{bond_angstrom} 0.0 0.0\n"),
        0.0,
    )
    .unwrap()
}

struct Reference {
    method: NddoMethod,
    /// Heat of formation at the input geometry, C–O = 1.16 Å (MOPAC optimization cycle 1).
    hof_at_input: f64,
    /// Heat of formation, optimized.
    hof_optimized: f64,
    /// C–O bond length after optimization, Ångström.
    bond: f64,
    /// Koopmans ionization potential at the optimized geometry, eV.
    ionization: f64,
    /// Net atomic charge on the carbon at the optimized geometry, e. The oxygens carry half of
    /// this each, by symmetry and by charge conservation.
    ///
    /// A much sharper probe than the heat of formation: `ΔHf` is a single number that a
    /// compensating pair of errors can survive, while the charges say whether the *density* is
    /// right. They are also the quantity AM1-BCC is built on.
    charge_carbon: f64,
    /// **The whole molecular-orbital spectrum** at the optimized geometry, eV, ascending — all
    /// twelve of them, occupied and virtual, exactly as MOPAC prints them.
    ///
    /// The sharpest single comparison in this file. The heat of formation is one number and the
    /// ionization potential is one eigenvalue; this is the entire eigenspectrum of the converged
    /// Fock matrix, so it constrains every occupied *and* virtual level at once. It also carries
    /// the degeneracies — CO₂'s `π_u` pair at −18.34 and `π_g` pair at −13.21 — which a broken
    /// two-centre rotation would split while leaving `ΔHf` and the HOMO almost untouched.
    orbital_energies: &'static [f64],
}

/// MOPAC 22 `tests/keywords/AM1.out`.
const AM1_CO2: Reference = Reference {
    method: NddoMethod::Am1,
    hof_at_input: -77.16706,
    hof_optimized: -79.86140,
    bond: 1.189308342,
    ionization: 13.214572,
    charge_carbon: 0.411447,
    orbital_energies: &[
        -41.22846, -39.93911, -22.73923, -18.33834, -18.33834, -18.09906, -13.21457, -13.21457,
        0.85334, 0.85334, 2.09586, 6.55189,
    ],
};

/// MOPAC 22 `tests/keywords/RM1.out`.
const RM1_CO2: Reference = Reference {
    method: NddoMethod::Rm1,
    hof_at_input: -81.09054,
    hof_optimized: -81.55398,
    bond: 1.173109199,
    ionization: 12.913991,
    charge_carbon: 0.441757,
    orbital_energies: &[
        -39.50961, -38.55784, -21.69304, -18.01887, -18.01887, -17.70249, -12.91399, -12.91399,
        0.99037, 0.99037, 2.49737, 6.77490,
    ],
};

/// Returns the heat-of-formation offset from MOPAC, so the caller can compare methods.
fn check(reference: &Reference) -> (f64, f64) {
    let params = Am1Parameters::for_method(reference.method).unwrap();
    let name = reference.method.display_name();

    // Single point at the input geometry.
    let sp = run_am1(&co2(1.16), &params, &Am1Options::default()).unwrap();
    assert!(sp.converged, "{name} CO2 single point did not converge");
    let d_input = sp.heat_of_formation_kcal - reference.hof_at_input;
    eprintln!(
        "    {name} CO2 @ 1.16 A   dHf {:>11.5}  MOPAC {:>11.5}  delta {:+.5} kcal/mol",
        sp.heat_of_formation_kcal, reference.hof_at_input, d_input
    );
    assert!(
        d_input.abs() < HOF_TOLERANCE_KCAL,
        "{name} heat of formation at the input geometry off by {d_input:.5} kcal/mol"
    );

    // Optimized.
    let opt = optimize(
        &co2(1.16),
        &params,
        &Am1Options::default(),
        &OptOptions::default(),
    )
    .unwrap();
    assert!(opt.converged, "{name} CO2 optimization did not converge");

    let d_opt = opt.scf.heat_of_formation_kcal - reference.hof_optimized;
    eprintln!(
        "    {name} CO2 optimized  dHf {:>11.5}  MOPAC {:>11.5}  delta {:+.5} kcal/mol",
        opt.scf.heat_of_formation_kcal, reference.hof_optimized, d_opt
    );
    assert!(
        d_opt.abs() < HOF_TOLERANCE_KCAL,
        "{name} optimized heat of formation off by {d_opt:.5} kcal/mol"
    );

    let bond =
        (opt.molecule.atoms[1].position - opt.molecule.atoms[0].position).norm() * BOHR_TO_ANGSTROM;
    let d_bond = bond - reference.bond;
    eprintln!(
        "    {name} CO2 optimized  C-O {bond:>11.6}  MOPAC {:>11.6}  delta {d_bond:+.6} A",
        reference.bond
    );
    assert!(
        d_bond.abs() < tolerance::BOND,
        "{name} optimized C-O bond off by {d_bond:.6} A"
    );

    // Koopmans ionization potential, at the optimized geometry (which is where MOPAC reports
    // it — comparing against the input geometry would confuse a geometry difference for a
    // parameter error).
    let ip = -opt.scf.homo_ev.unwrap();
    let d_ip = ip - reference.ionization;
    eprintln!(
        "    {name} CO2 optimized  IP  {ip:>11.6}  MOPAC {:>11.6}  delta {d_ip:+.6} eV",
        reference.ionization
    );
    assert!(
        d_ip.abs() < tolerance::IONIZATION,
        "{name} ionization potential off by {d_ip:.6} eV"
    );

    // **The whole orbital spectrum against MOPAC's own.** Twelve eigenvalues, occupied and
    // virtual, rather than the single frontier one above. This is the sharpest comparison here:
    // it pins every level of the converged Fock matrix, and it carries CO₂'s two degenerate
    // pairs, which a broken two-centre rotation would split while leaving ΔHf and the HOMO
    // almost unmoved.
    let mine = &opt.scf.mo_energies;
    assert_eq!(
        mine.len(),
        reference.orbital_energies.len(),
        "{name}: expected {} orbitals, got {}",
        reference.orbital_energies.len(),
        mine.len()
    );
    let mut worst_orbital = 0.0_f64;
    let mut worst_index = 0;
    for (i, (got, want)) in mine.iter().zip(reference.orbital_energies).enumerate() {
        let d = (got - want).abs();
        if d > worst_orbital {
            worst_orbital = d;
            worst_index = i;
        }
    }
    eprintln!(
        "    {name} CO2 optimized  {} orbitals: worst |delta| = {worst_orbital:.6} eV at #{} \
         ({:.5} vs MOPAC {:.5})",
        mine.len(),
        worst_index + 1,
        mine[worst_index],
        reference.orbital_energies[worst_index]
    );
    assert!(
        worst_orbital < tolerance::ORBITAL,
        "{name} orbital energy #{} off by {worst_orbital:.6} eV",
        worst_index + 1
    );
    // The degeneracies, explicitly: MOPAC prints them equal to five decimals, so they have to
    // come out equal here too. This is what a broken rotation matrix would break first.
    for (a, b) in [(3usize, 4usize), (6, 7), (8, 9)] {
        let split = (mine[a] - mine[b]).abs();
        assert!(
            split < 1.0e-6,
            "{name}: orbitals {} and {} should be degenerate, and differ by {split:.3e} eV",
            a + 1,
            b + 1
        );
    }

    // **Mulliken charges against MOPAC's own.** The heat of formation is one number, and a
    // compensating pair of errors survives it; the charges say whether the density itself is
    // right, and they are what AM1-BCC is built on.
    let d_qc = opt.scf.charges[0] - reference.charge_carbon;
    let d_qo = opt.scf.charges[1] - (-0.5 * reference.charge_carbon);
    eprintln!(
        "    {name} CO2 optimized  q(C) {:>10.6}  MOPAC {:>10.6}  delta {d_qc:+.6} e",
        opt.scf.charges[0], reference.charge_carbon
    );
    eprintln!(
        "    {name} CO2 optimized  q(O) {:>10.6}  MOPAC {:>10.6}  delta {d_qo:+.6} e",
        opt.scf.charges[1],
        -0.5 * reference.charge_carbon
    );
    assert!(
        d_qc.abs() < tolerance::CHARGE,
        "{name} carbon Mulliken charge off by {d_qc:.6} e"
    );
    assert!(
        d_qo.abs() < tolerance::CHARGE,
        "{name} oxygen Mulliken charge off by {d_qo:.6} e"
    );

    // CO2 is linear and centrosymmetric, so the dipole must vanish by symmetry and the
    // charges must sum to zero — cheap checks that catch a broken assembly. MOPAC reports
    // 0.000 for every dipole component here, so this is also the oracle comparison.
    assert!(
        opt.scf.dipole_magnitude < tolerance::DIPOLE,
        "{name} CO2 dipole should vanish by symmetry, got {}",
        opt.scf.dipole_magnitude
    );
    let qsum: f64 = opt.scf.charges.iter().sum();
    assert!(qsum.abs() < 1.0e-8, "{name} charges sum to {qsum}");
    assert!(
        opt.scf.charges[0] > 0.0 && opt.scf.charges[1] < 0.0,
        "{name} CO2 should have a positive carbon and negative oxygens, got {:?}",
        opt.scf.charges
    );
    // The two oxygens are equivalent by symmetry; a difference between them would mean the
    // assembly is not respecting it.
    assert!(
        (opt.scf.charges[1] - opt.scf.charges[2]).abs() < 1.0e-9,
        "{name} the two oxygens are inequivalent: {:?}",
        opt.scf.charges
    );

    (d_input, d_opt)
}

#[test]
fn am1_reproduces_the_mopac_am1_reference() {
    check(&AM1_CO2);
}

#[test]
fn rm1_reproduces_the_mopac_rm1_reference() {
    check(&RM1_CO2);
}

#[test]
fn the_residual_offset_is_the_constants_set_not_the_parameters() {
    // The claim this test defends: the disagreement with modern MOPAC is a property of the
    // crate's historical constants, so it must be the same for two different parameter sets
    // running through the same code. If a parameter table were mis-parsed, its offset would
    // move independently of the other's.
    let (am1_input, am1_opt) = check(&AM1_CO2);
    let (rm1_input, rm1_opt) = check(&RM1_CO2);

    eprintln!(
        "    offsets: AM1 {am1_input:+.5} / {am1_opt:+.5},  RM1 {rm1_input:+.5} / {rm1_opt:+.5} kcal/mol"
    );
    assert!(
        (am1_opt - rm1_opt).abs() < 0.02,
        "AM1 and RM1 offsets from MOPAC differ by {:.5} kcal/mol; they should track each \
         other if the cause is the shared constants rather than a parameter table",
        am1_opt - rm1_opt
    );
    assert!(
        am1_opt.signum() == rm1_opt.signum(),
        "AM1 and RM1 offsets have opposite signs, which a shared cause would not produce"
    );
}

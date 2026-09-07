// SPDX-License-Identifier: GPL-3.0-or-later

//! Embedded AM1 and RM1 parameter tables, and per-element reference data.
//!
//! The per-element parameters (`U_ss … alpha` plus the core-core Gaussian `K/L/M` triples) live
//! in the embedded CSVs below. The isolated-atom occupation *coefficients* are closed forms in
//! `(n_s, n_p)` reproducing MOPAC's `calpar.f`. [`EHEAT_KCAL`] tabulates the same quantity as
//! MOPAC's `eheat` and mostly agrees with it; `third_party/mopac/README.md` (Apache-2.0,
//! Copyright 2021 Virginia Polytechnic Institute and State University) records where they differ
//! and what was checked. [`MASS`] is **not** MOPAC's table -- its values are modern IUPAC
//! standard atomic weights, which differ from MOPAC's older set for a dozen elements. [`QN`],
//! [`N_S`] and [`N_P`] are periodic-table facts with no MOPAC-specific content.
//!
//! Index every array/function by **atomic number**.
//!
//! # Scientific sources
//!
//! * **AM1** — M. J. S. Dewar, E. G. Zoebisch, E. F. Healy & J. J. P. Stewart, "AM1: A New
//!   General Purpose Quantum Mechanical Molecular Model," *J. Am. Chem. Soc.* **107**,
//!   3902–3909 (1985), plus the per-element extension papers by Dewar and co-workers, as
//!   consolidated in MOPAC.
//! * **RM1** — G. B. Rocha, R. O. Freire, A. M. Simas & J. J. P. Stewart, "RM1: A
//!   Reparameterization of AM1 for H, C, N, O, P, S, F, Cl, Br and I," *J. Comput. Chem.*
//!   **27**, 1101–1111 (2006).
//!
//! # Provenance of the machine-readable tables
//!
//! The values above are published scientific facts; these are the particular tabulations of them
//! that this crate ships, and they came from different places:
//!
//! * `data/am1_parameters.csv` — from the **PySEQM** reference implementation (LANL,
//!   BSD-3-Clause). See `third_party/pyseqm/README.md`.
//! * `data/rm1_parameters.csv` — extracted from **MOPAC**'s Fortran source (Apache-2.0) by
//!   `tools/extract_rm1_parameters.py`, which is kept in the repository so the extraction is
//!   reproducible rather than a set of hand-copied numbers. See `third_party/mopac/README.md`.
//!
//! Each CSV repeats its own provenance in its header line, so the attribution travels with the
//! data even when the file is read on its own. `THIRD_PARTY_NOTICES.md` is the cross-cutting
//! account.

/// Raw AM1 parameter table (Dewar, Zoebisch, Healy & Stewart 1985 and the element-extension
/// papers), from PySEQM's `parameters_AM1_MOPAC.csv`.
pub const AM1_PARAM_CSV: &str = include_str!("data/am1_parameters.csv");

/// Raw RM1 parameter table — G. B. Rocha, R. O. Freire, A. M. Simas & J. J. P. Stewart, "RM1: A
/// Reparameterization of AM1 for H, C, N, O, P, S, F, Cl, Br and I," *J. Comput. Chem.* **27**,
/// 1101–1111 (2006) — extracted from MOPAC's Fortran tabulation by
/// `tools/extract_rm1_parameters.py`. Same schema and same functional form as AM1; see the CSV
/// header, `third_party/mopac/README.md` and `THIRD_PARTY_NOTICES.md` for provenance.
pub const RM1_PARAM_CSV: &str = include_str!("data/rm1_parameters.csv");

/// Valence-shell principal quantum number `n` per Z (0 for unsupported).
pub const QN: [u8; 87] = [
    0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
];

/// Number of s electrons in the neutral-atom valence configuration.
pub const N_S: [f64; 87] = [
    0.0, 1.0, 0.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.0,
    1.0, 2.0, 2.0, 2.0, 2.0, 1.0, 2.0, 2.0, 2.0, 2.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.0, 1.0,
    2.0, 2.0, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 0.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.0, 1.0, 2.0,
    2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
];

/// Number of p electrons in the neutral-atom valence configuration.
pub const N_P: [f64; 87] = [
    0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
];

/// Experimental gas-phase atomic heats of formation ΔH_f (kcal/mol), MOPAC `block.f`, indexed by
/// **atomic number**.
///
/// Carried the same fourteen-place shift above caesium that [`MASS`] did — see the note there.
/// It was less visible because someone had already noticed the symptom for the one element that
/// reaches it and written mercury's 14.690 into index 80 by hand, leaving the shifted copy at
/// index 66 in place; the entries between were simply the wrong atom's value.
///
/// Zero above Z = 56 means this crate has no value, not that none exists: MOPAC tabulates the
/// lanthanides and this table does not, and eighteen entries at Z <= 86 differ from MOPAC's
/// current release (sodium 25.850 against 25.650, vanadium 122.300 against 122.900). Neither
/// matters for any element a parameter set covers.
pub const EHEAT_KCAL: [f64; 87] = [
    0.0, 52.102, 0.0, 38.410, 76.960, 135.700, 170.890, 113.000, 59.559, 18.890, 0.0, 25.850,
    35.000, 79.490, 108.390, 75.570, 66.400, 28.990, 0.0, 21.420, 42.600, 90.300, 112.300, 122.300,
    95.000, 67.700, 99.300, 102.400, 102.800, 80.700, 31.170, 65.400, 89.500, 72.300, 54.300,
    26.740, 0.0, 19.600, 39.100, 101.500, 145.500, 172.400, 157.300, 0.0, 155.500, 133.000, 90.000,
    68.100, 26.720, 58.000, 72.200, 63.200, 47.000, 25.517, 0.0, 18.700, 42.500,
    // 57–71: La–Lu, none of which MOPAC tabulates.
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    // 72–86: Hf–Rn, which used to start at index 58.
    148.000, 186.900, 203.100, 185.000, 188.000, 160.000, 135.200, 88.000, 14.690, 43.550, 46.620,
    50.100, 0.0, 0.0, 0.0,
];

/// Atomic masses (amu), indexed by **atomic number**. Modern IUPAC standard atomic weights ---
/// **not** MOPAC's table, whose values differ for a dozen elements (Tc 98.9062 against 97.0, Ti
/// 47.90 against 47.867, W 183.85 against 183.84).
///
/// # The lanthanide gap
///
/// Through 0.2.2 this table skipped fourteen slots above caesium: La–Yb (57–70) were absent, so
/// every element above them sat fourteen places too low. Mercury's 200.59 was at index 66 and
/// `MASS[80]` was `0.0`.
///
/// A zero mass is not a wrong number, it is a division by zero. The mass-weighting
/// `C_{aα,bβ}/√(M_a M_b)` returned `±∞` for every row of a mercury atom, and that surfaced as
/// "eigendecomposition failed" out of a fluorite HgF₂ phonon calculation — three layers from the
/// table, with every intermediate quantity of the DFPT assembly verified finite. Hg is the only
/// element above `Z = 56` that AM1 parameterizes, so it was also the only one that could reach it.
///
/// **Where the gap came from is not known.** The obvious suspect was MOPAC, which omits exactly
/// those fourteen elements for want of NDDO parameters — but its `parameters_C.F90` is indexed by
/// atomic number and has real lanthanide entries, so the shape is not MOPAC's, and the *values*
/// here are not MOPAC's either. An older release, an intermediate source, or a transcription that
/// skipped fourteen rows would all produce this; the evidence does not distinguish them.
///
/// The gap is reopened: La–Lu carry their real masses even though no parameter set covers them,
/// because this table claims to be atomic masses by Z, and one that is right only where it happens
/// to be consulted is the same bug waiting for the next element. [`EHEAT_KCAL`] had the identical
/// shift and is corrected the same way. `tests/atomic_data.rs` pins it with a *structural* check —
/// mass rises with atomic number except at Ar/K, Co/Ni and Te/I — which catches a shift anywhere,
/// not only where someone thought to write an anchor.
pub const MASS: [f64; 87] = [
    0.0, 1.0079, 4.0026, 6.94, 9.01218, 10.81, 12.011, 14.0067, 15.9994, 18.9984, 20.179, 22.98977,
    24.305, 26.98154, 28.0855, 30.97376, 32.06, 35.453, 39.948, 39.098, 40.078, 44.956, 47.867,
    50.942, 51.996, 54.938, 55.845, 58.933, 58.693, 63.546, 65.38, 69.723, 72.63, 74.922, 78.971,
    79.904, 83.798, 85.468, 87.62, 88.906, 91.224, 92.906, 95.95, 97.0, 101.07, 102.91, 106.42,
    107.87, 112.41, 114.82, 118.71, 121.76, 127.6, 126.9, 131.29, 132.91, 137.33,
    // 57–70: La–Yb, the fourteen MOPAC leaves out.
    138.91, 140.12, 140.91, 144.24, 145.0, 150.36, 151.96, 157.25, 158.93, 162.50, 164.93, 167.26,
    168.93, 173.05, // 71–86: Lu–Rn, which used to start at index 57.
    174.97, 178.49, 180.95, 183.84, 186.21, 190.23, 192.22, 195.08, 196.97, 200.59, 204.38, 207.2,
    208.98, 209.0, 210.0, 222.0,
];

/// The atomic mass of `z`, refusing the zero that a gap in [`MASS`] would otherwise hand to a
/// mass-weighting as `1/0`.
///
/// Every vibrational quantity divides by `√(M_a M_b)`, and a missing mass therefore does not
/// produce a wrong frequency — it produces `±∞`, which the eigensolver reports as a failure to
/// decompose, at a distance from the actual cause. This is where the cause gets named.
pub fn atomic_mass(z: u8) -> crate::error::Result<f64> {
    let m = MASS.get(z as usize).copied().unwrap_or(0.0);
    if m > 0.0 {
        return Ok(m);
    }
    Err(crate::error::Am1Error::InvalidInput(format!(
        "no atomic mass is tabulated for element Z={z}, so the mass-weighted Hessian would \
         divide by zero; add it to `data_tables::MASS`"
    )))
}

/// [`atomic_mass`] for every atom of a molecule, checked before anything divides by one.
pub fn require_masses(molecule: &crate::system::Molecule) -> crate::error::Result<()> {
    for atom in &molecule.atoms {
        atomic_mass(atom.z)?;
    }
    Ok(())
}

// Average-of-configuration coefficients for the isolated-atom electronic energy
// (MOPAC `calpar.f`). For every AM1 element these reduce to closed forms in the
// neutral valence occupation (n_s, n_p); reproduced here as functions to avoid
// transcription error.

fn ns(z: u8) -> f64 {
    N_S.get(z as usize).copied().unwrap_or(0.0)
}
fn np(z: u8) -> f64 {
    N_P.get(z as usize).copied().unwrap_or(0.0)
}

/// `Gss` coefficient = n_s(n_s−1)/2.
pub fn gssc(z: u8) -> f64 {
    let n = ns(z);
    n * (n - 1.0) / 2.0
}
/// `Gsp` coefficient = n_s·n_p.
pub fn gspc(z: u8) -> f64 {
    ns(z) * np(z)
}
/// `Hsp` coefficient = −n_p (0 for the closed p⁶ shell).
pub fn hspc(z: u8) -> f64 {
    let n = np(z) as i32;
    if n == 6 {
        0.0
    } else {
        -(n as f64)
    }
}
/// `Gp2` coefficient (average-of-configuration of p^n).
pub fn gp2c(z: u8) -> f64 {
    match np(z) as i32 {
        2 => 1.5,
        3 => 4.5,
        4 => 6.5,
        5 => 10.0,
        _ => 0.0,
    }
}
/// `Gpp` coefficient (average-of-configuration of p^n).
pub fn gppc(z: u8) -> f64 {
    match np(z) as i32 {
        2 => -0.5,
        3 => -1.5,
        4 => -0.5,
        _ => 0.0,
    }
}

/// AM1 core charge (number of valence electrons) = `N_S[z] + N_P[z]`.
/// For every AM1 sp element this equals the true core charge (H=1, C=4, O=6, Zn=2, …),
/// avoiding the transition-metal `tore` ambiguity.
#[inline]
pub fn core_charge(z: u8) -> f64 {
    ns(z) + np(z)
}

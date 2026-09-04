// SPDX-License-Identifier: GPL-3.0-or-later

//! Embedded AM1 parameter tables and per-element reference data.
//!
//! The per-element AM1 parameters (`U_ss … alpha` plus the AM1 core-core Gaussian
//! `K/L/M` triples) live in the embedded CSV `data/am1_parameters.csv`, whose numeric
//! values are the standard published AM1 set (Dewar, Zoebisch, Healy & Stewart 1985 and
//! the per-element extension papers, as consolidated in MOPAC). The isolated-atom
//! occupation coefficients reproduce MOPAC's `calpar.f` average-of-configuration
//! coefficients; the experimental atomic heats of formation reproduce `block.f`. Index
//! every array/function by atomic number.
//!
//! Provenance / attribution: the machine-readable parameter table was taken from the
//! PySEQM reference implementation (LANL, BSD-3-Clause). See `THIRD_PARTY_NOTICES.md` and
//! `third_party/pyseqm/LICENSE` at the crate root, and the header of the CSV itself.

/// Raw AM1 parameter table, parsed from the embedded MOPAC CSV.
pub const AM1_PARAM_CSV: &str = include_str!("data/am1_parameters.csv");

/// Raw RM1 parameter table (Rocha, Freire, Simas & Stewart 2006), extracted from MOPAC's
/// Fortran tabulation by `tools/extract_rm1_parameters.py`. Same schema and same functional
/// form as AM1; see the CSV header and `THIRD_PARTY_NOTICES.md` for provenance.
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

/// Experimental gas-phase atomic heats of formation ΔH_f (kcal/mol), MOPAC `block.f`.
pub const EHEAT_KCAL: [f64; 87] = [
    0.0, 52.102, 0.0, 38.410, 76.960, 135.700, 170.890, 113.000, 59.559, 18.890, 0.0, 25.850,
    35.000, 79.490, 108.390, 75.570, 66.400, 28.990, 0.0, 21.420, 42.600, 90.300, 112.300, 122.300,
    95.000, 67.700, 99.300, 102.400, 102.800, 80.700, 31.170, 65.400, 89.500, 72.300, 54.300,
    26.740, 0.0, 19.600, 39.100, 101.500, 145.500, 172.400, 157.300, 0.0, 155.500, 133.000, 90.000,
    68.100, 26.720, 58.000, 72.200, 63.200, 47.000, 25.517, 0.0, 18.700, 42.500, 0.0, 148.000,
    186.900, 203.100, 185.000, 188.000, 160.000, 135.200, 88.000, 14.690, 43.550, 46.620, 50.100,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 14.690, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
];

/// Atomic masses (amu).
pub const MASS: [f64; 87] = [
    0.0, 1.0079, 4.0026, 6.94, 9.01218, 10.81, 12.011, 14.0067, 15.9994, 18.9984, 20.179, 22.98977,
    24.305, 26.98154, 28.0855, 30.97376, 32.06, 35.453, 39.948, 39.098, 40.078, 44.956, 47.867,
    50.942, 51.996, 54.938, 55.845, 58.933, 58.693, 63.546, 65.38, 69.723, 72.63, 74.922, 78.971,
    79.904, 83.798, 85.468, 87.62, 88.906, 91.224, 92.906, 95.95, 97.0, 101.07, 102.91, 106.42,
    107.87, 112.41, 114.82, 118.71, 121.76, 127.6, 126.9, 131.29, 132.91, 137.33, 174.97, 178.49,
    180.95, 183.84, 186.21, 190.23, 192.22, 195.08, 196.97, 200.59, 204.38, 207.2, 208.98, 209.0,
    210.0, 222.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
];

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

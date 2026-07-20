// SPDX-License-Identifier: GPL-3.0-or-later

//! Physical constants and unit conversions.
//!
//! Units policy: the semiempirical block is computed in **eV with distances in Bohr**.
//! To reproduce published MOPAC/AM1 numbers exactly we adopt MOPAC7's model constants
//! (`ev = 27.21`, `a0 = 0.529167 Å`, `1 eV = 23.061 kcal/mol`) rather than CODATA — the
//! AM1 parameters and the ρ additive terms were fit against these. The native API then
//! reports energies in Hartree (= eV / `AM1_EV`) and coordinates in Bohr; the ASE layer
//! reports eV / Å (AM1 energies are natively in eV, so that boundary is exact).

/// MOPAC7 eV per Hartree (the value AM1 was parametrized against).
pub const AM1_EV: f64 = 27.21;
/// MOPAC7 Bohr radius in Ångström.
pub const AM1_A0: f64 = 0.529167;

pub const HARTREE_TO_EV: f64 = AM1_EV;
pub const EV_TO_HARTREE: f64 = 1.0 / AM1_EV;

pub const ANGSTROM_TO_BOHR: f64 = 1.0 / AM1_A0;
pub const BOHR_TO_ANGSTROM: f64 = AM1_A0;

/// 1 eV in kcal/mol (MOPAC7 value); heats of formation are reported in kcal/mol.
pub const EV_TO_KCAL: f64 = 23.061;
pub const KCAL_TO_EV: f64 = 1.0 / EV_TO_KCAL;

/// Force conversion for the ASE boundary: Hartree/Bohr → eV/Å. (AM1 forces are natively
/// eV/Å; this is only used if a caller works from atomic-unit gradients.)
pub const HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM: f64 = HARTREE_TO_EV / BOHR_TO_ANGSTROM;

/// Atomic-unit dipole (e·a0) to Debye.
pub const AU_DIPOLE_TO_DEBYE: f64 = 2.541_746_473;

/// Cordero/Pyykkö-style covalent radii (Å), used only for geometric bond perception
/// in the AM1-BCC topology layer — generic element data, not AM1 model parameters.
/// Index by atomic number; unknown Z falls back to 1.5 Å.
pub fn covalent_radius_angstrom(z: u8) -> f64 {
    const RAD_A: [f64; 87] = [
        0.0, 0.31, 0.28, 1.28, 0.96, 0.84, 0.76, 0.71, 0.66, 0.57, 0.58, 1.66, 1.41, 1.21, 1.11,
        1.07, 1.05, 1.02, 1.06, 2.03, 1.76, 1.70, 1.60, 1.53, 1.39, 1.39, 1.32, 1.26, 1.24, 1.32,
        1.22, 1.22, 1.20, 1.19, 1.20, 1.20, 1.16, 2.20, 1.95, 1.90, 1.75, 1.64, 1.54, 1.47, 1.46,
        1.42, 1.39, 1.45, 1.44, 1.42, 1.39, 1.39, 1.38, 1.39, 1.40, 2.44, 2.15, 2.07, 2.04, 2.03,
        2.01, 1.99, 1.98, 1.98, 1.96, 1.94, 1.92, 1.92, 1.89, 1.90, 1.87, 1.87, 1.75, 1.70, 1.62,
        1.51, 1.44, 1.41, 1.36, 1.36, 1.32, 1.45, 1.46, 1.48, 1.40, 1.50, 1.50,
    ];
    RAD_A.get(z as usize).copied().unwrap_or(1.5)
}

// SPDX-License-Identifier: GPL-3.0-or-later

//! Infrared intensities, and the atomic polar tensor they come from.
//!
//! # The tensor
//!
//! The **atomic polar tensor** (APT) is `∂μ_α/∂R_{a,β}` — the molecular counterpart of the Born
//! effective charges [`crate::pbc::born_charges`] computes for a crystal, and in the same units
//! (e). With the dipole written as [`crate::dipole`] defines it,
//!
//! ```text
//! μ_α = Σ_b Z_b R_{b,α} − Tr[P M_α]
//! ```
//!
//! differentiating at the self-consistent density gives
//!
//! ```text
//! ∂μ_α/∂R_{a,β} = δ_αβ Q_a − Tr[ (∂P/∂R_{a,β}) M_α ]
//! ```
//!
//! because `∂M_α/∂R_{a,β}` is `δ_αβ` on atom `a`'s diagonal block and zero elsewhere, and its
//! trace against `P` supplies the `−δ_αβ p_a` that turns `Z_a` into the net charge `Q_a`. The
//! second term is the electrons rearranging, and it is the CPHF response
//! ([`crate::hessian::analytic_hessian_with_response`]) — the same `U` the Hessian is built from,
//! so an infrared spectrum costs a Hessian and nothing more.
//!
//! Expanding `Tr[∂P M_α]` reproduces the charge-transfer and `s`–`p` hybridization terms that
//! [`crate::pbc::born_charges`] writes out separately. They are the same expression; keeping it
//! in the compact form here is what stops the molecular and periodic versions from drifting.
//!
//! # The sum rule that checks it
//!
//! ```text
//! Σ_a ∂μ_α/∂R_{a,β} = q δ_αβ
//! ```
//!
//! Translating the whole molecule moves the charge `q` and nothing else. This is `3 × 3`
//! constraints on a `3 × 3N` tensor and it follows from charge conservation alone, so a violation
//! is a defect in the response rather than a property of the molecule. `tests/ir.rs` asserts it,
//! and it is a far sharper instrument than checking that a symmetric mode comes out dark.
//!
//! # Intensities
//!
//! ```text
//! ∂μ/∂Q_k = Σ_j (∂μ/∂R_j) L_{jk} / √m_j          L mass-weighted, orthonormal
//! A_k     = 42.2561 × |∂μ/∂Q_k|²                  km/mol, with ∂μ/∂Q in D·Å⁻¹·amu^{−1/2}
//! ```
//!
//! The `42.2561` is `N_A/(12 ε₀ c²)` expressed in those units — a conversion between unit systems
//! of an already-computed observable, involving no model quantity, so it uses CODATA values. The
//! step *into* those units, `1 e = 4.803 D/Å`, does involve one: it divides by the Bohr radius,
//! and this crate's Bohr is MOPAC7's `0.529167 Å` rather than CODATA's, because that is the
//! length its own dipoles are expressed in. Mixing the two would be a fraction of a percent on
//! every intensity, which is exactly the kind of error that never announces itself.

use crate::basis::Basis;
use crate::constants::{ANGSTROM_TO_BOHR, AU_DIPOLE_TO_DEBYE};
use crate::data_tables::MASS;
use crate::error::Result;
use crate::hessian::{
    analytic_hessian_with_response, vibrational_analysis_from_hessian, HessianResponse,
    ResponseChannel, VibrationalModes,
};
use crate::linalg::Matrix;
use crate::params::Am1Parameters;
use crate::scf::Am1Options;
use crate::system::Molecule;

/// `1 e` expressed in `Debye / Ångström`, in this crate's length unit (≈ 4.803).
///
/// Derived rather than written down: `AU_DIPOLE_TO_DEBYE / a₀[Å]`. See the module documentation
/// for why the crate's own `a₀` is the right one here.
pub const E_TO_DEBYE_PER_ANGSTROM: f64 = AU_DIPOLE_TO_DEBYE * ANGSTROM_TO_BOHR;

/// `N_A / (12 ε₀ c²)` in km/mol per `(D·Å⁻¹·amu^{−1/2})²`.
pub const IR_INTENSITY_KM_PER_MOL: f64 = 42.2561;

/// An infrared spectrum: the raw tensor and the mode-resolved intensities.
#[derive(Clone, Debug)]
pub struct IrSpectrum {
    /// The **atomic polar tensor** `∂μ_α/∂R_{a,β}`, a dense `3 × 3N` matrix in units of `e`.
    ///
    /// Row `α` is a Cartesian dipole component; column `3a + β` is atom `a`, axis `β`. This is
    /// the raw quantity — mode-independent, geometry-only — from which every intensity below is
    /// a projection.
    pub dipole_derivatives: Matrix,
    /// Harmonic frequencies (cm⁻¹), ascending, one per mode; negative denotes imaginary.
    pub frequencies_cm: Vec<f64>,
    /// Integrated absorption coefficient per mode, km/mol.
    pub intensities_km_per_mol: Vec<f64>,
    /// `∂μ/∂Q_k` per mode, a `3 × 3N` matrix in `D·Å⁻¹·amu^{−1/2}`; columns are modes.
    ///
    /// The dense per-mode tensor, kept because the intensity throws away the *direction* of the
    /// transition dipole and that direction is what a polarized measurement sees.
    pub mode_dipole_derivatives: Matrix,
    /// The normal modes themselves, including the rigid-body overlap that says which are which.
    pub modes: VibrationalModes,
}

impl IrSpectrum {
    /// Modes whose rigid-body overlap is below `threshold` (0.5 is a reasonable split), as
    /// `(index, frequency_cm, intensity)`.
    ///
    /// Translations and rotations have an intensity too — a rigid molecule with a net charge
    /// really does absorb — but it is not a vibrational band, and filtering by *what the
    /// eigenvector is* rather than by a frequency cutoff is what makes this correct for a linear
    /// molecule, which has five rigid-body modes rather than six.
    pub fn vibrational_bands(&self, threshold: f64) -> Vec<(usize, f64, f64)> {
        self.modes
            .translation_rotation_overlap
            .iter()
            .enumerate()
            .filter(|(_, o)| **o < threshold)
            .map(|(k, _)| (k, self.frequencies_cm[k], self.intensities_km_per_mol[k]))
            .collect()
    }
}

/// The atomic polar tensor `∂μ_α/∂R_{a,β}`, `3 × 3N`, in units of `e`.
///
/// Computed from an already-solved CPHF response, so a caller that has a Hessian in hand pays
/// nothing extra.
/// # Contracted in the MO block, not the AO one
///
/// The obvious form builds `∂P/∂R_j` — an `nao × nao` matrix — for each of the `3N` perturbations
/// and traces it against `M_α`. That is what this did, and it is an order too expensive: each
/// build is `O(nao² n_occ)`, so the loop is `O(N⁴)` for three numbers per perturbation.
///
/// Writing `∂P = B + Bᵀ` with `B = w C_v U C_oᵀ` and using that `M_α` is symmetric,
///
/// ```text
/// Tr[∂P M_α] = 2 Tr[B M_α] = 2w Tr[Uᵀ (C_vᵀ M_α C_o)]
/// ```
///
/// so the `nao²` object never has to exist: project `M_α` into the occupied–virtual block **once**
/// (three small matrices), and every perturbation is then one Frobenius product of `n_vir × n_occ`.
/// `O(nao² n_vir)` once plus `O(ndof · n_ov)`, which is `O(N³)`.
///
/// The factor is `2w` — **4** for RHF (`w = 2`) and **2 per spin** for UHF (`w = 1`), summing to 4
/// again. That is the same convention the periodic Hessian's relaxation term uses, and getting it
/// wrong is caught immediately by the translational sum rule `Σ_a ∂μ/∂R_a = q δ`, which
/// `tests/ir.rs` holds to 3e-15.
pub fn dipole_derivatives_from_response(
    molecule: &Molecule,
    params: &Am1Parameters,
    response: &HessianResponse,
) -> Result<Matrix> {
    let basis = Basis::build(molecule, params)?;
    let m = crate::dipole::dipole_operator(molecule, &basis, params)?;
    let charges = crate::pbc::ewald::net_charges(molecule, &basis, params, &response.scf.density)?;
    let ndof = response.ndof();

    // `W^σ_α = C_vᵀ M_α C_o`, once per spin channel per Cartesian component.
    let project = |ch: &ResponseChannel| -> [Matrix; 3] {
        std::array::from_fn(|a| ch.virtuals.transpose_matmul(&m[a]).matmul(&ch.occupied))
    };
    let w_alpha = project(&response.alpha);
    let w_beta = response.beta.as_ref().map(project);

    let mut apt = Matrix::zeros(3, ndof);
    for dof in 0..ndof {
        let (atom, axis) = (dof / 3, dof % 3);
        for alpha in 0..3 {
            let own = if alpha == axis { charges[atom] } else { 0.0 };
            let mut trace = 2.0
                * response.alpha.occupation
                * response.alpha.u_ov[dof].frobenius_dot(&w_alpha[alpha]);
            if let (Some(b), Some(wb)) = (&response.beta, &w_beta) {
                trace += 2.0 * b.occupation * b.u_ov[dof].frobenius_dot(&wb[alpha]);
            }
            apt[(alpha, dof)] = own - trace;
        }
    }
    Ok(apt)
}

/// The atomic polar tensor at a geometry, solving the CPHF for it.
pub fn dipole_derivatives(
    molecule: &Molecule,
    params: &Am1Parameters,
    options: &Am1Options,
) -> Result<Matrix> {
    let response = analytic_hessian_with_response(molecule, params, options, 1.0e-3)?;
    dipole_derivatives_from_response(molecule, params, &response)
}

/// The full infrared spectrum: atomic polar tensor, normal modes, and intensities.
///
/// One Hessian, one CPHF solve. Evaluate at a **stationary point** — the intensities are defined
/// against harmonic normal modes, and away from a minimum those modes are not what is observed.
pub fn ir_spectrum(
    molecule: &Molecule,
    params: &Am1Parameters,
    options: &Am1Options,
) -> Result<IrSpectrum> {
    let response = analytic_hessian_with_response(molecule, params, options, 1.0e-3)?;
    ir_spectrum_from_response(molecule, params, &response)
}

/// [`ir_spectrum`] from a CPHF response already in hand.
///
/// The seam that lets one solve serve the whole vibrational group. A caller wanting the Hessian,
/// the frequencies, the normal modes, the atomic polar tensor *and* the intensities was paying for
/// as many CPHF solves as it asked questions, because each entry point ran its own; they are all
/// contractions of this one `response`.
pub fn ir_spectrum_from_response(
    molecule: &Molecule,
    params: &Am1Parameters,
    response: &HessianResponse,
) -> Result<IrSpectrum> {
    let apt = dipole_derivatives_from_response(molecule, params, response)?;
    let modes = vibrational_analysis_from_hessian(molecule, response.hessian.clone())?;
    Ok(assemble(molecule, apt, modes))
}

/// Project an atomic polar tensor onto normal modes and convert to km/mol.
fn assemble(molecule: &Molecule, apt: Matrix, modes: VibrationalModes) -> IrSpectrum {
    let ndof = apt.cols;
    let inv_sqrt_mass: Vec<f64> = (0..ndof)
        .map(|j| 1.0 / MASS[molecule.atoms[j / 3].z as usize].sqrt())
        .collect();

    let mut mode_apt = Matrix::zeros(3, ndof);
    let mut intensities = vec![0.0; ndof];
    for k in 0..ndof {
        let mut norm2 = 0.0;
        for alpha in 0..3 {
            // ∂μ_α/∂Q_k = Σ_j (∂μ_α/∂R_j) L_{jk} / √m_j, then e → D/Å.
            let mut acc = 0.0;
            for j in 0..ndof {
                acc += apt[(alpha, j)] * modes.modes[(j, k)] * inv_sqrt_mass[j];
            }
            let v = acc * E_TO_DEBYE_PER_ANGSTROM;
            mode_apt[(alpha, k)] = v;
            norm2 += v * v;
        }
        intensities[k] = IR_INTENSITY_KM_PER_MOL * norm2;
    }

    IrSpectrum {
        dipole_derivatives: apt,
        frequencies_cm: modes.frequencies_cm.clone(),
        intensities_km_per_mol: intensities,
        mode_dipole_derivatives: mode_apt,
        modes,
    }
}

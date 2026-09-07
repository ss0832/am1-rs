// SPDX-License-Identifier: GPL-3.0-or-later

//! Periodic boundary conditions.
//!
//! Layout:
//!
//! - [`kpoints`]: Brillouin-zone meshes and the Bloch phase.
//! - [`complex`]: complex Hermitian matrices and their eigendecomposition. NDDO's orthonormal
//!   AO basis makes `S(k) = I`, so this is the *standard* Hermitian problem rather than a
//!   generalized one.
//! - [`phonon`]: real-space force constants `Φ(T)`, the dynamical matrix `D(q)`, and band
//!   structures.
//!
//! The Γ-point energy path does not live here: at `k = 0` every Bloch phase is 1, so it is
//! the ordinary molecular assembly run over an image-aware pair list, and it stays in
//! `hamiltonian`/`fock`/`scf` where the molecular code is. Only what genuinely needs complex
//! arithmetic or a k-mesh is in this module.
//!
//! The Γ-point **Hessian** is in `hessian` for the same reason, and is the molecular one: it
//! walks whichever pair list the SCF used, so a periodic molecule gets the image pairs. That
//! works precisely because `P(0,T) = P(Γ)` for every translation at `k = 0`; away from Γ it
//! would not, which is why [`phonon`] reaches other `q` through a supercell rather than by
//! generalizing the Hessian.

pub mod berry;
pub mod complex;
pub mod dfpt;
pub mod ewald;
pub mod ewald1d;
pub mod ewald2d;
pub mod extent;
pub mod finite_field;
pub mod gradient;
pub mod hessian;
pub(crate) mod kerker;
pub mod kpoints;
pub mod optimizer;
pub mod phonon;
pub mod scf;

pub use complex::{hermitian_eigen, CEigen, CMatrix};
pub use dfpt::{
    dynamical_matrix_dfpt, dynamical_matrix_dfpt_with, force_constants_at_q,
    force_constants_at_q_with, frequencies_dfpt, frequencies_dfpt_with, DfptOptions, DfptResult,
    LongRange,
};
pub use ewald::{
    default_alpha, phased_delta, q_cartesian, EwaldSum, LongRangeKernel, LongRangeMonopole,
    PhasedDelta,
};
pub use ewald1d::{AxisConvention, Ewald1D};
pub use ewald2d::{default_alpha_2d, Ewald2D, SheetConvention};
pub use extent::{
    dielectric_tensor_with_extent, epsilon_from_polarizability, extent_axis_mixing,
    ExtentConvention,
};
pub use finite_field::{run_finite_field, FiniteFieldOptions, FiniteFieldResult};
pub use gradient::{pbc_energy_and_gradient, pbc_gradient, PbcGradient};
pub use hessian::{
    born_charges, dielectric_function, dielectric_origin_sensitivity, dielectric_tensor,
    pbc_hessian, pbc_hessian_skeleton, polarizability, PhononResponse,
};
pub use kpoints::{all_real, KMesh, KPoint};
pub use optimizer::{optimize_periodic, PbcOptOptions, PbcOptResult, PbcOptStep};
pub use phonon::{build_supercell, q_path, ForceConstants};
pub use scf::{run_pbc_scf, PbcOptions, PbcResult, RealSpaceBlocks};

/// How far an occupation may sit from an integer before a response calculation refuses it.
///
/// # Why this is a constant and not an option
///
/// It is not a tolerance in the usual sense — a knob trading accuracy for cost — but the exact
/// point at which the surrounding code stops being able to represent the level. `hessian.rs`
/// classifies a band as occupied when it holds more than `full − 1e-6` electrons and as virtual
/// when it holds less than `1e-6`, and **a band between those is in neither list**: it is dropped
/// from the response, contributing nothing. Choosing a looser threshold here would not make such
/// a band representable, it would only make the dropping quiet again.
///
/// So the check is exactly "the set of bands the response cannot represent is non-empty", and
/// changing the number would break that identity. [`PbcOptions::require_integer_occupations`] is
/// the way out, and it is a yes/no because the choice is between an error and a known-wrong
/// number, not between two accuracies.
pub const INTEGER_OCCUPATION_TOL: f64 = 1.0e-6;

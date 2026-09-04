// SPDX-License-Identifier: GPL-3.0-or-later
//! # am1-rs
//!
//! A Rust-native implementation of the **AM1** (Austin Model 1) semiempirical NDDO
//! quantum-chemistry method, structured after the `gfn1-rs` GFN1-xTB prototype.
//!
//! Provides AM1 heats of formation, Mulliken and **AM1-BCC** partial charges (for
//! AMBER), analytic nuclear gradients, and L-BFGS geometry optimization. A Python
//! binding and an ASE `Calculator` are shipped under the `python` feature.
//!
//! References: Dewar, Zoebisch, Healy & Stewart, *JACS* **107**, 3902 (1985) (AM1);
//! Dewar & Thiel, *JACS* **99**, 4899 (1977) (MNDO integrals); Jakalian *et al.*,
//! *J. Comput. Chem.* **21**, 132 (2000) & **23**, 1623 (2002) (AM1-BCC).

// Index loops over paired arrays are the natural form for the tensor algebra throughout this
// crate: `for i in 0..3 { for j in 0..3 { h[(off+i, off+j)] += te.e1b[i][j] } }` says what it
// means, and the iterator rewrite clippy suggests would obscure which index belongs to which
// operand. Allowed at the crate level rather than sprinkled over a few dozen sites.
#![allow(clippy::needless_range_loop)]
// There is no `unsafe` in this crate and there is no reason for there to be: the numerical
// kernels are dense array arithmetic that the bounds checker does not measurably slow down, and
// the parallelism is rayon's, which is safe by construction. `forbid` rather than `deny` so that
// the decision cannot be undone by a local `#[allow]` in a future edit.
#![forbid(unsafe_code)]

pub mod basis;
pub mod bcc;
pub mod constants;
pub mod data_tables;
pub mod dipole;
pub mod divide_conquer;
pub mod dual;
pub mod dual2;
pub mod error;
pub mod farfield;
pub mod fermi;
pub mod fock;
pub mod gradient;
pub mod hamiltonian;
pub mod hessian;
pub mod integrals;
pub mod ir;
pub mod lattice;
pub mod linalg;
pub mod math;
pub mod method;
pub mod molden;
pub mod neighbors;
pub mod optimizer;
pub mod overlap;
pub mod overlap_numeric;
pub mod params;
pub mod pbc;
pub mod repulsion;
pub mod scf;
pub mod system;
pub mod timing;
pub mod topology;

pub use bcc::{am1_bcc_charges, BccResult};
pub use dipole::{
    dipole_from_density, dipole_operator, field_core_energy, field_gradient, field_hamiltonian,
    nuclear_dipole,
};

#[cfg(feature = "python")]
pub mod python;

pub use divide_conquer::{
    build_subsystems, divide_conquer_gradient, divide_conquer_stress, partition_atoms,
    partition_weight_sum, run_divide_conquer, DcOptions, DcResult, Subsystem,
};
pub use error::{Am1Error, Result};
pub use fermi::{fill, Filling, Level, Occupations};
pub use gradient::{
    analytic_gradient, closed_form_gradient, electronic_gradient_fixed_density,
    energy_at_fixed_density, fixed_density_gradient, numerical_gradient, GradientResult,
};
pub use hamiltonian::CoreBuildOptions;
pub use hessian::{
    analytic_hessian, analytic_hessian_with_response, numerical_hessian, vibrational_analysis,
    vibrational_analysis_from_hessian, HessianResponse, ResponseChannel, VibrationalModes,
};
pub use ir::{dipole_derivatives, ir_spectrum, IrSpectrum};
pub use lattice::{ImageOffset, Lattice, Periodicity};
pub use linalg::Matrix;
pub use math::{Mat3, Vec3};
pub use method::NddoMethod;
pub use molden::{to_molden, write_molden};
pub use neighbors::{NeighborList, PairImage};
pub use optimizer::{optimize, OptOptions, OptResult};
pub use params::{Am1Element, Am1Parameters};
pub use pbc::{
    pbc_energy_and_gradient, pbc_gradient, run_pbc_scf, KMesh, KPoint, PbcGradient, PbcOptions,
    PbcResult,
};
pub use scf::{
    run_am1, Am1Calculator, Am1Options, Am1Result, BetaOrbitals, ScfAccelerator, ScfReference,
};
pub use system::{symbol_to_z, z_to_symbol, Atom, Molecule};

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

pub mod basis;
pub mod bcc;
pub mod constants;
pub mod data_tables;
pub mod dual;
pub mod dual2;
pub mod error;
pub mod fock;
pub mod gradient;
pub mod hamiltonian;
pub mod hessian;
pub mod integrals;
pub mod optimizer;
pub mod linalg;
pub mod math;
pub mod overlap;
pub mod overlap_numeric;
pub mod params;
pub mod repulsion;
pub mod scf;
pub mod system;
pub mod topology;

pub use bcc::{am1_bcc_charges, BccResult};

#[cfg(feature = "python")]
pub mod python;

pub use error::{Am1Error, Result};
pub use linalg::Matrix;
pub use math::{Mat3, Vec3};
pub use gradient::{
    analytic_gradient, closed_form_gradient, electronic_gradient_fixed_density,
    energy_at_fixed_density, fixed_density_gradient, numerical_gradient, GradientResult,
};
pub use hessian::{analytic_hessian, numerical_hessian, vibrational_analysis, VibrationalModes};
pub use optimizer::{optimize, OptOptions, OptResult};
pub use params::{Am1Element, Am1Parameters};
pub use scf::{run_am1, Am1Calculator, Am1Options, Am1Result, ScfAccelerator, ScfReference};
pub use system::{symbol_to_z, z_to_symbol, Atom, Molecule};

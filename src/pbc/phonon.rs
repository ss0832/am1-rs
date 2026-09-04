// SPDX-License-Identifier: GPL-3.0-or-later

//! Phonons: real-space force constants `Φ(T)`, the dynamical matrix `D(q)`, and band structures.
//!
//! # Where `Φ(T)` comes from, and why it is a supercell
//!
//! The Γ-point Hessian ([`crate::analytic_hessian`] on a periodic [`Molecule`]) gives
//! `Σ_T Φ(0,T)` — the force constants summed over every lattice translation. That is exactly
//! what is needed at `q = 0` and useless anywhere else, because `D(q) = Σ_T Φ(0,T) e^{iq·T}`
//! needs the translations resolved, not summed.
//!
//! Two routes give the resolved `Φ(T)`:
//!
//! * **Supercell.** Compute the Γ Hessian of an `n₁ × n₂ × n₃` supercell and read `Φ_ab(T)` off
//!   its blocks: the force constant between atom `a` in the home cell and atom `b` in cell `T`
//!   *is* an element of that Hessian. Exact at every `q` commensurate with the supercell, which
//!   is the set of `q` the supercell can represent at all.
//! * **Density-functional perturbation theory.** Solve the response at each `q` directly. This
//!   needs a CPSCF coupling `k` and `k+q`, which is a different solver from the Γ one.
//!
//! This module takes the supercell route. It is exact where it applies, it reuses the Γ Hessian
//! that is already validated against finite differences, and it does not require a second
//! response solver. Its limit is the supercell size: `Φ(T)` is truncated at the supercell
//! boundary, so `q` between the commensurate points is an interpolation rather than a
//! calculation. [`ForceConstants::commensurate_q`] enumerates the points where it is exact.
//!
//! # LO–TO splitting
//!
//! In a polar material the dipole–dipole force constants have an `R⁻³` tail, so `Φ(T)` is **not**
//! short-ranged and no supercell captures it. The `q → 0` limit of `D(q)` is then direction
//! dependent, and a truncated Fourier sum structurally cannot be — however large the supercell.
//!
//! That piece is supplied analytically by [`ForceConstants::dynamical_matrix_with_lo_to`], from
//! the Born effective charges ([`crate::pbc::born_charges`]) and the electronic dielectric tensor
//! ([`crate::pbc::dielectric_tensor`]):
//!
//! ```text
//! D_NA(q)_{aα,bβ} = (4π/Ω) (q·Z*_a)_α (q·Z*_b)_β / (q·ε_∞·q) / √(m_a m_b)
//! ```
//!
//! [`ForceConstants::frequencies`] leaves it out and is the right function for a non-polar
//! system; [`ForceConstants::frequencies_with_lo_to`] includes it and needs the direction along
//! which `q → 0` is being taken, because at exactly `q = 0` the term is undefined — which is the
//! physics, not a limitation, and is refused rather than guessed.
//!
//! # Three dimensions only
//!
//! That expression is the three-dimensional one. `4π/(Ω q·ε∞·q)` is the Fourier transform of the
//! dipole–dipole interaction in 3D and `Ω` is a **volume**; in two dimensions the kernel is
//! `2π/(A q)`, and in one the non-analytic part vanishes as `q² ln q`, so a genuinely
//! 1D-periodic chain has **no** LO–TO splitting as `q → 0`.
//!
//! Version 0.2.0 did not enforce this. It applied the formula to chains with `Ω` taken from
//! [`crate::lattice::Lattice::measure`], which returns a *length* for a chain and an *area* for a
//! slab, and the "127 cm⁻¹ shift on a polar water chain" recorded here in that release was an
//! artifact of the resulting dimensional mismatch rather than a physical splitting. Since 0.2.1
//! a cell that is not fully periodic is refused. The low-dimensional non-analytic terms are not
//! implemented; see `docs/pbc.md`.
//!
//! Measured on a polar molecular crystal (one water per 4.5 Å cubic cell): the added term is its
//! closed form to `2 × 10⁻¹⁵`, it raises eigenvalues by up to `3 × 10⁻³` eV/(Å²·amu) and lowers
//! none, and the `q → 0` limit differs by 1.04 cm⁻¹ between two approach directions. On a
//! homonuclear crystal, where inversion symmetry and the acoustic sum rule force `Z* = 0`
//! (measured: `4 × 10⁻¹⁴` e), it is identically zero.
//!
//! The remaining caveat is `ε_∞` itself, which comes from a clamped-ion field response rather
//! than a Berry-phase polarization; see [`crate::pbc::dielectric_tensor`].

use std::collections::HashMap;

use crate::basis::Basis;
use crate::data_tables::MASS;
use crate::error::{Am1Error, Result};
use crate::lattice::{ImageOffset, Lattice};
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::params::Am1Parameters;
use crate::pbc::complex::{hermitian_eigen, CMatrix};
use crate::pbc::kpoints::KPoint;
use crate::scf::Am1Options;
use crate::system::{Atom, Molecule};

/// `sqrt(eV / (Å²·amu))` → cm⁻¹, the same conversion the molecular vibrational analysis uses.
pub const SQRT_EV_PER_ANG2_AMU_TO_CM: f64 = crate::hessian::SQRT_EV_PER_ANG2_AMU_TO_CM;

/// Real-space force constants of a periodic system, resolved by lattice translation.
#[derive(Clone, Debug)]
pub struct ForceConstants {
    /// `Φ_ab(T)`, a `3·nat × 3·nat` block per translation, in eV/Bohr².
    ///
    /// Row `3a + i` is atom `a` of the home cell along axis `i`; column `3b + j` is atom `b` of
    /// the cell displaced by `T`.
    pub blocks: HashMap<ImageOffset, Matrix>,
    /// Atomic masses of the primitive cell, amu.
    pub masses: Vec<f64>,
    /// The primitive lattice.
    pub lattice: Lattice,
    /// The supercell the constants were extracted from.
    pub supercell: [usize; 3],
    /// Atoms in the primitive cell.
    pub nat: usize,
}

/// Where a supercell atom came from: which primitive atom, in which image cell.
fn supercell_atom_index(cell: usize, prim: usize, nat: usize) -> usize {
    cell * nat + prim
}

/// Enumerate the supercell's image cells in a fixed order, with their translations.
fn supercell_cells(supercell: [usize; 3]) -> Vec<(usize, [i32; 3])> {
    let mut out = Vec::with_capacity(supercell[0] * supercell[1] * supercell[2]);
    let mut index = 0;
    for i in 0..supercell[0] {
        for j in 0..supercell[1] {
            for k in 0..supercell[2] {
                out.push((index, [i as i32, j as i32, k as i32]));
                index += 1;
            }
        }
    }
    out
}

/// Build the `n₁ × n₂ × n₃` supercell of `primitive`, in the atom order this module assumes.
pub fn build_supercell(primitive: &Molecule, supercell: [usize; 3]) -> Result<Molecule> {
    let cell = primitive
        .cell
        .ok_or_else(|| Am1Error::InvalidInput("a supercell needs a primitive cell".into()))?;
    for (axis, &n) in supercell.iter().enumerate() {
        if n == 0 {
            return Err(Am1Error::InvalidInput(format!(
                "supercell repeat along axis {axis} must be at least 1"
            )));
        }
        if n > 1 && !cell.periodic[axis] {
            return Err(Am1Error::InvalidInput(format!(
                "cannot repeat {n} times along axis {axis}: that direction is not periodic"
            )));
        }
    }

    let nat = primitive.atoms.len();
    let mut atoms = Vec::with_capacity(nat * supercell[0] * supercell[1] * supercell[2]);
    for (_, t) in supercell_cells(supercell) {
        let shift = cell.translation(ImageOffset { n: t });
        for atom in &primitive.atoms {
            atoms.push(Atom {
                z: atom.z,
                position: atom.position + shift,
            });
        }
    }

    let scaled = Lattice::from_vectors(
        cell.cell.col[0] * supercell[0] as f64,
        cell.cell.col[1] * supercell[1] as f64,
        cell.cell.col[2] * supercell[2] as f64,
        cell.periodic,
    )?;
    Ok(Molecule {
        atoms,
        charge: primitive.charge * (supercell[0] * supercell[1] * supercell[2]) as f64,
        multiplicity: primitive.multiplicity,
        cell: Some(scaled),
    })
}

impl ForceConstants {
    /// Force constants from the Γ Hessian of a supercell.
    ///
    /// `Φ_ab(T)` is read directly off that Hessian: the force constant coupling atom `a` of the
    /// home cell to atom `b` of the cell at `T`. Translational invariance means only the home
    /// cell's rows are needed.
    pub fn from_supercell(
        primitive: &Molecule,
        params: &Am1Parameters,
        options: &Am1Options,
        supercell: [usize; 3],
    ) -> Result<Self> {
        let cell = primitive
            .cell
            .ok_or_else(|| Am1Error::InvalidInput("phonons need a periodic cell".into()))?;
        let nat = primitive.atoms.len();
        let big = build_supercell(primitive, supercell)?;
        let hessian = crate::hessian::analytic_hessian(&big, params, options, 1.0e-3)?;

        let mut blocks: HashMap<ImageOffset, Matrix> = HashMap::new();
        for (cell_index, t) in supercell_cells(supercell) {
            let mut block = Matrix::zeros(3 * nat, 3 * nat);
            for a in 0..nat {
                // Home cell is index 0 by construction of `supercell_cells`.
                let row_atom = supercell_atom_index(0, a, nat);
                for b in 0..nat {
                    let col_atom = supercell_atom_index(cell_index, b, nat);
                    for i in 0..3 {
                        for j in 0..3 {
                            block[(3 * a + i, 3 * b + j)] =
                                hessian[(3 * row_atom + i, 3 * col_atom + j)];
                        }
                    }
                }
            }
            blocks.insert(ImageOffset { n: t }, block);
        }

        let masses = primitive.atoms.iter().map(|a| MASS[a.z as usize]).collect();

        Ok(Self {
            blocks,
            masses,
            lattice: cell,
            supercell,
            nat,
        })
    }

    /// The force-constant blocks in a **fixed** order, sorted by translation.
    ///
    /// # Why this exists
    ///
    /// [`Self::blocks`] is a `HashMap`, and every float sum over it — the Bloch sum below, and
    /// both acoustic-sum-rule passes — is therefore summed in whatever order that map iterates.
    /// Rust's `HashMap` seeds each instance from a thread-local counter, so **two maps built from
    /// the same insertions in the same process iterate differently**, and floating-point addition
    /// is not associative. The result was a phonon spectrum that changed between identical calls.
    ///
    /// Measured before the fix: five identical `lo_to_frequencies` calls in one process, on a
    /// water crystal in a 4.5 Å cube, agreed on four of them and differed by **1798 cm⁻¹** on the
    /// fifth — one O–H stretch collapsing to a near-zero mode. The SCF underneath was bit-identical
    /// every time (same energy, same 115 iterations), which is what located it here. The system is
    /// ill-conditioned enough for a last-bit difference to reorder near-degenerate eigenvectors,
    /// but the defect is the order-dependence, not the conditioning: a physical result must not
    /// depend on a hash seed.
    ///
    /// `tests/pbc_phonon.rs` asserts bit-identical repeats.
    fn ordered_blocks(&self) -> Vec<(&ImageOffset, &Matrix)> {
        let mut out: Vec<(&ImageOffset, &Matrix)> = self.blocks.iter().collect();
        out.sort_unstable_by_key(|(offset, _)| offset.n);
        out
    }

    /// The dynamical matrix `D(q) = Σ_T Φ(0,T) e^{iq·T} / √(m_a m_b)`, in eV/(Å²·amu).
    ///
    /// The phase uses **fractional** coordinates, `q·T = 2π(f₁n₁ + f₂n₂ + f₃n₃)`, so the
    /// Cartesian reciprocal vectors never enter — the same convention as the k-point sampling.
    pub fn dynamical_matrix(&self, q: KPoint) -> CMatrix {
        // eV/Bohr² → eV/Å², matching the molecular vibrational analysis.
        let a0_sq = crate::constants::ANGSTROM_TO_BOHR * crate::constants::ANGSTROM_TO_BOHR;
        let mut d = CMatrix::zeros(3 * self.nat);
        for (offset, block) in self.ordered_blocks() {
            let (cos, sin) = q.phase(*offset);
            for a in 0..self.nat {
                for b in 0..self.nat {
                    let inv_mass = 1.0 / (self.masses[a] * self.masses[b]).sqrt();
                    for i in 0..3 {
                        for j in 0..3 {
                            let v = block[(3 * a + i, 3 * b + j)] * a0_sq * inv_mass;
                            d.add(3 * a + i, 3 * b + j, v * cos, v * sin);
                        }
                    }
                }
            }
        }
        // `Φ(T)` truncated at the supercell boundary is not exactly symmetric under `T → −T`,
        // so symmetrize rather than let a tiny anti-Hermitian part produce complex frequencies.
        d.hermitianize();
        d
    }

    /// [`Self::dynamical_matrix`] plus the **non-analytic** term that produces LO–TO splitting.
    ///
    /// # Why it cannot come from `Φ(T)`
    ///
    /// In a polar material the dipole–dipole force constants decay as `R⁻³`, so `Φ(T)` is not
    /// short-ranged and no finite supercell captures it. Fourier-transforming a truncated `Φ(T)`
    /// therefore gets `q → 0` wrong no matter how large the supercell is: the limit is
    /// **direction dependent**, and a truncated sum has no way to be.
    ///
    /// The missing piece is added analytically:
    ///
    /// ```text
    /// D_NA(q)_{aα,bβ} = (4π/Ω) · (q·Z*_a)_α (q·Z*_b)_β / (q·ε_∞·q) / √(m_a m_b)
    /// ```
    ///
    /// with `q` the **Cartesian** phonon wavevector. It depends on `q` only through its
    /// direction, which is what makes the `q → 0` limit direction dependent and the longitudinal
    /// branch stiffer than the transverse ones.
    ///
    /// `direction` is that Cartesian direction; it need not be normalized, and it must not be
    /// zero — at exactly `q = 0` the term is undefined, which is the physics rather than a
    /// limitation. Pass the direction along which the limit is being taken.
    ///
    /// `born` comes from [`crate::pbc::born_charges`] and `epsilon` from
    /// [`crate::pbc::dielectric_tensor`]; the caveats on the latter apply here too.
    pub fn dynamical_matrix_with_lo_to(
        &self,
        q: KPoint,
        direction: Vec3,
        born: &[[[f64; 3]; 3]],
        epsilon: &[[f64; 3]; 3],
        measure: f64,
    ) -> Result<CMatrix> {
        // Three-dimensional only, and this is a correction rather than a restriction: the form
        // above *is* the 3D one. The `4π/(Ω q·ε·q)` kernel is the Fourier transform of the
        // dipole–dipole interaction in three dimensions; in two it is `2π/(A q)` and in one the
        // non-analytic part vanishes as `q² ln q`, so a genuinely 1D-periodic chain has **no**
        // LO–TO splitting at `q → 0`.
        //
        // Until 0.2.1 this was applied to chains, with `Ω` silently being `Lattice::measure` — a
        // *length* for a chain and an *area* for a slab. The result was dimensionally not a
        // dielectric response, and the splitting it produced was an artifact. See docs/pbc.md;
        // the correct low-dimensional terms are not implemented.
        if !self.lattice.is_fully_periodic() {
            return Err(Am1Error::InvalidInput(
                "the LO-TO non-analytic term implemented here is three-dimensional: its \
                 4π/(Ω q·ε∞·q) kernel is the 3D dipole-dipole Fourier transform, and Ω must be a \
                 volume. A 1D chain has no LO-TO splitting as q → 0 (the term vanishes as \
                 q² ln q) and a slab needs 2π/(A q); neither is implemented. Use \
                 `dynamical_matrix` for a chain or a slab."
                    .into(),
            ));
        }
        let mut d = self.dynamical_matrix(q);
        let n = direction.norm();
        if n < 1.0e-12 {
            return Err(Am1Error::InvalidInput(
                "the non-analytic term needs a direction to take the q → 0 limit along; it is \
                 undefined at exactly q = 0"
                    .into(),
            ));
        }
        if born.len() != self.nat {
            return Err(Am1Error::InvalidInput(format!(
                "expected {} Born-charge tensors, got {}",
                self.nat,
                born.len()
            )));
        }
        let qhat = [direction.x / n, direction.y / n, direction.z / n];

        // q·ε_∞·q
        let mut denom = 0.0;
        for a in 0..3 {
            for b in 0..3 {
                denom += qhat[a] * epsilon[a][b] * qhat[b];
            }
        }
        if denom.abs() < 1.0e-12 {
            return Err(Am1Error::InvalidInput(
                "q·ε_∞·q vanishes, so the non-analytic term is singular".into(),
            ));
        }

        // (q·Z*_a)_α = Σ_γ q_γ Z*_{a,γα}
        let qz: Vec<[f64; 3]> = born
            .iter()
            .map(|z| {
                let mut v = [0.0_f64; 3];
                for (alpha, vv) in v.iter_mut().enumerate() {
                    for (gamma, qg) in qhat.iter().enumerate() {
                        *vv += qg * z[gamma][alpha];
                    }
                }
                v
            })
            .collect();

        let a0_sq = crate::constants::ANGSTROM_TO_BOHR * crate::constants::ANGSTROM_TO_BOHR;
        let prefactor = 4.0 * std::f64::consts::PI / (measure * denom);
        for a in 0..self.nat {
            for b in 0..self.nat {
                let inv_mass = 1.0 / (self.masses[a] * self.masses[b]).sqrt();
                for i in 0..3 {
                    for j in 0..3 {
                        let v = prefactor * qz[a][i] * qz[b][j] * a0_sq * inv_mass;
                        d.add(3 * a + i, 3 * b + j, v, 0.0);
                    }
                }
            }
        }
        d.hermitianize();
        Ok(d)
    }

    /// Harmonic frequencies at `q` including LO–TO splitting. See
    /// [`Self::dynamical_matrix_with_lo_to`].
    pub fn frequencies_with_lo_to(
        &self,
        q: KPoint,
        direction: Vec3,
        born: &[[[f64; 3]; 3]],
        epsilon: &[[f64; 3]; 3],
        measure: f64,
    ) -> Result<Vec<f64>> {
        let d = self.dynamical_matrix_with_lo_to(q, direction, born, epsilon, measure)?;
        let eigen = hermitian_eigen(&d)?;
        Ok(eigen
            .values
            .iter()
            .map(|&lambda| {
                if lambda >= 0.0 {
                    SQRT_EV_PER_ANG2_AMU_TO_CM * lambda.sqrt()
                } else {
                    -SQRT_EV_PER_ANG2_AMU_TO_CM * (-lambda).sqrt()
                }
            })
            .collect())
    }

    /// Harmonic frequencies at `q`, cm⁻¹, ascending. Negative denotes an imaginary mode.
    pub fn frequencies(&self, q: KPoint) -> Result<Vec<f64>> {
        let eigen = hermitian_eigen(&self.dynamical_matrix(q))?;
        Ok(eigen
            .values
            .iter()
            .map(|&lambda| {
                if lambda >= 0.0 {
                    SQRT_EV_PER_ANG2_AMU_TO_CM * lambda.sqrt()
                } else {
                    -SQRT_EV_PER_ANG2_AMU_TO_CM * (-lambda).sqrt()
                }
            })
            .collect())
    }

    /// Frequencies along a path of `q` points, one row per point.
    pub fn band_structure(&self, path: &[KPoint]) -> Result<Vec<Vec<f64>>> {
        path.iter().map(|q| self.frequencies(*q)).collect()
    }

    /// The `q` points this supercell represents exactly.
    ///
    /// Everything else is an interpolation of a truncated `Φ(T)`, which is worth being able to
    /// distinguish from a calculation.
    pub fn commensurate_q(&self) -> Vec<KPoint> {
        let mut out = Vec::new();
        let n = self.supercell;
        let total = (n[0] * n[1] * n[2]) as f64;
        for i in 0..n[0] {
            for j in 0..n[1] {
                for k in 0..n[2] {
                    out.push(KPoint {
                        fractional: [
                            i as f64 / n[0] as f64,
                            j as f64 / n[1] as f64,
                            k as f64 / n[2] as f64,
                        ],
                        weight: 1.0 / total,
                    });
                }
            }
        }
        out
    }

    /// Largest violation of the acoustic sum rule, `Σ_T Σ_b Φ_ab(T)`, in eV/Bohr².
    ///
    /// Translating every atom by the same vector cannot change the energy, so each of those sums
    /// must vanish. What it actually measures here is the truncation of `Φ(T)` at the supercell
    /// boundary, since the Γ Hessian it is built from satisfies the rule to roundoff.
    pub fn acoustic_sum_rule_error(&self) -> f64 {
        let ordered = self.ordered_blocks(); // fixed summation order; see `ordered_blocks`
        let mut worst = 0.0_f64;
        for a in 0..self.nat {
            for i in 0..3 {
                for j in 0..3 {
                    let mut sum = 0.0;
                    for (_, block) in &ordered {
                        for b in 0..self.nat {
                            sum += block[(3 * a + i, 3 * b + j)];
                        }
                    }
                    worst = worst.max(sum.abs());
                }
            }
        }
        worst
    }

    /// Impose the acoustic sum rule by correcting the on-site block.
    ///
    /// `Φ_aa(0) ← Φ_aa(0) − Σ_{T,b} Φ_ab(T)`, which is the standard fix: the self-term is the one
    /// least determined by the calculation, and forcing the sum through it guarantees three
    /// modes go to exactly zero at `q = 0` rather than to whatever the truncation left behind.
    ///
    /// This is a **correction, not a refinement** — it moves error into the on-site block rather
    /// than removing it. [`Self::acoustic_sum_rule_error`] before calling this is the honest
    /// measure of how much was wrong.
    pub fn enforce_acoustic_sum_rule(&mut self) {
        let origin = ImageOffset::origin();
        let mut corrections = vec![[[0.0_f64; 3]; 3]; self.nat];
        {
            // Fixed summation order; see `ordered_blocks`. This one matters most of the three:
            // the correction is *subtracted* from the on-site block, so an order-dependent value
            // here changes `Φ` itself and every `D(q)` built from it afterwards.
            let ordered = self.ordered_blocks();
            for (a, correction) in corrections.iter_mut().enumerate() {
                for i in 0..3 {
                    for j in 0..3 {
                        let mut sum = 0.0;
                        for (_, block) in &ordered {
                            for b in 0..self.nat {
                                sum += block[(3 * a + i, 3 * b + j)];
                            }
                        }
                        correction[i][j] = sum;
                    }
                }
            }
        }
        let block = self
            .blocks
            .entry(origin)
            .or_insert_with(|| Matrix::zeros(3 * self.nat, 3 * self.nat));
        for (a, correction) in corrections.iter().enumerate() {
            for i in 0..3 {
                for j in 0..3 {
                    block[(3 * a + i, 3 * a + j)] -= correction[i][j];
                }
            }
        }
    }
}

/// A straight-line path of `q` points between the given corners, `points_per_segment` each.
pub fn q_path(corners: &[[f64; 3]], points_per_segment: usize) -> Vec<KPoint> {
    let mut out = Vec::new();
    for pair in corners.windows(2) {
        for step in 0..points_per_segment {
            let t = step as f64 / points_per_segment as f64;
            out.push(KPoint {
                fractional: [
                    pair[0][0] + t * (pair[1][0] - pair[0][0]),
                    pair[0][1] + t * (pair[1][1] - pair[0][1]),
                    pair[0][2] + t * (pair[1][2] - pair[0][2]),
                ],
                weight: 1.0,
            });
        }
    }
    if let Some(last) = corners.last() {
        out.push(KPoint {
            fractional: *last,
            weight: 1.0,
        });
    }
    out
}

/// Number of AOs a molecule would use, for sizing diagnostics.
pub fn basis_size(molecule: &Molecule, params: &Am1Parameters) -> Result<usize> {
    Ok(Basis::build(molecule, params)?.nao)
}

/// Unused placeholder to keep `Vec3` imported for the supercell shift arithmetic.
const _: Option<Vec3> = None;

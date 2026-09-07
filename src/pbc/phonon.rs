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

/// One `q` point's phonons: the frequencies and the vectors that go with them.
///
/// See [`ForceConstants::modes`] for why both vector conventions are returned rather than one.
#[derive(Clone, Debug)]
pub struct PhononModes {
    /// Harmonic frequencies (cm⁻¹), ascending. Negative denotes an imaginary mode.
    pub frequencies_cm: Vec<f64>,
    /// Mass-weighted polarization vectors `e(q)`, `3N × 3N`, **columns** are modes and are
    /// orthonormal. Column `k` matches `frequencies_cm[k]`.
    pub polarization: CMatrix,
    /// Cartesian displacements `e_a / √m_a`, same shape, columns are modes.
    ///
    /// Deliberately **not** renormalized: the normalization lives on `polarization`, and rescaling
    /// these would silently change what a displacement along a mode means. This is the matching
    /// convention to [`crate::hessian::VibrationalModes::cartesian_displacements`].
    pub displacements: CMatrix,
}

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

/// Every translation congruent to `t` modulo the supercell lattice that puts `delta = r_b − r_a`
/// at the shortest distance — the Wigner–Seitz images of the pair, plural when they tie.
///
/// Searching `−1, 0, +1` supercell periods along each axis is enough: `t` already lies inside one
/// supercell period, so the nearest congruent translation cannot be more than one period away.
/// A non-periodic axis, or one the supercell does not repeat along, contributes only `0` — there
/// is no other image to fold onto.
///
/// The tie tolerance is on the *squared* distance and relative to the cell, so it scales with the
/// lattice rather than being an absolute length that would be too tight for a large cell and too
/// loose for a small one.
fn minimum_images(
    lattice: &Lattice,
    supercell: [usize; 3],
    t: [i32; 3],
    delta: Vec3,
) -> Vec<ImageOffset> {
    let span = |axis: usize| -> i32 {
        if lattice.periodic[axis] {
            1
        } else {
            0
        }
    };
    let mut best = f64::INFINITY;
    let mut candidates: Vec<(ImageOffset, f64)> = Vec::new();
    for i in -span(0)..=span(0) {
        for j in -span(1)..=span(1) {
            for k in -span(2)..=span(2) {
                let n = [
                    t[0] + i * supercell[0] as i32,
                    t[1] + j * supercell[1] as i32,
                    t[2] + k * supercell[2] as i32,
                ];
                let offset = ImageOffset { n };
                let d2 = (delta + lattice.translation(offset)).norm2();
                if d2 < best {
                    best = d2;
                }
                candidates.push((offset, d2));
            }
        }
    }
    // Relative on the scale of a lattice vector: two images are "the same distance" when the
    // difference is below what the geometry can resolve, not below a fixed number of Bohr.
    let scale = lattice
        .cell
        .col
        .iter()
        .fold(1.0_f64, |m, v| m.max(v.norm2()));
    let tolerance = 1.0e-8 * scale;
    let mut out: Vec<ImageOffset> = candidates
        .into_iter()
        .filter(|(_, d2)| *d2 <= best + tolerance)
        .map(|(offset, _)| offset)
        .collect();
    // A fixed order so that the `HashMap` insertion sequence — and therefore nothing at all — is
    // the same on every run. See `ordered_blocks`.
    out.sort_unstable_by_key(|offset| offset.n);
    out.dedup_by_key(|offset| offset.n);
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
    /// `Φ_ab(T)` is the force constant coupling atom `a` of the home cell to atom `b` of the cell
    /// at `T`. Translational invariance means only the home cell's rows are needed.
    ///
    /// # The translation the Hessian element belongs to is not the one it is indexed by
    ///
    /// A supercell's Γ Hessian is an **aliased** sum,
    ///
    /// ```text
    /// H_{(0,a),(t,b)} = Σ_N Φ_ab(T_t + N)
    /// ```
    ///
    /// over the supercell lattice `N`, because the supercell is itself periodic. Reading the
    /// element off as `Φ_ab(T_t)` — which is what releases through 0.2.2 did — assigns it to the
    /// translation the *atom list* happened to be built with, and `supercell_cells` builds them
    /// as `0, 1, … n−1` along each axis. For `n = 3` that labels the neighbour on one side `+2`
    /// when it is physically at `−1`.
    ///
    /// At a **commensurate** `q = m/n` the two labels give the same Bloch phase
    /// (`e^{2πi m(n−1)/n} = e^{−2πi m/n}`), which is why every test that only ever asked for the
    /// `q` points the supercell represents exactly could not see this. Anywhere else they do not:
    /// on a 3× chain at the zone boundary `q = ½` the phase is `+1` under the old labelling and
    /// `−1` under the right one. Every interpolated band — that is, every band-structure plot —
    /// was wrong between the commensurate points.
    ///
    /// The fix is the standard one: assign each atom-pair block to the **minimum image** of
    /// `r_b + T − r_a` over `T ≡ T_t` (mod the supercell lattice), splitting equally when several
    /// images tie. Splitting matters as much as folding: picking one of a tied pair by index
    /// order would break the point-group symmetry the tie exists because of, and the two halves
    /// at `±T` are what make `Φ(T) = Φ(−T)ᵀ` hold — which is the property `dynamical_matrix`
    /// used to have to repair with `hermitianize`.
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
            for a in 0..nat {
                // Home cell is index 0 by construction of `supercell_cells`.
                let row_atom = supercell_atom_index(0, a, nat);
                for b in 0..nat {
                    let col_atom = supercell_atom_index(cell_index, b, nat);
                    let images = minimum_images(
                        &cell,
                        supercell,
                        t,
                        primitive.atoms[b].position - primitive.atoms[a].position,
                    );
                    let share = 1.0 / images.len() as f64;
                    for image in images {
                        let block = blocks
                            .entry(image)
                            .or_insert_with(|| Matrix::zeros(3 * nat, 3 * nat));
                        for i in 0..3 {
                            for j in 0..3 {
                                block[(3 * a + i, 3 * b + j)] +=
                                    share * hessian[(3 * row_atom + i, 3 * col_atom + j)];
                            }
                        }
                    }
                }
            }
        }

        // `atomic_mass`, not `MASS[z]`: a gap in the table is a `1/0` in every dynamical matrix
        // built from these, and it should say so here rather than at the eigensolver.
        let masses = primitive
            .atoms
            .iter()
            .map(|a| crate::data_tables::atomic_mass(a.z))
            .collect::<Result<Vec<f64>>>()?;

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
        // `Φ(T)` truncated at the supercell boundary is not *exactly* symmetric under `T → −T`,
        // so symmetrize rather than let a tiny anti-Hermitian part produce complex frequencies.
        // Since the minimum-image folding in `from_supercell` the residue is small — measured at
        // 6e-11 against a 9.8 eV/Bohr² scale on an H₂ chain, where filing the blocks by index
        // order left 7e-3 — but it is not zero, and this costs nothing.
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
        Ok(self.modes(q)?.frequencies_cm)
    }

    /// The phonons at `q`: frequencies **and** the vectors that say what moves.
    ///
    /// # Why this exists
    ///
    /// [`Self::frequencies`] returns a list of numbers, and a list of numbers cannot answer the
    /// question a phonon calculation is usually run to answer. Tracing an imaginary mode on
    /// trimethylaluminium needed its eigenvector and there was no way to get one from here at all
    /// — the direction had to be borrowed from the *molecular* Hessian at the same geometry, which
    /// works only because the two agree in a large cell and is not a general answer.
    ///
    /// # Two vectors, and they are not the same
    ///
    /// The dynamical matrix is mass-weighted, so its eigenvectors are the **polarization vectors**
    /// `e(q)` — orthonormal, `eᵀe = 1`, and the right thing for anything that sums over modes
    /// (structure factors, mode Grüneisen parameters, electron–phonon matrix elements). What an
    /// atom actually *does* is `u_a = e_a / √m_a`, which is not normalized and is the right thing
    /// for displacing a structure along a mode. Both are returned rather than one, because
    /// choosing for the caller is how a factor of `√m` gets silently applied twice.
    ///
    /// Complex, unlike the molecular case: at `q ≠ 0` the pattern carries a Bloch phase between
    /// cells, and the imaginary part is that phase rather than a numerical residue.
    pub fn modes(&self, q: KPoint) -> Result<PhononModes> {
        self.modes_of(&self.dynamical_matrix(q))
    }

    /// [`Self::modes`] on a dynamical matrix already built — with the LO–TO term, say.
    pub fn modes_of(&self, d: &CMatrix) -> Result<PhononModes> {
        let eigen = hermitian_eigen(d)?;
        let n = 3 * self.nat;
        let frequencies_cm = eigen
            .values
            .iter()
            .map(|&lambda| {
                if lambda >= 0.0 {
                    SQRT_EV_PER_ANG2_AMU_TO_CM * lambda.sqrt()
                } else {
                    -SQRT_EV_PER_ANG2_AMU_TO_CM * (-lambda).sqrt()
                }
            })
            .collect();
        let polarization = CMatrix {
            n,
            re: eigen.vectors_re,
            im: eigen.vectors_im,
        };
        let mut displacements = polarization.clone();
        for row in 0..n {
            // Row `3a + i` belongs to atom `a`, so the mass is the atom's, not the mode's.
            let inv_sqrt_m = 1.0 / self.masses[row / 3].sqrt();
            for col in 0..n {
                displacements.re[(row, col)] *= inv_sqrt_m;
                displacements.im[(row, col)] *= inv_sqrt_m;
            }
        }
        Ok(PhononModes {
            frequencies_cm,
            polarization,
            displacements,
        })
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

    /// `max |Φ_ab(T) − Φ_ba(−T)ᵀ|`, in eV/Bohr².
    ///
    /// The second derivative of the energy does not care which of two atoms is differentiated
    /// first, so this is exactly zero for the true force constants. What it measures here is the
    /// residue of the real-space and exchange cutoffs: the supercell's Γ Hessian is only
    /// translationally invariant to the extent that those truncations are, and
    /// `Φ_ab(T) = H[(0,a),(T,b)]` versus `Φ_ba(−T) = H[(0,b),(−T,a)]` reads two *different*
    /// elements of it.
    ///
    /// Measured on graphene in a 2×2 supercell: **8.2e-4** against a `|Φ|` of 18.7 eV/Bohr², and
    /// that is the number that used to put diamond's acoustic modes at −275 cm⁻¹. See
    /// [`Self::enforce_acoustic_sum_rule`].
    pub fn transpose_asymmetry(&self) -> f64 {
        let ndof = 3 * self.nat;
        let mut worst = 0.0_f64;
        for (offset, block) in self.ordered_blocks() {
            let mirror = self.blocks.get(&offset.negated());
            for i in 0..ndof {
                for j in 0..ndof {
                    let other = mirror.map_or(0.0, |m| m[(j, i)]);
                    worst = worst.max((block[(i, j)] - other).abs());
                }
            }
        }
        worst
    }

    /// Impose `Φ_ab(T) = Φ_ba(−T)ᵀ` by averaging each block with its mirror.
    ///
    /// Creates the mirror block when it is missing, which is why the translation set comes out
    /// symmetric under negation afterwards.
    pub fn symmetrize_transpose(&mut self) {
        let ndof = 3 * self.nat;
        let offsets: Vec<ImageOffset> = {
            let mut v: Vec<ImageOffset> = self.blocks.keys().copied().collect();
            // Fixed order; see `ordered_blocks`. Averaging is order-independent, but *creating*
            // the missing mirrors is not — a mirror created inside the loop would be averaged
            // again when its own turn came.
            v.sort_unstable_by_key(|o| o.n);
            v
        };
        for offset in &offsets {
            self.blocks
                .entry(offset.negated())
                .or_insert_with(|| Matrix::zeros(ndof, ndof));
        }
        let mut averaged: HashMap<ImageOffset, Matrix> = HashMap::new();
        for offset in self.blocks.keys().copied().collect::<Vec<_>>() {
            let block = &self.blocks[&offset];
            let mirror = &self.blocks[&offset.negated()];
            let mut out = Matrix::zeros(ndof, ndof);
            for i in 0..ndof {
                for j in 0..ndof {
                    out[(i, j)] = 0.5 * (block[(i, j)] + mirror[(j, i)]);
                }
            }
            averaged.insert(offset, out);
        }
        self.blocks = averaged;
    }

    /// Impose the acoustic sum rule, and the transpose symmetry it has to share the matrix with.
    ///
    /// # Why this is two rules and not one
    ///
    /// The sum rule `Σ_{T,b} Φ_ab(T) = 0` is a statement about the **rows** of `Φ`. Imposing it
    /// through the on-site block — `Φ_aa(0) ← Φ_aa(0) − Σ_{T,b} Φ_ab(T)`, the standard fix, since
    /// the self-term is the one least determined by the calculation — makes the row sums vanish
    /// exactly.
    ///
    /// That is not enough to put three modes at zero, and through 0.2.2 it did not.
    /// [`Self::dynamical_matrix`] symmetrizes `D(q)` before diagonalizing it, so the spectrum
    /// sees the *average* of the row and column sums, and the column sums are only zero if
    /// `Φ_ab(T) = Φ_ba(−T)ᵀ`. The supercell Hessian satisfies that only as well as its real-space
    /// and exchange cutoffs do. Measured on graphene: row sums `1.8e-9`, column sums `8.2e-4`.
    /// **Diamond in its fcc primitive cell came out with acoustic modes at −275, −275 and −65
    /// cm⁻¹ while `acoustic_sum_rule_error()` read `1.2e-15`** — the sum rule genuinely held, and
    /// it was the wrong sum rule to hold alone.
    ///
    /// So the two are imposed together, by **alternating projection**: symmetrize, impose the row
    /// sums, repeat.
    ///
    /// # Where the alternation stops, and why that is the right place
    ///
    /// It does not reach zero, and the reason is worth stating rather than iterating against. The
    /// row-sum step subtracts `R_a = Σ_{T,b} Φ_ab(T)` from `Φ_aa(0)`, and `Φ_aa(0)` must be
    /// symmetric for the transpose symmetry to hold. Splitting `R_a` into its symmetric and
    /// antisymmetric parts, the symmetrization removes `sym(R_a)` and puts the row sum back at
    /// `antisym(R_a)`; the next row-sum step removes that and breaks the symmetry by the same
    /// amount. The alternation therefore has a **fixed point** at `|antisym(R_a)|`, reached in a
    /// few passes, and more passes do nothing. Measured on graphene: `4.1e-4 → 6.5e-6` and then
    /// stationary.
    ///
    /// `antisym(R_a)` is a **rotational** sum-rule residue — the translational rule constrains
    /// only the symmetric part — and it is left behind by the real-space and exchange cutoffs
    /// rather than by anything this routine could fix. On graphene it is `6.5e-6` eV/Bohr²
    /// against a `|Φ|` of 18.7, which reaches the acoustic branch as **0.7 cm⁻¹**. The three
    /// acoustic frequencies come out below the precision the output is printed to, which is the
    /// property that matters; driving the matrix element itself to zero would mean moving that
    /// error somewhere less visible, not removing it.
    ///
    /// Each pass is `O(#T · nat² · 9)` floating-point operations on a matrix the Hessian already
    /// paid `O(nat³)` to produce, so the iteration is free in every practical sense.
    ///
    /// This is a **correction, not a refinement**: it moves truncation error into the on-site
    /// block rather than removing it. [`Self::acoustic_sum_rule_error`] and
    /// [`Self::transpose_asymmetry`] *before* calling this are the honest measure of how much was
    /// wrong.
    pub fn enforce_acoustic_sum_rule(&mut self) {
        let scale = self
            .blocks
            .values()
            .flat_map(|m| m.as_slice().iter())
            .fold(0.0_f64, |x, v| x.max(v.abs()))
            .max(1.0);
        let target = 1.0e-13 * scale;
        let mut previous = f64::INFINITY;
        for _ in 0..16 {
            self.symmetrize_transpose();
            self.enforce_row_sum_rule();
            let residual = self.transpose_asymmetry();
            // Stop at the fixed point as well as at roundoff: once a pass stops improving things
            // by more than a tenth, the remainder is the rotational residue described above and
            // further passes only move it between the two constraints.
            if residual <= target || residual > 0.9 * previous {
                break;
            }
            previous = residual;
        }
        // The row sums go last: after the final symmetrization they and the column sums are the
        // same quantity, and `q = 0` is where they have to be zero.
        self.symmetrize_transpose();
        self.enforce_row_sum_rule();
    }

    /// The row-sum half of [`Self::enforce_acoustic_sum_rule`], on its own.
    fn enforce_row_sum_rule(&mut self) {
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

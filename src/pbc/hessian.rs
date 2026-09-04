// SPDX-License-Identifier: GPL-3.0-or-later

//! Analytic force constants at `q = 0` with **k-point sampling**.
//!
//! # What this adds over the Γ-only Hessian
//!
//! [`crate::hessian::analytic_hessian`] already handles a periodic cell: it runs the pair list
//! over lattice images and its CPHF is the molecular one, which is exactly right at Γ because
//! `P(0,T) = P(Γ)` for every translation there. That identity is also the Γ path's defect. A
//! real-space density matrix that does not decay makes NDDO's two-centre exchange — whose
//! integral falls off only as `1/R` — diverge over the image sum, and the Γ path has to taper it
//! away by hand. The taper is an approximation standing in for physics, and it sits inside the
//! second derivative as much as inside the energy.
//!
//! Sampling `k` removes it. `P(0,T) = Σ_k w_k e^{−ik·T} P(k)` decays for a gapped system because
//! the phases interfere, so the exchange converges on its own and the force constants are the
//! ones the Hamiltonian actually implies rather than the ones the taper leaves behind.
//!
//! # Structure
//!
//! Two terms, as always:
//!
//! **Skeleton**, the second derivative at fixed density. Identical in form to the molecular case
//! — one `3 × 3` block per image pair from second-order AD of that pair's energy — except that
//! each pair contracts against **its own** translation's density block `P(0,T)` rather than
//! against one global matrix. That single change is what makes this the k-point Hessian.
//!
//! **Response**, the CPHF. At `q = 0` the perturbation `∂/∂R` is lattice-periodic, so it does not
//! mix k points and each k gets its own equation:
//!
//! ```text
//! (ε_a(k) − ε_i(k)) U^x_{ai}(k) + [G(ΔP(U))](k)_{ai} = −G^x_{ai}(k)
//! ```
//!
//! The k points are still coupled, but only through the density: `ΔP(0,T)` is a weighted sum
//! over the whole mesh, and the response Fock built from it is then Bloch-summed back to every
//! k. So one application of the orbital Hessian is one real-space two-electron build plus one
//! Bloch sum and projection per k — the same cost structure as the molecular CPHF, times the
//! mesh.
//!
//! Because NDDO's AO basis is orthonormal, `S(k) = I`: there is no Pulay term and no `S⁽¹⁾`,
//! which is what keeps the equations above as simple as their molecular counterparts.

use rayon::prelude::*;

use crate::basis::Basis;
use crate::dual::Scalar;
use crate::dual2::Dual2;
use crate::error::{Am1Error, Result};
use crate::linalg::Matrix;
use crate::neighbors::NeighborList;
use crate::params::Am1Parameters;
use crate::pbc::complex::{hermitian_eigen, CMatrix};
use crate::pbc::kpoints::KPoint;
use crate::pbc::scf::{build_realspace_core, run_pbc_scf, PbcOptions, RealSpaceBlocks};
use crate::system::Molecule;

/// Convergence tolerance on the CPHF residual, per perturbation.
const CPHF_TOL: f64 = 1.0e-8;
/// Iteration cap for the k-point CPHF.
const CPHF_MAX_ITER: usize = 200;
/// Occupied–virtual pairs closer than this in energy have their response dropped.
///
/// The orbital-rotation denominator `ε_a − ε_i` vanishes for a degenerate occupied/virtual pair,
/// and the rotation between two degenerate orbitals is not determined by the equations — it does
/// not contribute to the density either, so dropping it is correct rather than merely expedient.
const DEGENERACY_FLOOR: f64 = 1.0e-10;

/// One converged k point's orbitals, in the form the response equations need.
struct KOrbitals {
    k: KPoint,
    energies: Vec<f64>,
    // The per-level occupations are *not* stored. They are computed in `solve_orbitals`, used
    // there to derive `occ`/`vir`, and dropped — a copy kept "for a future smeared-response
    // path" was dead weight that had to be silenced with `#[allow(dead_code)]`. When that path
    // arrives it can carry them then.
    /// Indices of levels treated as occupied and virtual for the response.
    occ: Vec<usize>,
    vir: Vec<usize>,
    /// The occupied and virtual coefficient columns, gathered into compact `nao × n` blocks.
    ///
    /// `occ`/`vir` are index lists, not ranges — smearing can leave a partially filled level in
    /// neither — so the products in `project_ov` and `response_density` need a contiguous copy.
    /// Gathering it *here* rather than there is the difference between doing it once per k point
    /// and doing it once per perturbation per CPHF iteration; done per call it cost more than
    /// the `O(nao⁴)` loop nest those products replaced.
    ///
    /// The full coefficient matrix is not kept: every consumer wants one of these two blocks.
    co_re: Matrix,
    co_im: Matrix,
    cv_re: Matrix,
    cv_im: Matrix,
}

/// Analytic force-constant matrix at `q = 0`, with the k-mesh from `options`.
///
/// Returns the `3N × 3N` Cartesian Hessian in eV/Bohr², symmetrized. Restricted (closed-shell)
/// only for now; an unrestricted request is rejected rather than silently answered with the
/// restricted equations.
pub fn pbc_hessian(
    molecule: &Molecule,
    params: &Am1Parameters,
    options: &PbcOptions,
) -> Result<Matrix> {
    let response = solve_phonon_response(molecule, params, options)?;
    let nat = molecule.atoms.len();
    let ndof = 3 * nat;

    // ---- 1) Skeleton second derivative, contracting each pair against its own P(0,T). ----
    let mut hess = skeleton(
        molecule,
        params,
        &response.basis,
        &response.neighbors,
        &response.scf.density,
        &response.exchange_channels()?,
        options,
    )?;

    // ---- 2) Orbital-relaxation term, from the CPHF solution. ----
    //
    // `H_relax[a][b] = Σ_σ Σ_k 2 f w_k Re(G^a(k)* · U^b(k))`, and
    // `Re(G* U) = G_re·U_re + G_im·U_im`. Stacking each k point's ov-blocks as the rows of an
    // `ndof × n_ov` matrix turns the whole sum into two matrix products per k point instead of an
    // `ndof² · n_ov` scalar nest that was not even parallelized.
    //
    // The 2 is the occupied-virtual / virtual-occupied pair; `f` is what one orbital holds. So a
    // restricted run has one channel at weight 4, and an unrestricted one has two channels at
    // weight 2 apiece — which is the same 4 when the shell is closed, and that identity is what
    // `tests/pbc_uhf_response.rs` checks by forcing UHF on a closed shell.
    let mut relax = Matrix::zeros(ndof, ndof);
    for channel in &response.channels {
        for (ki, orb) in channel.orbitals.iter().enumerate() {
            let n_ov = orb.vir.len() * orb.occ.len();
            if n_ov == 0 {
                continue;
            }
            let mut g_re = Matrix::zeros(ndof, n_ov);
            let mut g_im = Matrix::zeros(ndof, n_ov);
            let mut u_re = Matrix::zeros(ndof, n_ov);
            let mut u_im = Matrix::zeros(ndof, n_ov);
            for a in 0..ndof {
                let span = a * n_ov..(a + 1) * n_ov;
                g_re.as_mut_slice()[span.clone()].copy_from_slice(&channel.gov[ki][a].re);
                g_im.as_mut_slice()[span.clone()].copy_from_slice(&channel.gov[ki][a].im);
                u_re.as_mut_slice()[span.clone()].copy_from_slice(&channel.u[ki][a].re);
                u_im.as_mut_slice()[span].copy_from_slice(&channel.u[ki][a].im);
            }
            let w = 2.0 * response.fill * orb.k.weight;
            g_re.matmul_transpose_acc_seq(&u_re, &mut relax, w);
            g_im.matmul_transpose_acc_seq(&u_im, &mut relax, w);
        }
    }
    for a in 0..ndof {
        for b in 0..ndof {
            hess[(a, b)] += relax[(a, b)];
        }
    }

    // Symmetrize: the two terms are each symmetric only up to the CPHF residual.
    let mut sym = Matrix::zeros(ndof, ndof);
    for i in 0..ndof {
        for j in 0..ndof {
            sym[(i, j)] = 0.5 * (hess[(i, j)] + hess[(j, i)]);
        }
    }
    Ok(sym)
}

/// The skeleton (fixed-density) half of [`pbc_hessian`], on its own.
///
/// A testable seam, not an API: splitting the Hessian into its two halves is the only way to
/// tell which of them a finite-difference disagreement belongs to, and reconstructing the
/// skeleton from outside would mean writing the same expression a second time — which is
/// exactly the duplication that lets the two drift apart.
#[doc(hidden)]
pub fn pbc_hessian_skeleton(
    molecule: &Molecule,
    params: &Am1Parameters,
    options: &PbcOptions,
) -> Result<Matrix> {
    let scf = run_pbc_scf(molecule, params, options)?;
    let basis = Basis::build(molecule, params)?;
    let neighbors = NeighborList::build(molecule, options.realspace_cutoff);
    // The exchange channels for a **restricted** ground state. This seam exists to isolate the
    // skeleton half of the Hessian for a finite-difference comparison, and the response half is
    // what the open-shell work generalised; splitting an unrestricted skeleton out on its own has
    // no consumer, so it is refused rather than answered with the restricted expression.
    if scf.unrestricted {
        return Err(Am1Error::InvalidInput(
            "the skeleton seam is restricted-only; `pbc_hessian` handles open shells".into(),
        ));
    }
    let channels = [ExchangeChannel {
        blocks: &scf.density,
        origin: scf.density.origin()?,
        scale: 0.5,
    }];
    let raw = skeleton(
        molecule,
        params,
        &basis,
        &neighbors,
        &scf.density,
        &channels,
        options,
    )?;
    let n = raw.rows;
    let mut sym = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            sym[(i, j)] = 0.5 * (raw[(i, j)] + raw[(j, i)]);
        }
    }
    Ok(sym)
}

/// Everything the phonon response produces, so the Hessian and the Born charges share one solve.
///
/// The CPHF is the expensive part — one real-space two-electron build per iteration per
/// perturbation — and both quantities are contractions of the *same* `U`. Solving it twice would
/// be wasteful and, worse, would let the two drift apart.
pub struct PhononResponse {
    /// Converged periodic SCF.
    pub scf: crate::pbc::scf::PbcResult,
    basis: Basis,
    neighbors: NeighborList,
    translations: Vec<crate::lattice::ImageOffset>,
    /// One channel restricted, two (α then β) unrestricted.
    channels: Vec<ChannelResponse>,
    /// The ground-state density of each channel — what the exchange contracts against. Kept
    /// rather than rebuilt, because the skeleton needs exactly the split the response was solved
    /// with, and rebuilding it is where the two would drift apart.
    channel_densities: Vec<RealSpaceBlocks>,
    /// Electrons one orbital holds: `2` restricted, `1` per unrestricted channel. Every place
    /// that turns `U` back into a density or an energy needs it, so it is carried rather than
    /// re-derived from `channels.len()` at each of them.
    fill: f64,
}

/// One spin channel's orbitals, bare perturbation and CPHF solution.
///
/// Restricted, there is one of these and it stands for both spins. Unrestricted, there are two and
/// they are coupled — `G^σ(ΔP) = J(ΔP_tot) − K(ΔP_σ)` reads the *total* response density, so the
/// two cannot be solved independently.
struct ChannelResponse {
    orbitals: Vec<KOrbitals>,
    gov: Vec<Vec<COv>>,
    u: Vec<Vec<COv>>,
}

impl PhononResponse {
    /// The exchange channels, for [`skeleton`].
    fn exchange_channels(&self) -> Result<Vec<ExchangeChannel<'_>>> {
        let scale = crate::pbc::scf::exchange_scale_for(self.fill);
        self.channel_densities
            .iter()
            .map(|blocks| {
                Ok(ExchangeChannel {
                    blocks,
                    origin: blocks.origin()?,
                    scale,
                })
            })
            .collect()
    }

    /// `ΔP_tot(0,T)` for **one** perturbation, summed over the spin channels.
    ///
    /// One at a time because every consumer wants one at a time: the Born charges read each
    /// perturbation's origin block and move on, and the polarizability reads three of them. The
    /// version that returned a `Vec` held `ndof · n_T · nao²` doubles so that its callers could
    /// index it — see [`response_density_one`].
    fn response_density_total(&self, nao: usize, x: usize) -> RealSpaceBlocks {
        let mut out: Option<RealSpaceBlocks> = None;
        for channel in &self.channels {
            let part = response_density_one(
                &channel.u,
                &channel.orbitals,
                &self.translations,
                nao,
                self.fill,
                x,
            );
            match &mut out {
                None => out = Some(part),
                Some(acc) => acc.add_assign(&part),
            }
        }
        out.unwrap_or_else(|| RealSpaceBlocks::zeros(&self.translations, nao))
    }
}

/// Run the periodic SCF and solve the `q = 0` CPHF for every Cartesian perturbation.
/// The converged ground state, resolved into spin channels, with each channel's orbitals.
///
/// Everything both response builders need before they diverge — they differ only in the *bare
/// perturbation*, which is `∂F/∂R` for the phonon response and the dipole operator for the field
/// one. Sharing this is what gave the field response its open-shell path: the split, the
/// occupancies and the per-channel Fock are the same three things either way, and a second copy of
/// them is a second place for the exchange weight or the occupancy threshold to be wrong.
struct ChannelGround {
    scf: crate::pbc::scf::PbcResult,
    basis: Basis,
    neighbors: NeighborList,
    translations: Vec<crate::lattice::ImageOffset>,
    core: RealSpaceBlocks,
    pairs: crate::pbc::scf::PeriodicPairs,
    /// The density each channel's exchange contracts against: one entry restricted, `P^α` and
    /// `P^β` unrestricted.
    densities: Vec<RealSpaceBlocks>,
    /// Each channel's orbitals at every k.
    orbitals: Vec<Vec<KOrbitals>>,
    /// Electrons one orbital holds: `2` restricted, `1` per unrestricted channel.
    fill: f64,
}

impl ChannelGround {
    /// The exchange weight for one channel: `0.5` when it stands for both spins, `1.0` otherwise.
    fn scale(&self) -> f64 {
        crate::pbc::scf::exchange_scale_for(self.fill)
    }

    /// Channel `i`'s densities, in the form a Fock build takes.
    fn spin_density(&self, i: usize) -> SpinDensity<'_> {
        SpinDensity {
            total: &self.scf.density,
            spin: &self.densities[i],
            scale: self.scale(),
        }
    }

    /// Assemble the response once each channel's bare perturbation is known.
    fn into_response(
        self,
        molecule: &Molecule,
        params: &Am1Parameters,
        options: &PbcOptions,
        govs: Vec<Vec<Vec<COv>>>,
    ) -> Result<PhononResponse> {
        let us = solve_cphf(
            molecule,
            params,
            &self.basis,
            &self.core,
            &self.pairs,
            &self.translations,
            &self.orbitals,
            &govs,
            options,
            &self.neighbors,
            self.fill,
        )?;
        let channels = self
            .orbitals
            .into_iter()
            .zip(govs)
            .zip(us)
            .map(|((orbitals, gov), u)| ChannelResponse { orbitals, gov, u })
            .collect();
        Ok(PhononResponse {
            scf: self.scf,
            basis: self.basis,
            neighbors: self.neighbors,
            translations: self.translations,
            channels,
            channel_densities: self.densities,
            fill: self.fill,
        })
    }
}

/// Run the periodic SCF and recover each spin channel's orbitals from it.
fn solve_channel_ground(
    molecule: &Molecule,
    params: &Am1Parameters,
    options: &PbcOptions,
    what: &str,
) -> Result<ChannelGround> {
    let cell = molecule
        .cell
        .ok_or_else(|| Am1Error::InvalidInput(format!("a {what} needs a cell")))?;
    let scf = run_pbc_scf(molecule, params, options)?;
    if !scf.converged {
        return Err(Am1Error::InvalidInput(format!(
            "the periodic SCF did not converge; a {what} built on it would be meaningless"
        )));
    }
    let basis = Basis::build(molecule, params)?;
    let neighbors = NeighborList::build(molecule, options.realspace_cutoff);
    let translations = cell.image_offsets(options.realspace_cutoff);
    let k_points = options.resolve_kpoints(&cell)?;

    let (core, pairs) = build_realspace_core(
        molecule,
        &basis,
        params,
        &neighbors,
        &translations,
        options.exchange_cutoff,
        options.electric_field,
    )?;

    let delta = long_range_delta(molecule, params, &neighbors, options)?;
    let total = &scf.density;
    let total_origin = total.origin()?;

    // The density each spin channel's Fock contracts its **exchange** against, and how many
    // electrons one of its orbitals holds.
    //
    // Restricted: one channel, the total density at half weight, two electrons per orbital.
    // Unrestricted: two, `P^α = (P + S)/2` and `P^β = (P − S)/2` from what the SCF returns, each
    // at full weight with one electron per orbital.
    let (densities, fill) = crate::pbc::scf::spin_channel_densities(&scf);
    let scale = crate::pbc::scf::exchange_scale_for(fill);

    let occupancies: Vec<Occupancy> = if densities.len() == 2 {
        let (na, nb) = spin_populations(molecule, params, options)?;
        vec![
            Occupancy {
                per_level: 1.0,
                count: na,
            },
            Occupancy {
                per_level: 1.0,
                count: nb,
            },
        ]
    } else {
        vec![closed_shell_occupancy(molecule, params, options)?]
    };

    // Rebuild the converged Fock so the orbitals can be recovered. `PbcResult` keeps only the
    // real-space density — one extra diagonalization per k point is cheaper than carrying the
    // whole mesh's coefficients through the SCF for the rare caller that wants them.
    let mut orbitals = Vec::with_capacity(densities.len());
    for (channel, occupancy) in densities.iter().zip(&occupancies) {
        let fock = crate::pbc::scf::build_realspace_fock(
            &core,
            &pairs,
            total_origin,
            channel,
            scale,
            &basis,
            molecule,
            params,
            delta.as_ref(),
        )?;
        orbitals.push(solve_orbitals(
            &fock, &k_points, &basis, *occupancy, options,
        )?);
    }

    Ok(ChannelGround {
        scf,
        basis,
        neighbors,
        translations,
        core,
        pairs,
        densities,
        orbitals,
        fill,
    })
}

fn solve_phonon_response(
    molecule: &Molecule,
    params: &Am1Parameters,
    options: &PbcOptions,
) -> Result<PhononResponse> {
    let ground = solve_channel_ground(molecule, params, options, "periodic response")?;
    let mut govs = Vec::with_capacity(ground.orbitals.len());
    for (i, orb) in ground.orbitals.iter().enumerate() {
        govs.push(perturbed_fock_ov(
            molecule,
            params,
            &ground.basis,
            &ground.neighbors,
            ground.spin_density(i),
            orb,
            options,
        )?);
    }
    ground.into_response(molecule, params, options, govs)
}
/// Born effective charges `Z*_{a,αβ} = ∂(V P_α)/∂u_{a,β}`, one `3 × 3` tensor per atom.
///
/// # What a Born charge is here
///
/// It is the dipole a cell acquires per unit displacement of one atom, and it is what carries
/// the long-range electrostatics of a lattice vibration — without it a polar crystal's LO and TO
/// branches stay degenerate at `q → 0`, which is wrong by an amount that is not small.
///
/// In this model the cell dipole is `Σ_b Q_b R_b + Σ_b μ_b`, with `μ_b` the on-site `sp`
/// hybridization moment the SCF already reports. Differentiating,
///
/// ```text
/// Z*_{a,αβ} = Q_a δ_αβ  +  Σ_b R_{b,α} ∂Q_b/∂u_{a,β}  +  Σ_b ∂μ_{b,α}/∂u_{a,β}
/// ```
///
/// The first term is the atom's own charge moving. The other two are the electrons
/// rearranging, and they come from the same CPHF the Hessian uses.
///
/// # Why it is well defined under periodic boundary conditions when the dipole is not
///
/// `Σ_b R_b Q_b` depends on the choice of cell origin, and the polarization of a periodic solid
/// is famously defined only modulo a quantum. The **derivative** is not: charge is conserved, so
/// `Σ_b ∂Q_b/∂u_a = 0`, and the origin dependence cancels term by term. That is why this is
/// computable here while an absolute polarization is not.
///
/// # The check that matters
///
/// `Σ_a Z*_a = 0` — the acoustic sum rule for Born charges. Translating the whole crystal
/// produces no dipole. It follows from charge conservation and nothing else, so a violation is a
/// bug in the response rather than a physical effect. `tests/pbc_born_charges.rs` asserts it.
pub fn born_charges(
    molecule: &Molecule,
    params: &Am1Parameters,
    options: &PbcOptions,
) -> Result<Vec<[[f64; 3]; 3]>> {
    let response = solve_phonon_response(molecule, params, options)?;
    born_charges_from_response(molecule, params, &response)
}

fn born_charges_from_response(
    molecule: &Molecule,
    params: &Am1Parameters,
    response: &PhononResponse,
) -> Result<Vec<[[f64; 3]; 3]>> {
    let basis = &response.basis;
    let nat = molecule.atoms.len();
    let nao = basis.nao;
    let origin = response
        .scf
        .density
        .get(crate::lattice::ImageOffset::origin())
        .ok_or_else(|| Am1Error::InvalidInput("density is missing the origin block".into()))?;
    let charges = crate::pbc::ewald::net_charges(molecule, basis, params, origin)?;

    // One perturbation's response density at a time. Only its **origin** block is read, so
    // holding all `3N` of them — which is what this did — kept `ndof · n_T · nao²` doubles alive
    // to use `ndof · nao²` of them.
    let mut out = vec![[[0.0_f64; 3]; 3]; nat];
    for a in 0..nat {
        for beta in 0..3 {
            let dp = response.response_density_total(nao, 3 * a + beta);
            let block = dp
                .get(crate::lattice::ImageOffset::origin())
                .ok_or_else(|| {
                    Am1Error::InvalidInput("response density is missing the origin block".into())
                })?;
            for alpha in 0..3 {
                // 1) The atom's own charge moving with it.
                let mut acc = if alpha == beta { charges[a] } else { 0.0 };
                for b in 0..nat {
                    let off = basis.atom_offset[b];
                    let norb = basis.atom_norb[b];
                    // 2) Charge transfer: `∂Q_b = −∂p_b`, the population response.
                    let mut dq = 0.0;
                    for k in 0..norb {
                        dq -= block[(off + k, off + k)];
                    }
                    let rb = molecule.atoms[b].position;
                    acc += [rb.x, rb.y, rb.z][alpha] * dq;

                    // 3) The on-site sp hybridization moment, `μ_b = −2 dd_b P_{s,p}`, which the
                    //    dipole in `crate::scf` also carries. Omitting it would make `Z*` the
                    //    point-charge Born charge rather than this model's.
                    let elem = params.element(molecule.atoms[b].z)?;
                    if norb == 4 {
                        acc += -2.0 * elem.dd * block[(off, off + 1 + alpha)];
                    }
                }
                out[a][alpha][beta] = acc;
            }
        }
    }
    Ok(out)
}

/// `(α, ε_∞)` as returned by [`dielectric_tensor`], each a Cartesian 3 × 3.
pub type DielectricTensors = ([[f64; 3]; 3], [[f64; 3]; 3]);

/// The clamped-ion polarizability `α_αβ = ∂p_α/∂E_β` (Bohr³), in **any** dimensionality.
///
/// `α` is a response, and a response is well defined whatever the cell is periodic in: it is the
/// derivative of this model's own dipole with respect to this model's own field operator, and the
/// origin dependence that would spoil an absolute dipole cancels in the derivative because charge
/// is conserved. What is *not* defined below three dimensions is the step from `α` to a
/// dimensionless `ε_∞` — see [`dielectric_tensor`].
///
/// Available for a chain and a slab **since 0.2.2**. Through 0.2.1 the only entry point was
/// `dielectric_tensor`, which refused them for the `ε_∞` conversion's sake and took `α` down with
/// it — the response was computable all along.
///
/// # Reading it in reduced dimensionality
///
/// `α` itself is a volume in every case; what changes is what you may divide it by. The
/// susceptibility per unit cell measure is
///
/// | | measure | `α/measure` | dimensionless `ε_∞`? |
/// |---|---|---|---|
/// | crystal | volume, Bohr³ | dimensionless | yes, `1 + 4πα/Ω` |
/// | slab | area, Bohr² | a **length** | no — needs a thickness the cell does not fix |
/// | chain | length, Bohr | an **area** | no |
///
/// The two-dimensional susceptibility with units of length is the quantity the monolayer
/// literature reports, and it is `α/A` from here. Turning it into a dielectric constant needs a
/// thickness, which is a choice about the material and not something a supercell knows — supply
/// one and [`crate::pbc::dielectric_tensor_with_extent`] will do the conversion, including the
/// depolarization factor that makes the out-of-plane law `1/(1 − 4πχ)` rather than `1 + 4πχ`.
pub fn polarizability(
    molecule: &Molecule,
    params: &Am1Parameters,
    options: &PbcOptions,
) -> Result<[[f64; 3]; 3]> {
    Ok(polarizability_and_dielectric(molecule, params, options)?.0)
}

/// The macroscopic **longitudinal** dielectric function `ε(q)` along `q`, in any dimensionality.
///
/// # Why this exists and `ε_∞` alone does not
///
/// [`dielectric_tensor`] returns `ε_∞ = 1 + 4πα/Ω`, which is a **constant** — and that is a
/// three-dimensional accident, not the general case. The general relation is
///
/// ```text
/// ε(q) = 1 − v_d(q) χ⁰(q),      χ⁰(q) → −q² (q̂·α·q̂) / measure
/// ```
///
/// with `v_d` the bare Coulomb kernel of that dimensionality — the same object
/// [`crate::pbc::ewald::LongRangeKernel`] is built around, and whose divergence rates
/// `tests/phased_lowdim.rs` measures. Putting the three kernels in:
///
/// | | `v_d(q)` | `ε(q)` | at `q → 0` |
/// |---|---|---|---|
/// | crystal | `4π/q²` | `1 + 4π (q̂·α·q̂)/Ω` | a **constant** — this is `ε_∞` |
/// | slab, `q` in plane | `2π/\|q\|` | `1 + 2π (q̂·α·q̂) \|q\| / A` | **→ 1** |
/// | chain, `q` along it | `2 K₀(\|q\|ρ)` | `1 + 2 K₀(\|q\|ρ) q² (q̂·α·q̂) / L` | **→ 1** |
///
/// So a sheet or a wire has **no long-wavelength dielectric constant**: it does not screen a field
/// whose wavelength exceeds its own extent, and `ε(q) → 1`. That is not a limitation of this
/// implementation — it is the reason `ε_∞ = 1 + 4πα/Ω` cannot be evaluated there, and the reason a
/// slab has no LO–TO splitting at Γ. The two facts are the same fact, and
/// `tests/pbc_dielectric.rs` measures both.
///
/// The two-dimensional form is thickness-free, which is what makes it an intrinsic property of the
/// layer: `2π χ₂D` is the Rytova–Keldysh screening length, with `χ₂D = α/A` from
/// [`polarizability`]. Assigning a slab a thickness and quoting `1 + 4πχ₂D/d` is a different,
/// model-dependent number — [`crate::pbc::dielectric_tensor_with_extent`] returns it since 0.2.2,
/// with the thickness required rather than assumed, and it is a *choice* in a way that `ε(q)` and
/// `χ₂D` are not.
///
/// # `chain_radius`
///
/// The one-dimensional Coulomb kernel is a logarithm and has no value without a reference length:
/// `K₀(|q|ρ) ≈ ln(2/(|q|ρ)) − γ`. `ρ` is the transverse radius the wire's charge is spread over,
/// there is no natural choice, and it is therefore **required** for a chain and refused rather
/// than guessed — the same rule [`crate::pbc::ewald1d::AxisConvention`] applies to a charged
/// chain's energy. Ignored in two and three dimensions, where the kernel needs nothing.
pub fn dielectric_function(
    molecule: &Molecule,
    params: &Am1Parameters,
    options: &PbcOptions,
    q: crate::math::Vec3,
    chain_radius: Option<f64>,
) -> Result<f64> {
    let cell = molecule
        .cell
        .ok_or_else(|| Am1Error::InvalidInput("a dielectric function needs a cell".into()))?;
    let q_norm = q.norm();
    if q_norm < 1.0e-12 {
        return Err(Am1Error::InvalidInput(
            "the dielectric function is evaluated at a wavevector, and `q = 0` is exactly where \
             the three dimensionalities differ: a crystal has a finite direction-dependent limit, \
             a slab and a chain go to 1. Pass a small non-zero `q` and take the limit yourself, \
             or use `dielectric_tensor` for a crystal's constant."
                .into(),
        ));
    }
    // `q` must lie in the periodic subspace: a wavevector along a non-periodic direction is not a
    // Bloch label, and the response to it is a molecular polarizability rather than a dielectric
    // function.
    let along = cell.periodic_component(q);
    if (q - along).norm() > 1.0e-9 * q_norm {
        return Err(Am1Error::InvalidInput(
            "the wavevector has a component along a non-periodic direction, where there is no \
             Bloch label and no macroscopic dielectric function. Give a `q` inside the periodic \
             subspace."
                .into(),
        ));
    }

    let alpha = polarizability(molecule, params, options)?;
    let u = q / q_norm;
    let uv = [u.x, u.y, u.z];
    // `q̂·α·q̂`, the longitudinal polarizability along this direction.
    let mut longitudinal = 0.0;
    for (a, ua) in uv.iter().enumerate() {
        for (b, ub) in uv.iter().enumerate() {
            longitudinal += ua * alpha[a][b] * ub;
        }
    }
    let measure = cell.measure();
    let pi = std::f64::consts::PI;

    let coulomb = match cell.n_periodic() {
        3 => 4.0 * pi / (q_norm * q_norm),
        2 => 2.0 * pi / q_norm,
        1 => {
            let radius = chain_radius.ok_or_else(|| {
                Am1Error::InvalidInput(
                    "a chain's dielectric function needs a transverse radius: the 1D Coulomb \
                     kernel is `2 K₀(|q|ρ)`, a logarithm at small `q`, and it has no value \
                     without one. There is no natural choice, so it is required rather than \
                     guessed."
                        .into(),
                )
            })?;
            // Negated deliberately, so a `NaN` radius is rejected rather than sailing through a
            // `radius <= 0.0` that is false for it.
            #[allow(clippy::neg_cmp_op_on_partial_ord)]
            if !(radius > 0.0) {
                return Err(Am1Error::InvalidInput(format!(
                    "the chain radius must be positive, got {radius}"
                )));
            }
            2.0 * bessel_k0(q_norm * radius)
        }
        _ => {
            return Err(Am1Error::InvalidInput(
                "a molecule has no dielectric function; use the molecular polarizability".into(),
            ))
        }
    };

    Ok(1.0 + coulomb * q_norm * q_norm * longitudinal / measure)
}

/// Modified Bessel function of the second kind, order zero.
///
/// Abramowitz & Stegun 9.8.5 and 9.8.6 — the standard polynomial pair, accurate to about `1e-7`
/// absolute over the whole range, which is far below what a dielectric function built on a CPHF
/// polarizability can distinguish.
///
/// Written here rather than pulled in: it is the only special function the crate needs beyond
/// `erf`, and [`crate::pbc::ewald1d`] goes out of its way to *avoid* Bessel functions in the
/// energy — for a good reason, since they are singular on the chain axis where that module has to
/// be smooth. Here the argument is `|q|ρ > 0` by construction and the singularity at zero is the
/// physical logarithm, so there is nothing to avoid.
fn bessel_k0(x: f64) -> f64 {
    assert!(x > 0.0, "K0 is evaluated only for positive argument");
    if x <= 2.0 {
        // 9.8.1 for `I₀`, whose series runs in `(x/3.75)²` — **not** `(x/2)²`, which is the
        // variable the `K₀` series beside it uses. Writing one variable for both put `K₀(0.1)` at
        // 2.445975 against the table's 2.427069, which is 0.8 % and nothing else in this file
        // would have caught.
        let u = (x / 3.75) * (x / 3.75);
        let i0 = 1.0
            + u * (3.515_622_9
                + u * (3.089_942_4
                    + u * (1.206_749_2 + u * (0.265_973_2 + u * (0.036_076_8 + u * 0.004_581_3)))));
        // 9.8.5 for `K₀` itself, in `t = (x/2)²`.
        let t = (x / 2.0) * (x / 2.0);
        -(x / 2.0).ln() * i0 - 0.577_215_66
            + t * (0.422_784_20
                + t * (0.230_697_56
                    + t * (0.034_885_90
                        + t * (0.002_626_98 + t * (0.000_107_50 + t * 0.000_007_4)))))
    } else {
        // 9.8.6, in `t = 2/x`.
        let t = 2.0 / x;
        let poly = 1.253_314_14
            + t * (-0.078_323_58
                + t * (0.021_895_68
                    + t * (-0.010_624_46
                        + t * (0.005_878_72 + t * (-0.002_515_40 + t * 0.000_532_08)))));
        poly * (-x).exp() / x.sqrt()
    }
}
/// Clamped-ion polarizability `α_αβ = ∂p_α/∂E_β` (Bohr³) and electronic dielectric tensor
/// `ε_∞,αβ = δ_αβ + 4π α_αβ / Ω`.
///
/// # The approximation, stated plainly
///
/// The perturbation is a uniform field coupled to this model's own dipole operator: `+E·R_a` on
/// atom `a`'s diagonal, and `+E_i · dd_a` on its `s`–`p_i` block. That is the same dipole the
/// SCF reports and the same one [`born_charges`] differentiates, so the three are consistent
/// with each other.
///
/// It is **not** the Berry-phase polarization: this is the standard tight-binding / clamped-ion
/// treatment, and the position operator it uses is not a well-defined periodic operator, since
/// `R_a` is fixed only modulo a lattice vector.
///
/// # The origin dependence that does not appear
///
/// The obvious worry is that such a perturbation depends on where the cell origin is put. It
/// does not, and the reason is worth stating because it is the same reason [`born_charges`] is
/// well defined:
///
/// * shifting the origin adds a **constant** to the whole diagonal of the perturbation, which
///   moves every orbital energy by the same amount and leaves the response untouched; and
/// * the response conserves charge, `Σ_b ∂Q_b/∂E = 0`, so `Σ_b R_b ∂Q_b/∂E` is origin
///   independent term by term.
///
/// Returns `(α, ε_∞)`: the clamped-ion electronic polarizability and the dielectric tensor built
/// from it, each a Cartesian 3 × 3.
///
/// This is measured, not argued: [`dielectric_origin_sensitivity`] recomputes `ε_∞` with the
/// cell shifted, and `tests/pbc_dielectric.rs` finds **1.6 × 10⁻¹⁵** — machine precision —
/// for a shift of 1.7 Bohr along the periodic axis.
///
/// What remains approximate is the *clamped-ion, dipole* character of the operator itself, which
/// no amount of origin invariance fixes. For a system where charge circulates around the
/// periodic loop rather than responding locally, this is not the right quantity.
fn polarizability_and_dielectric(
    molecule: &Molecule,
    params: &Am1Parameters,
    options: &PbcOptions,
) -> Result<DielectricTensors> {
    let cell = molecule
        .cell
        .ok_or_else(|| Am1Error::InvalidInput("a dielectric response needs a cell".into()))?;
    // `ε_∞ = 1 + 4πα/Ω` is the three-dimensional relation, and `Ω` has to be a volume.
    // `Lattice::measure` returns a *length* for a chain and an *area* for a slab, so applying
    // this to either produced a number that was not a dielectric constant in any unit system —
    // which is what happened before 0.2.1, and what the LO-TO splitting reported for a "polar
    // chain" was built on. The depolarization factor of a low-dimensional system is a different
    // calculation, not a different denominator.

    let response = solve_field_response(molecule, params, options)?;
    let basis = &response.basis;
    let nao = basis.nao;
    let nat = molecule.atoms.len();

    // `α_αβ = ∂p_α/∂E_β`, with `p` the same dipole the field couples to. One perturbation at a
    // time — there are only three here, but each is `n_T · nao²` and only its origin block is read.
    let mut alpha = [[0.0_f64; 3]; 3];
    for beta in 0..3 {
        let dp = response.response_density_total(nao, beta);
        let block = dp
            .get(crate::lattice::ImageOffset::origin())
            .ok_or_else(|| {
                Am1Error::InvalidInput("response density is missing the origin block".into())
            })?;
        for a in 0..nat {
            let elem = params.element(molecule.atoms[a].z)?;
            let off = basis.atom_offset[a];
            let norb = basis.atom_norb[a];
            let mut dq = 0.0;
            for k in 0..norb {
                dq -= block[(off + k, off + k)];
            }
            let r = molecule.atoms[a].position;
            for (av, rv) in [r.x, r.y, r.z].iter().enumerate() {
                alpha[av][beta] += rv * dq;
            }
            if norb == 4 {
                for av in 0..3 {
                    alpha[av][beta] += -2.0 * elem.dd * block[(off, off + 1 + av)];
                }
            }
        }
    }

    // Into atomic units.
    //
    // The CPHF is solved in this crate's interior units: orbital energies in eV, positions in
    // Bohr. The perturbation is the dipole operator `M` (Bohr), so `U ~ M/Δε` carries Bohr/eV
    // and the `α = Σ_a R_a ΔQ_a` assembled above is in `e²·Bohr²/eV`, not Bohr³. One factor of
    // `E_h` converts it: `α[Bohr³] = α[e²Bohr²/eV] · (eV per Hartree)`.
    //
    // This was missing until 0.2.1, and nothing caught it: `ε_∞ = 1 + 4πα/Ω` came out 27.21×
    // too close to 1, while staying symmetric, positive-definite and origin-independent — every
    // property the tests here checked. It took comparing the *magnitude* against the isolated
    // molecule's finite-field polarizability, which agrees to 0.9 % once this factor is applied.
    for row in alpha.iter_mut() {
        for v in row.iter_mut() {
            *v *= crate::constants::HARTREE_TO_EV;
        }
    }

    // `ε_∞ = 1 + 4πα/Ω` is the **three-dimensional** relation. Assembled unconditionally here
    // because the caller decides what it means: `dielectric_tensor` returns it and refuses a
    // reduced-dimensional cell, `polarizability` drops it and does not.
    let measure = cell.measure();
    let mut epsilon = [[0.0_f64; 3]; 3];
    for a in 0..3 {
        for b in 0..3 {
            epsilon[a][b] =
                if a == b { 1.0 } else { 0.0 } + 4.0 * std::f64::consts::PI * alpha[a][b] / measure;
        }
    }
    Ok((alpha, epsilon))
}

/// Clamped-ion polarizability `α` (Bohr³) and electronic dielectric tensor `ε_∞`, **3D only**.
///
/// `ε_∞ = 1 + 4πα/Ω` needs `Ω` to be a volume, and [`crate::lattice::Lattice::measure`] returns an
/// area for a slab and a length for a chain — so applying this to either produces a number that is
/// not a dielectric constant in any unit system. That is what 0.2.0 did, and it is what the "127
/// cm⁻¹ of LO–TO splitting on a polar chain" was built on.
///
/// For a chain or a slab there are two well-defined answers and this is neither of them:
/// [`polarizability`] is the response itself, and
/// [`crate::pbc::dielectric_tensor_with_extent`] is `ε` once the caller supplies the thickness or
/// cross-section that the supercell does not fix. Both are available since 0.2.2.
pub fn dielectric_tensor(
    molecule: &Molecule,
    params: &Am1Parameters,
    options: &PbcOptions,
) -> Result<DielectricTensors> {
    let cell = molecule
        .cell
        .ok_or_else(|| Am1Error::InvalidInput("a dielectric tensor needs a cell".into()))?;
    if !cell.is_fully_periodic() {
        return Err(Am1Error::InvalidInput(
            "the electronic dielectric tensor is three-dimensional: ε∞ = 1 + 4πα/Ω needs Ω to be \
             a volume, and a chain or a slab has only a length or an area. Two things are \
             available instead. `pbc::polarizability` returns the same α this would and leaves \
             the conversion alone. `pbc::dielectric_tensor_with_extent` does the conversion, once \
             you name the thickness (slab) or cross-section (wire) the supercell does not fix — \
             it is a claim about where the material stops, so it is an argument rather than a \
             default."
                .into(),
        ));
    }
    polarizability_and_dielectric(molecule, params, options)
}

/// How much [`dielectric_tensor`] moves when the cell origin is shifted by `shift` (Bohr).
///
/// The honest measure of the position-operator approximation. Zero would mean the result is
/// origin-independent after all; a large number means the field perturbation is not describing
/// this system's response and the dielectric tensor should not be relied on.
pub fn dielectric_origin_sensitivity(
    molecule: &Molecule,
    params: &Am1Parameters,
    options: &PbcOptions,
    shift: crate::math::Vec3,
) -> Result<f64> {
    let (_, base) = dielectric_tensor(molecule, params, options)?;
    let mut shifted = molecule.clone();
    for atom in &mut shifted.atoms {
        atom.position += shift;
    }
    let (_, moved) = dielectric_tensor(&shifted, params, options)?;
    let mut worst = 0.0_f64;
    for a in 0..3 {
        for b in 0..3 {
            worst = worst.max((base[a][b] - moved[a][b]).abs());
        }
    }
    Ok(worst)
}

/// Run the periodic SCF and solve the CPHF for the three uniform-field perturbations.
fn solve_field_response(
    molecule: &Molecule,
    params: &Am1Parameters,
    options: &PbcOptions,
) -> Result<PhononResponse> {
    let ground = solve_channel_ground(molecule, params, options, "field response")?;

    // The field perturbation is on-site, so it lives entirely in the `T = 0` block and carries
    // no Bloch phase — which is exactly why it can reuse the phonon CPHF unchanged.
    //
    // The operator itself comes from `crate::dipole`, not from a copy written out here. There
    // are three consumers of this sign convention — the molecular field, this periodic field
    // response, and the Born charges — and a second transcription of `+R_a` on the diagonal
    // plus `+dd` on both `(s, p_β)` and `(p_β, s)` is a place for them to drift apart silently.
    // A sign error here does not fail: it returns a plausible polarizability.
    //
    // It is also **spin-independent**: a one-electron operator couples to both channels the same
    // way, so the same three blocks are projected onto each channel's own orbitals. What differs
    // between the channels is the orbitals, and — through the CPHF — how they respond.
    let nao = ground.basis.nao;
    let m = crate::dipole::dipole_operator(molecule, &ground.basis, params)?;
    let mut blocks: Vec<RealSpaceBlocks> = (0..3)
        .map(|_| RealSpaceBlocks::zeros(&ground.translations, nao))
        .collect();
    for (beta, block) in blocks.iter_mut().enumerate() {
        let origin = block.origin_mut()?;
        for mu in 0..nao {
            for nu in 0..nao {
                origin[(mu, nu)] += m[beta][(mu, nu)];
            }
        }
    }

    let mut govs = Vec::with_capacity(ground.orbitals.len());
    for channel in &ground.orbitals {
        let mut per_channel = Vec::with_capacity(channel.len());
        for orb in channel {
            let mut per_k = Vec::with_capacity(3);
            for block in &blocks {
                let fk = block.bloch_sum(&orb.k);
                per_k.push(project_ov(&fk, orb));
            }
            per_channel.push(per_k);
        }
        govs.push(per_channel);
    }

    ground.into_response(molecule, params, options, govs)
}
/// The long-range correction matrix for this cell, or `None`.
///
/// `params` carries the Klopman–Ohno tail, which needs each element's `ρ⁰`. Every consumer passes
/// it, because the tail shifts the Fock diagonal: a response built without it while the SCF was
/// built with it would be a response to a different Hamiltonian.
pub(crate) fn long_range_delta(
    molecule: &Molecule,
    params: &Am1Parameters,
    neighbors: &NeighborList,
    options: &PbcOptions,
) -> Result<Option<Matrix>> {
    Ok(crate::pbc::ewald::LongRangeMonopole::for_molecule_with(
        molecule,
        options
            .klopman_ohno_tail
            .then_some((params, options.realspace_cutoff)),
        neighbors,
        options.ewald,
    )?
    .map(|(m, _)| m.delta))
}

/// A complex `n_vir × n_occ` block, stored as two real matrices.
#[derive(Clone)]
struct COv {
    re: Vec<f64>,
    im: Vec<f64>,
}

impl COv {
    fn zeros(n: usize) -> Self {
        Self {
            re: vec![0.0; n],
            im: vec![0.0; n],
        }
    }
    #[inline]
    fn get_flat(&self, i: usize) -> (f64, f64) {
        (self.re[i], self.im[i])
    }
}

/// How many electrons one level holds when full, and how many there are to place.
///
/// Restricted, a level holds 2 and the count is the whole electron count. Unrestricted, each
/// channel is solved on its own: a level holds 1 and the count is that channel's population. The
/// two spins are **not** filled against a shared chemical potential here, because the multiplicity
/// fixes `n_α` and `n_β` separately — which is the same convention the periodic SCF uses.
#[derive(Clone, Copy)]
struct Occupancy {
    /// Electrons per level when full: 2 restricted, 1 per channel unrestricted.
    per_level: f64,
    /// Electrons to place in this channel.
    count: f64,
}

/// Valence electrons in the cell, after the net charge.
pub(crate) fn cell_electrons(
    molecule: &Molecule,
    params: &Am1Parameters,
    options: &PbcOptions,
) -> Result<f64> {
    let mut n = -options.charge;
    for atom in &molecule.atoms {
        n += params.element(atom.z)?.core_charge;
    }
    Ok(n)
}

/// Doubly-occupied levels holding every electron.
fn closed_shell_occupancy(
    molecule: &Molecule,
    params: &Am1Parameters,
    options: &PbcOptions,
) -> Result<Occupancy> {
    Ok(Occupancy {
        per_level: 2.0,
        count: cell_electrons(molecule, params, options)?,
    })
}

/// `(n_α, n_β)` from the electron count and the multiplicity.
///
/// The multiplicity fixes the difference, `n_α − n_β = 2S = multiplicity − 1`, and the total fixes
/// the sum. Rejected rather than rounded when the two are inconsistent — an even electron count
/// with an even multiplicity has no integer solution, and silently moving half an electron is the
/// kind of thing that returns a plausible number.
pub(crate) fn spin_populations(
    molecule: &Molecule,
    params: &Am1Parameters,
    options: &PbcOptions,
) -> Result<(f64, f64)> {
    let n = cell_electrons(molecule, params, options)?;
    let unpaired = options.multiplicity as f64 - 1.0;
    let n_alpha = 0.5 * (n + unpaired);
    let n_beta = 0.5 * (n - unpaired);
    if n_beta < -1.0e-9 {
        return Err(Am1Error::InvalidInput(format!(
            "multiplicity {} needs {unpaired} unpaired electrons but the cell has only {n}",
            options.multiplicity
        )));
    }
    Ok((n_alpha, n_beta.max(0.0)))
}

/// Diagonalize the converged Fock at every k and classify the levels.
fn solve_orbitals(
    fock: &RealSpaceBlocks,
    k_points: &[KPoint],
    basis: &Basis,
    occupancy: Occupancy,
    options: &PbcOptions,
) -> Result<Vec<KOrbitals>> {
    let n_elec = occupancy.count;

    let mut levels: Vec<(usize, usize, f64)> = Vec::new();
    let mut solved = Vec::new();
    for (ki, k) in k_points.iter().enumerate() {
        let hk = fock.bloch_sum(k);
        let eig = hermitian_eigen(&hk)?;
        for (i, e) in eig.values.iter().enumerate() {
            levels.push((ki, i, *e));
        }
        solved.push(eig);
    }

    // Fill against one chemical potential across the whole mesh, weighted by k.
    let fermi_levels: Vec<crate::fermi::Level> = levels
        .iter()
        .map(|(ki, _, e)| crate::fermi::Level {
            energy: *e,
            weight: occupancy.per_level * k_points[*ki].weight,
        })
        .collect();
    let filling = if options.smearing_ev > 0.0 {
        crate::fermi::Filling::Fermi {
            kt: options.smearing_ev,
        }
    } else {
        crate::fermi::Filling::Aufbau
    };
    let occ = crate::fermi::fill(&fermi_levels, n_elec, filling)?;

    let nao = basis.nao;
    let mut out = Vec::new();
    for (ki, k) in k_points.iter().enumerate() {
        let eig = &solved[ki];
        let mut occupations = vec![0.0; nao];
        for (idx, (kj, i, _)) in levels.iter().enumerate() {
            if *kj == ki {
                // `fill` returns the fraction of each level's weight that is filled; the
                // occupation per orbital is that times what a full level holds — 2 for a
                // restricted spin pair, 1 for one unrestricted channel.
                occupations[*i] = occupancy.per_level * occ.fractions[idx];
            }
        }
        // A level is "occupied" for the response if it carries charge, "virtual" if it has room.
        // With smearing a level can be both, and then it belongs to neither: a rotation inside
        // the partially filled manifold does not change the density to first order.
        let mut occ_idx = Vec::new();
        let mut vir_idx = Vec::new();
        for (i, f) in occupations.iter().enumerate() {
            // Against what a *full* level holds, not against 2. Hard-coding 2 here meant that on
            // the unrestricted path — where a full level holds 1 — no level was ever classified
            // occupied, `n_ov` was zero at every k, and the whole orbital-relaxation term
            // silently vanished. It cost 74 % of the force constants and raised no error.
            if *f > occupancy.per_level - 1.0e-6 {
                occ_idx.push(i);
            } else if *f < 1.0e-6 {
                vir_idx.push(i);
            }
        }
        let (co_re, co_im) = gather_columns(&eig.vectors_re, &eig.vectors_im, &occ_idx);
        let (cv_re, cv_im) = gather_columns(&eig.vectors_re, &eig.vectors_im, &vir_idx);
        out.push(KOrbitals {
            k: *k,
            energies: eig.values.clone(),
            occ: occ_idx,
            vir: vir_idx,
            co_re,
            co_im,
            cv_re,
            cv_im,
        });
    }
    Ok(out)
}

/// `Cᵥ† M C_o` for a complex `M` given as real/imaginary parts.
///
/// # Why this is two products and not one loop nest
///
/// Written as a single nest over `(a, i, μ, ν)` — which is how it read until 0.2.1 — the inner
/// `(M C_o)_{μi}` is recomputed for **every virtual `a`**, making the whole thing
/// `n_v · n_o · nao²`. Since `n_v` and `n_o` both grow with `nao`, that is `O(nao⁴)`, and it sat
/// inside the periodic CPHF, which calls it once per perturbation per iteration.
///
/// Factoring it as `T = M C_o` (`nao² n_o`) followed by `Cᵥ† T` (`nao n_v n_o`) makes it
/// `O(nao³)`, and hands both halves to a blocked kernel instead of a scalar nest. The complex
/// products are four real ones apiece: `(A + iB)(C + iD) = (AC − BD) + i(AD + BC)`.
fn project_ov(m: &CMatrix, orb: &KOrbitals) -> COv {
    let nao = m.re.rows;
    let (nv, no) = (orb.vir.len(), orb.occ.len());
    let mut out = COv::zeros(nv * no);
    if nv == 0 || no == 0 || nao == 0 {
        return out;
    }

    // T = M C_o, nao × n_o. Sequential: the caller is already parallel over perturbations.
    let mut t_re = m.re.matmul_seq(&orb.co_re);
    m.im.matmul_acc_seq(&orb.co_im, &mut t_re, -1.0);
    let mut t_im = m.re.matmul_seq(&orb.co_im);
    m.im.matmul_acc_seq(&orb.co_re, &mut t_im, 1.0);

    // Cᵥ† T with Cᵥ† = (conj Cᵥ)ᵀ. Conjugating flips the sign of Cᵥ's imaginary part, so the
    // usual complex product `(AC − BD) + i(AD + BC)` becomes `(AC + BD) + i(AD − BC)`.
    let mut re = orb.cv_re.transpose_matmul_seq(&t_re);
    orb.cv_im.transpose_matmul_acc_seq(&t_im, &mut re, 1.0);
    let mut im = orb.cv_re.transpose_matmul_seq(&t_im);
    orb.cv_im.transpose_matmul_acc_seq(&t_re, &mut im, -1.0);
    out.re.copy_from_slice(re.as_slice());
    out.im.copy_from_slice(im.as_slice());
    out
}

/// Copy the columns named by `idx` out of a complex coefficient matrix, into a compact block.
///
/// `occ`/`vir` are index lists rather than contiguous ranges — smearing can leave a partially
/// filled level in both — so the compact `nao × n` blocks the matmuls need have to be gathered.
fn gather_columns(c_re: &Matrix, c_im: &Matrix, idx: &[usize]) -> (Matrix, Matrix) {
    let nao = c_re.rows;
    let mut re = Matrix::zeros(nao, idx.len());
    let mut im = Matrix::zeros(nao, idx.len());
    for (col, &j) in idx.iter().enumerate() {
        for mu in 0..nao {
            re[(mu, col)] = c_re[(mu, j)];
            im[(mu, col)] = c_im[(mu, j)];
        }
    }
    (re, im)
}

/// One spin channel of the density the skeleton's **exchange** contracts against.
///
/// Borrowed rather than copied, and with the origin block pulled out, because the inner loop asks
/// for `P^σ(0,T)` once per `(μ,ν,λ,σ)` and a missing translation falls back to the origin — the
/// same rule the restricted code used, kept so that behaviour does not change with the refactor.
struct ExchangeChannel<'a> {
    blocks: &'a RealSpaceBlocks,
    origin: &'a Matrix,
    /// `0.5` when one channel stands for both spins, `1.0` per spin otherwise.
    scale: f64,
}

/// Skeleton (fixed-density) second derivative, one `3 × 3` block per image pair.
fn skeleton(
    molecule: &Molecule,
    params: &Am1Parameters,
    basis: &Basis,
    neighbors: &NeighborList,
    density: &RealSpaceBlocks,
    exchange_channels: &[ExchangeChannel<'_>],
    options: &PbcOptions,
) -> Result<Matrix> {
    let nat = molecule.atoms.len();
    let ndof = 3 * nat;
    let mut hess = Matrix::zeros(ndof, ndof);
    let beta = |e: &crate::params::Am1Element, orb: u8| if orb == 0 { e.beta_s } else { e.beta_p };

    let origin = density
        .get(crate::lattice::ImageOffset::origin())
        .ok_or_else(|| Am1Error::InvalidInput("density is missing the origin block".into()))?;

    type Block = (usize, usize, [[f64; 3]; 3]);
    let blocks: Vec<Result<Block>> = neighbors
        .pairs
        .par_iter()
        .map(|pair| -> Result<Block> {
            let eu = params.element(molecule.atoms[pair.i].z)?;
            let ev = params.element(molecule.atoms[pair.j].z)?;
            let (a, b, delta, offset) = if eu.has_p() || !ev.has_p() {
                (pair.i, pair.j, pair.delta, pair.offset)
            } else {
                (pair.j, pair.i, pair.delta * -1.0, pair.offset.negated())
            };
            let ea = params.element(molecule.atoms[a].z)?;
            let eb = params.element(molecule.atoms[b].z)?;
            let pa = molecule.atoms[a].position;
            let pb = pa + delta;
            let dvec = [
                Dual2::var(delta.x, 0),
                Dual2::var(delta.y, 1),
                Dual2::var(delta.z, 2),
            ];
            let te = crate::integrals::pair_two_electron_g::<Dual2>(ea, eb, dvec);
            let s = crate::overlap::diatom_overlap_dual2(ea, pa, eb, pb)?;
            let (oa, ob) = (basis.atom_offset[a], basis.atom_offset[b]);
            let (na, nb) = (basis.atom_norb[a], basis.atom_norb[b]);

            // This is the one line that makes it the k-point Hessian: the resonance and exchange
            // terms couple `μ` in cell 0 with `λ` in cell `T`, so they contract against `P(0,T)`.
            // At Γ every block is the same matrix and this reduces to the molecular expression;
            // with a mesh they differ, and the difference is the physics the taper was standing
            // in for.
            let pt = density.get(offset).unwrap_or(origin);

            let mut epair = Dual2::constant(0.0);
            for i in 0..na {
                let bi = beta(ea, basis.aos[oa + i].orb);
                for j in 0..nb {
                    let bj = beta(eb, basis.aos[ob + j].orb);
                    epair = epair + s[i][j] * (pt[(oa + i, ob + j)] * (bi + bj));
                }
            }
            // Electron–core attraction and Coulomb are on-site, so they use `P(0,0)`.
            for i in 0..na {
                for j in 0..na {
                    epair = epair + te.e1b[i][j] * origin[(oa + i, oa + j)];
                }
            }
            for k in 0..nb {
                for l in 0..nb {
                    epair = epair + te.e2a[k][l] * origin[(ob + k, ob + l)];
                }
            }
            let taper = match options.exchange_cutoff {
                Some(rc) => crate::hamiltonian::exchange_taper_scalar::<Dual2>(
                    (dvec[0] * dvec[0] + dvec[1] * dvec[1] + dvec[2] * dvec[2]).sqrt(),
                    rc,
                ),
                None => Dual2::constant(1.0),
            };
            for mu in 0..na {
                for nu in 0..na {
                    for la in 0..nb {
                        for si in 0..nb {
                            // The pair loop is over **unordered** pairs, so the ½ of
                            // `½ Σ_μνλσ P P (μν|λσ)` and the double counting of `(a,b)` with
                            // `(b,a)` cancel: no extra factor belongs here. The exchange keeps a
                            // ½ because the restricted same-spin density is half the total.
                            // These are the coefficients `crate::pbc::gradient` differentiates,
                            // and they have to be the same ones or the Hessian is the second
                            // derivative of a different energy than the force is the first
                            // derivative of.
                            let w = te.two_e(mu, nu, la, si);
                            let coul = origin[(oa + mu, oa + nu)] * origin[(ob + la, ob + si)];
                            // Summed over spin channels: `−Σ_σ s P^σ P^σ`, with `s = ½` and one
                            // channel carrying the total restricted, `s = 1` and two channels
                            // unrestricted. Those agree when `P^α = P^β = P/2`, which is what
                            // makes forcing UHF on a closed shell reproduce the RHF Hessian.
                            let mut exch = 0.0;
                            for channel in exchange_channels {
                                let cp = channel.blocks.get(offset).unwrap_or(channel.origin);
                                exch -=
                                    channel.scale * cp[(oa + mu, ob + la)] * cp[(oa + nu, ob + si)];
                            }
                            epair = epair + w * coul + w * taper * exch;
                        }
                    }
                }
            }
            // Core–core repulsion.
            let r = (dvec[0] * dvec[0] + dvec[1] * dvec[1] + dvec[2] * dvec[2]).sqrt();
            epair = epair
                + crate::repulsion::pair_core_energy_scalar::<Dual2>(
                    ea,
                    eb,
                    molecule.atoms[a].z,
                    molecule.atoms[b].z,
                    r,
                );
            Ok((a, b, epair.h))
        })
        .collect();

    for block in blocks {
        let (a, b, hb) = block?;
        for i in 0..3 {
            for j in 0..3 {
                let v = hb[i][j];
                hess[(3 * a + i, 3 * a + j)] += v;
                hess[(3 * b + i, 3 * b + j)] += v;
                hess[(3 * a + i, 3 * b + j)] -= v;
                hess[(3 * b + i, 3 * a + j)] -= v;
            }
        }
    }

    // Long-range monopole correction, fixed-charge part. Not optional and not small: leaving it
    // out is what made this path disagree with the validated molecular Hessian by 2.7e-3
    // eV/Bohr² on a water chain. A hydrogen chain does **not** show it, because its net atomic
    // charges vanish by symmetry and so does the correction's derivative — which is why the
    // first three tests here passed while the code was still wrong.
    if let Some((_, kernel)) =
        crate::pbc::ewald::LongRangeMonopole::for_molecule(molecule, neighbors, options.ewald)?
    {
        let charges = crate::pbc::ewald::net_charges(molecule, basis, params, origin)?;
        for (c, d, block) in crate::pbc::ewald::LongRangeMonopole::energy_hessian(
            molecule, neighbors, &kernel, &charges,
        )? {
            for i in 0..3 {
                for j in 0..3 {
                    hess[(3 * c + i, 3 * d + j)] += block[i][j];
                    if c != d {
                        hess[(3 * d + j, 3 * c + i)] += block[i][j];
                    }
                }
            }
        }
    }
    Ok(hess)
}

/// Skeleton derivative Fock, Bloch-summed and projected to the occ–virt block at every k.
///
/// Returned as `[k][perturbation]`.
#[allow(clippy::too_many_arguments)]
fn perturbed_fock_ov(
    molecule: &Molecule,
    params: &Am1Parameters,
    basis: &Basis,
    neighbors: &NeighborList,
    densities: SpinDensity<'_>,
    orbitals: &[KOrbitals],
    options: &PbcOptions,
) -> Result<Vec<Vec<COv>>> {
    let nat = molecule.atoms.len();
    let ndof = 3 * nat;
    let blocks = perturbed_fock_realspace(molecule, params, basis, neighbors, densities, options)?;
    let mut out = Vec::with_capacity(orbitals.len());
    for orb in orbitals {
        let mut per_k = Vec::with_capacity(ndof);
        for x in 0..ndof {
            let fk = blocks[x].bloch_sum(&orb.k);
            per_k.push(project_ov(&fk, orb));
        }
        out.push(per_k);
    }
    Ok(out)
}

/// The density a Fock build contracts against: the total for Coulomb, one channel for exchange.
///
/// The same split [`crate::pbc::scf::build_realspace_fock`] takes, and for the same reason — a
/// restricted build passes the total for both with `spin_scale = 0.5`, an unrestricted one passes
/// its own channel with `spin_scale = 1`, and one routine serves both. Grouped into a struct
/// rather than three more positional arguments because `(&p, &p, 0.5)` at a call site says nothing
/// about which of the two roles each `p` is playing.
#[derive(Clone, Copy)]
struct SpinDensity<'a> {
    /// Both spins summed. Coulomb and the long-range monopole see this.
    total: &'a RealSpaceBlocks,
    /// The channel the exchange contracts against.
    spin: &'a RealSpaceBlocks,
    /// `0.5` restricted, `1.0` unrestricted.
    scale: f64,
}

/// `∂F^σ(0,T)/∂R_c` for every Cartesian degree of freedom, in real space.
fn perturbed_fock_realspace(
    molecule: &Molecule,
    params: &Am1Parameters,
    basis: &Basis,
    neighbors: &NeighborList,
    densities: SpinDensity<'_>,
    options: &PbcOptions,
) -> Result<Vec<RealSpaceBlocks>> {
    let density = densities.total;
    use crate::dual::Dual;
    let nat = molecule.atoms.len();
    let nao = basis.nao;
    let beta = |e: &crate::params::Am1Element, orb: u8| if orb == 0 { e.beta_s } else { e.beta_p };
    let origin = density
        .get(crate::lattice::ImageOffset::origin())
        .ok_or_else(|| Am1Error::InvalidInput("density is missing the origin block".into()))?;

    let mut out: Vec<RealSpaceBlocks> = (0..3 * nat)
        .map(|_| RealSpaceBlocks::zeros(&density.translations, nao))
        .collect();

    for pair in &neighbors.pairs {
        let eu = params.element(molecule.atoms[pair.i].z)?;
        let ev = params.element(molecule.atoms[pair.j].z)?;
        let (a, b, delta, offset) = if eu.has_p() || !ev.has_p() {
            (pair.i, pair.j, pair.delta, pair.offset)
        } else {
            (pair.j, pair.i, pair.delta * -1.0, pair.offset.negated())
        };
        let ea = params.element(molecule.atoms[a].z)?;
        let eb = params.element(molecule.atoms[b].z)?;
        let pa = molecule.atoms[a].position;
        let pb = pa + delta;
        let te = crate::integrals::pair_two_electron_dual(ea, eb, delta);
        let s = crate::overlap::diatom_overlap_dual(ea, pa, eb, pb)?;
        let (oa, ob) = (basis.atom_offset[a], basis.atom_offset[b]);
        let (na, nb) = (basis.atom_norb[a], basis.atom_norb[b]);
        let r_dual = Dual {
            v: pair.r,
            d: [delta.x / pair.r, delta.y / pair.r, delta.z / pair.r],
        };
        let taper = match options.exchange_cutoff {
            Some(rc) => crate::hamiltonian::exchange_taper_scalar::<Dual>(r_dual, rc),
            None => Dual::constant(1.0),
        };

        // `∂/∂R_c` is `+∂/∂δ` when `c` is the second atom and `−∂/∂δ` when it is the first. A
        // self-image pair has `c` as both and the two cancel, which is correct.
        for (atom, sign) in [(b, 1.0_f64), (a, -1.0)] {
            for axis in 0..3 {
                let dof = 3 * atom + axis;
                // Resonance, on the T block and its mirror.
                //
                // The **½** is not optional and is easy to lose. The Fock element is
                // `F_μν = ½(β_μ + β_ν) S_μν`; the `(β_μ + β_ν)` without the ½ is the *energy*
                // coefficient, which already counts the `μν` and `νμ` orderings together.
                // Writing the energy coefficient into the Fock and then also placing it in both
                // blocks counts the pair twice. Measured: 23 eV/Bohr² of error at Γ, invisible on
                // a system whose orbital relaxation happens to be small.
                for (blk_offset, transposed) in [(offset, false), (offset.negated(), true)] {
                    let Some(blk) = out[dof].get_mut(blk_offset) else {
                        continue;
                    };
                    for i in 0..na {
                        let bi = beta(ea, basis.aos[oa + i].orb);
                        for j in 0..nb {
                            let bj = beta(eb, basis.aos[ob + j].orb);
                            let val = sign * 0.5 * (bi + bj) * s[i][j].d[axis];
                            if transposed {
                                blk[(ob + j, oa + i)] += val;
                            } else {
                                blk[(oa + i, ob + j)] += val;
                            }
                        }
                    }
                }
                // Core attraction and Coulomb: on-site, T = 0.
                let blk = out[dof].origin_mut()?;
                for i in 0..na {
                    for j in 0..na {
                        blk[(oa + i, oa + j)] += sign * te.e1b[i][j].d[axis];
                    }
                }
                for k in 0..nb {
                    for l in 0..nb {
                        blk[(ob + k, ob + l)] += sign * te.e2a[k][l].d[axis];
                    }
                }
                for mu in 0..na {
                    for nu in 0..na {
                        for la in 0..nb {
                            for si in 0..nb {
                                let w = te.two_e(mu, nu, la, si);
                                blk[(oa + mu, oa + nu)] +=
                                    sign * origin[(ob + la, ob + si)] * w.d[axis];
                                blk[(ob + la, ob + si)] +=
                                    sign * origin[(oa + mu, oa + nu)] * w.d[axis];
                            }
                        }
                    }
                }
            }
        }
        // Exchange lands on the T block and contracts P(0,T); handled separately so the taper
        // derivative rides along with it.
        for (atom, sign) in [(b, 1.0_f64), (a, -1.0)] {
            for axis in 0..3 {
                let dof = 3 * atom + axis;
                // The **spin** channel, not the total: exchange contracts `P^σ(0,T)`. Restricted,
                // `SpinDensity::restricted` makes this the total again at half weight.
                let pt = densities
                    .spin
                    .get(offset)
                    .unwrap_or(densities.spin.origin()?)
                    .clone();
                // `F(0,T)_{μλ}` and its partner `F(0,−T)_{λμ}` are the two orderings of the same
                // physical coupling, and the Fock carries both. At Γ every block is summed with
                // phase 1, so writing only one of them just rescales a term and hides the error;
                // with a mesh the two carry conjugate phases and the Hessian breaks. Measured:
                // exact at Γ, 2.5e-2 eV/Bohr² wrong at three k points.
                let mut k_block = vec![0.0_f64; na * nb];
                for mu in 0..na {
                    for la in 0..nb {
                        let mut acc = 0.0;
                        for nu in 0..na {
                            for si in 0..nb {
                                let w = te.two_e(mu, nu, la, si);
                                let d = taper.v * w.d[axis] + taper.d[axis] * w.v;
                                acc += pt[(oa + nu, ob + si)] * d;
                            }
                        }
                        k_block[mu * nb + la] = -densities.scale * sign * acc;
                    }
                }
                if let Some(blk) = out[dof].get_mut(offset) {
                    for mu in 0..na {
                        for la in 0..nb {
                            blk[(oa + mu, ob + la)] += k_block[mu * nb + la];
                        }
                    }
                }
                if let Some(blk) = out[dof].get_mut(offset.negated()) {
                    for mu in 0..na {
                        for la in 0..nb {
                            blk[(ob + la, oa + mu)] += k_block[mu * nb + la];
                        }
                    }
                }
            }
        }
    }

    // Long-range monopole correction in the perturbed Fock: `−∂V_a/∂R_c` on atom `a`'s own
    // diagonal, at `T = 0`. This is the term that lets the CPHF see the atomic charges
    // rearranging in response to a displacement; without it the response half is as wrong as
    // the skeleton half was without its counterpart.
    if let Some((_, kernel)) =
        crate::pbc::ewald::LongRangeMonopole::for_molecule(molecule, neighbors, options.ewald)?
    {
        let cell = molecule.cell.ok_or_else(|| {
            Am1Error::InvalidInput("the long-range kernel exists, so a cell must too".into())
        })?;
        let charges = crate::pbc::ewald::net_charges(molecule, basis, params, origin)?;

        // Every `Δ'(R_b − R_a)` once, into a table, instead of recomputing it inside the loops.
        //
        // The loops below ask for the same lattice sums repeatedly: the `a == c` branch walks
        // every `b` for each `c`, and the `a != c` branch asks for `(c, a)` for every ordered
        // pair — about `2·nat²` evaluations of a sum over every translation and reciprocal
        // vector, where only `nat(nat−1)/2` of them are distinct. `Δ'` is **odd** in the
        // separation (`Δ` is even, and its gradient is therefore odd), so half the table is the
        // negation of the other half. That is a fourfold reduction in the dominant cost here,
        // paid for with `O(nat²)` of `Vec3` — the pair list is already `O(nat²)`.
        let table = {
            use rayon::prelude::*;
            let pairs: Vec<(usize, usize)> = (0..nat)
                .flat_map(|a| ((a + 1)..nat).map(move |b| (a, b)))
                .collect();
            let values: Vec<crate::math::Vec3> = pairs
                .par_iter()
                .map(|&(a, b)| {
                    let r = molecule.atoms[b].position - molecule.atoms[a].position;
                    crate::pbc::ewald::delta_gradient(r, &cell, &neighbors.translations, &kernel)
                })
                .collect();
            let mut t = vec![crate::math::Vec3::zero(); nat * nat];
            for (&(a, b), v) in pairs.iter().zip(&values) {
                t[a * nat + b] = *v;
                t[b * nat + a] = *v * -1.0;
            }
            t
        };

        for c in 0..nat {
            for a in 0..nat {
                // `∂Δ_ab/∂R_c = g_ab (δ_bc − δ_ac)`, so only pairs involving `c` survive.
                let mut dv = crate::math::Vec3::zero();
                if a == c {
                    for (b, qb) in charges.iter().enumerate() {
                        if b == c {
                            continue;
                        }
                        dv -= table[a * nat + b] * *qb;
                    }
                } else {
                    dv += table[a * nat + c] * charges[c];
                }
                let off = basis.atom_offset[a];
                for (axis, d) in [dv.x, dv.y, dv.z].iter().enumerate() {
                    let blk = out[3 * c + axis].origin_mut()?;
                    for k in 0..basis.atom_norb[a] {
                        blk[(off + k, off + k)] -= d;
                    }
                }
            }
        }
    }
    Ok(out)
}

/// Solve the CPHF at every k for every perturbation, by damped fixed-point iteration.
#[allow(clippy::too_many_arguments)]
fn solve_cphf(
    molecule: &Molecule,
    params: &Am1Parameters,
    basis: &Basis,
    _core: &RealSpaceBlocks,
    pairs: &crate::pbc::scf::PeriodicPairs,
    translations: &[crate::lattice::ImageOffset],
    orbitals: &[Vec<KOrbitals>],
    gov: &[Vec<Vec<COv>>],
    options: &PbcOptions,
    neighbors: &NeighborList,
    fill: f64,
) -> Result<Vec<Vec<Vec<COv>>>> {
    let ndof = gov[0][0].len();
    let nao = basis.nao;
    let delta = long_range_delta(molecule, params, neighbors, options)?;
    let zero_core = RealSpaceBlocks::zeros(translations, nao);
    // `G^σ(ΔP) = J(ΔP_tot) − K(ΔP_σ)`: restricted, one channel carries both spins and the
    // exchange runs at half weight; unrestricted, each channel contracts its own `ΔP_σ` in full.
    let scale = crate::pbc::scf::exchange_scale_for(fill);

    // Start from the uncoupled solution, `U = −G/(ε_a − ε_i)`.
    let mut u: Vec<Vec<Vec<COv>>> = orbitals
        .iter()
        .zip(gov)
        .map(|(orbs, g)| {
            orbs.iter()
                .enumerate()
                .map(|(ki, orb)| {
                    (0..ndof)
                        .map(|x| divide_by_denominator(&g[ki][x], orb))
                        .collect()
                })
                .collect()
        })
        .collect();

    for iteration in 0..CPHF_MAX_ITER {
        let mut worst = 0.0_f64;
        // One response density and one response Fock per perturbation, per spin channel.
        let mut next: Vec<Vec<Vec<COv>>> = orbitals
            .iter()
            .map(|orbs| orbs.iter().map(|_| Vec::with_capacity(ndof)).collect())
            .collect();

        for x in 0..ndof {
            // **This perturbation's** response densities, built here and dropped at the end of the
            // iteration rather than all `3N` up front. The work is the same either way — the loop
            // nest is identical, only its order changed — but the resident set goes from
            // `(1 + n_channels) · ndof · n_T · nao²` doubles to `(1 + n_channels) · n_T · nao²`.
            // That is a factor of `3N` on what was the largest array in this solver.
            //
            // Each channel's density, and their sum. The **sum** drives the Coulomb half of the
            // kernel and the long-range term, which is what couples the two spins: solving them
            // independently would drop `J(ΔP_β)` from `G^α` entirely.
            let dp_channel: Vec<RealSpaceBlocks> = u
                .iter()
                .zip(orbitals)
                .map(|(uc, orbs)| response_density_one(uc, orbs, translations, nao, fill, x))
                .collect();
            let mut dp_total = dp_channel[0].clone();
            for extra in &dp_channel[1..] {
                dp_total.add_assign(extra);
            }

            for (ci, orbs) in orbitals.iter().enumerate() {
                let dfock = crate::pbc::scf::build_realspace_fock(
                    &zero_core,
                    pairs,
                    dp_total.origin()?,
                    &dp_channel[ci],
                    scale,
                    basis,
                    molecule,
                    params,
                    None,
                )?;
                // The long-range term of the response uses the population form, not the
                // net-charge form: only the part linear in `P` belongs in an operator that must
                // be linear in its argument. See `crate::fock::build_fock_spin_with`. It reads
                // the **total** response density — it is a Coulomb term and both spins feed it.
                let dfock = add_response_long_range(
                    dfock,
                    molecule,
                    basis,
                    params,
                    delta.as_ref(),
                    &dp_total,
                )?;
                for (ki, orb) in orbs.iter().enumerate() {
                    let fk = dfock.bloch_sum(&orb.k);
                    let projected = project_ov(&fk, orb);
                    let mut combined = gov[ci][ki][x].clone();
                    for i in 0..combined.re.len() {
                        combined.re[i] += projected.re[i];
                        combined.im[i] += projected.im[i];
                    }
                    let candidate = divide_by_denominator(&combined, orb);
                    for i in 0..candidate.re.len() {
                        worst = worst
                            .max((candidate.re[i] - u[ci][ki][x].re[i]).abs())
                            .max((candidate.im[i] - u[ci][ki][x].im[i]).abs());
                    }
                    next[ci][ki].push(candidate);
                }
            }
        }
        // Damped update; the fixed-point step is the preconditioned residual, so `worst` is
        // directly comparable to the molecular solver's convergence measure.
        for (ci, per_c) in next.into_iter().enumerate() {
            for (ki, per_k) in per_c.into_iter().enumerate() {
                for (x, cand) in per_k.into_iter().enumerate() {
                    for i in 0..cand.re.len() {
                        u[ci][ki][x].re[i] = 0.5 * u[ci][ki][x].re[i] + 0.5 * cand.re[i];
                        u[ci][ki][x].im[i] = 0.5 * u[ci][ki][x].im[i] + 0.5 * cand.im[i];
                    }
                }
            }
        }
        if worst < CPHF_TOL {
            return Ok(u);
        }
        if iteration + 1 == CPHF_MAX_ITER {
            return Err(Am1Error::CphfNotConverged {
                perturbations: ndof,
                iterations: CPHF_MAX_ITER,
                residual: worst,
            });
        }
    }
    Ok(u)
}

/// `U = −G / (ε_a − ε_i)`, with degenerate pairs dropped.
fn divide_by_denominator(g: &COv, orb: &KOrbitals) -> COv {
    let no = orb.occ.len();
    let mut out = COv::zeros(g.re.len());
    for (vi, &a) in orb.vir.iter().enumerate() {
        for (oi, &i) in orb.occ.iter().enumerate() {
            let d = orb.energies[a] - orb.energies[i];
            let idx = vi * no + oi;
            if d.abs() < DEGENERACY_FLOOR {
                continue;
            }
            out.re[idx] = -g.re[idx] / d;
            out.im[idx] = -g.im[idx] / d;
        }
    }
    out
}

/// `ΔP(0,T) = Σ_k w_k e^{−ik·T} f·[C_v U C_o† + h.c.]`, one set of blocks per perturbation.
///
/// `fill` is what one orbital holds: 2 for a restricted spin pair, 1 for one unrestricted channel.
/// `ΔP(0,T) = Σ_k w_k e^{−ik·T} f·[C_v U C_o† + h.c.]` for **one** perturbation.
///
/// `fill` is what one orbital holds: 2 for a restricted spin pair, 1 for one unrestricted channel.
///
/// # One at a time, deliberately
///
/// This used to build all `3N` of them and hand back a `Vec`. The work is identical either way —
/// the loop nest is the same, only its order changed — but the *resident* set is not: the caller
/// uses one perturbation at a time, so holding all of them cost `ndof · n_T · nao²` doubles where
/// `n_T · nao²` will do. That is a factor of `3N` on what was the largest array in the `q = 0`
/// response, and on a hundred-atom cell with a hundred translations it is gigabytes.
///
/// The CPHF also kept a **second** copy of the same size, the summed `ΔP_tot` the spin channels
/// couple through. Streaming removes that too: the sum is over channels, not over perturbations,
/// so at one `x` it is one block set.
fn response_density_one(
    u: &[Vec<COv>],
    orbitals: &[KOrbitals],
    translations: &[crate::lattice::ImageOffset],
    nao: usize,
    fill: f64,
    x: usize,
) -> RealSpaceBlocks {
    let mut out_x = RealSpaceBlocks::zeros(translations, nao);
    for (ki, orb) in orbitals.iter().enumerate() {
        let no = orb.occ.len();
        // Build ΔP(k) in the AO basis: f·[C_v U C_o† + h.c.].
        //
        // As two products rather than one nest, for the reason spelled out on `project_ov`:
        // contracting `(v, o, μ, ν)` in a single loop rebuilds `C_v U` for every occupied index
        // and costs `n_v n_o nao²`, i.e. `O(nao⁴)`. Going through `A = C_v U` (`nao n_v n_o`)
        // then `A C_o†` (`nao² n_o`) is `O(nao³)`.
        let nv = orb.vir.len();
        let mut u_re = Matrix::zeros(nv, no);
        let mut u_im = Matrix::zeros(nv, no);
        for vi in 0..nv {
            for oi in 0..no {
                let (ur, ui) = u[ki][x].get_flat(vi * no + oi);
                u_re[(vi, oi)] = ur;
                u_im[(vi, oi)] = ui;
            }
        }
        // A = C_v U, nao × n_o.
        let mut a_re = orb.cv_re.matmul_seq(&u_re);
        orb.cv_im.matmul_acc_seq(&u_im, &mut a_re, -1.0);
        let mut a_im = orb.cv_re.matmul_seq(&u_im);
        orb.cv_im.matmul_acc_seq(&u_re, &mut a_im, 1.0);
        // B = A C_o†; conjugating C_o flips the sign of its imaginary part.
        let mut b_re = a_re.matmul_transpose_seq(&orb.co_re);
        a_im.matmul_transpose_acc_seq(&orb.co_im, &mut b_re, 1.0);
        let mut b_im = a_im.matmul_transpose_seq(&orb.co_re);
        a_re.matmul_transpose_acc_seq(&orb.co_im, &mut b_im, -1.0);
        // f·(B + B†), which is what makes ΔP Hermitian.
        let mut dp = CMatrix::zeros(nao);
        for mu in 0..nao {
            for nu in 0..nao {
                let re = fill * (b_re[(mu, nu)] + b_re[(nu, mu)]);
                let im = fill * (b_im[(mu, nu)] - b_im[(nu, mu)]);
                dp.add(mu, nu, re, im);
            }
        }
        // Back to real space with the k weight and the inverse Bloch phase.
        for (t, block) in out_x
            .translations
            .clone()
            .iter()
            .zip(out_x.blocks.iter_mut())
        {
            let (c, s) = orb.k.phase(*t);
            for mu in 0..nao {
                for nu in 0..nao {
                    let (re, im) = dp.get(mu, nu);
                    // Re[ e^{−ik·T} (re + i im) ] · w
                    block[(mu, nu)] += orb.k.weight * (c * re + s * im);
                }
            }
        }
    }
    out_x
}

/// Apply the long-range monopole response — the part linear in `P` only.
fn add_response_long_range(
    mut fock: RealSpaceBlocks,
    molecule: &Molecule,
    basis: &Basis,
    params: &Am1Parameters,
    delta: Option<&Matrix>,
    dp: &RealSpaceBlocks,
) -> Result<RealSpaceBlocks> {
    let Some(d) = delta else { return Ok(fock) };
    let _ = params;
    let nat = molecule.atoms.len();
    let origin_p = dp.origin()?;
    let mut population = vec![0.0; nat];
    for (a, pop) in population.iter_mut().enumerate() {
        let off = basis.atom_offset[a];
        for k in 0..basis.atom_norb[a] {
            *pop += origin_p[(off + k, off + k)];
        }
    }
    let blk = fock.origin_mut()?;
    for a in 0..nat {
        let mut shift = 0.0;
        for (b, pop) in population.iter().enumerate() {
            shift += d[(a, b)] * pop;
        }
        let off = basis.atom_offset[a];
        for k in 0..basis.atom_norb[a] {
            blk[(off + k, off + k)] += shift;
        }
    }
    Ok(fock)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The `O(nao⁴)` loop nest `project_ov` used to be, kept as the reference the factored
    /// version is checked against.
    fn project_ov_naive(m: &CMatrix, orb: &KOrbitals) -> COv {
        let nao = m.re.rows;
        let (nv, no) = (orb.vir.len(), orb.occ.len());
        let mut out = COv::zeros(nv * no);
        for vi in 0..nv {
            for oi in 0..no {
                let mut re = 0.0;
                let mut im = 0.0;
                for mu in 0..nao {
                    // (M C_o)_{mu,i}, rebuilt for every virtual — this is the O(nao⁴).
                    let mut tre = 0.0;
                    let mut tim = 0.0;
                    for nu in 0..nao {
                        let (mr, mi) = m.get(mu, nu);
                        let (cr, ci) = (orb.co_re[(nu, oi)], orb.co_im[(nu, oi)]);
                        tre += mr * cr - mi * ci;
                        tim += mr * ci + mi * cr;
                    }
                    // conj(C_v)_{mu,a} · that
                    let (vr, vi_) = (orb.cv_re[(mu, vi)], -orb.cv_im[(mu, vi)]);
                    re += vr * tre - vi_ * tim;
                    im += vr * tim + vi_ * tre;
                }
                out.re[vi * no + oi] = re;
                out.im[vi * no + oi] = im;
            }
        }
        out
    }

    /// Deterministic pseudo-random fill; no rand dependency, and reproducible across runs.
    fn filled(rows: usize, cols: usize, seed: u64) -> Matrix {
        let mut m = Matrix::zeros(rows, cols);
        let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        for v in m.as_mut_slice() {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *v = ((s >> 11) as f64 / (1u64 << 53) as f64) - 0.5;
        }
        m
    }

    fn orbitals_for(nao: usize, no: usize, nv: usize) -> KOrbitals {
        let c_re = filled(nao, nao, 7);
        let c_im = filled(nao, nao, 13);
        let occ: Vec<usize> = (0..no).collect();
        let vir: Vec<usize> = (nao - nv..nao).collect();
        let (co_re, co_im) = gather_columns(&c_re, &c_im, &occ);
        let (cv_re, cv_im) = gather_columns(&c_re, &c_im, &vir);
        KOrbitals {
            k: KPoint::gamma(),
            energies: vec![0.0; nao],
            occ,
            vir,
            co_re,
            co_im,
            cv_re,
            cv_im,
        }
    }

    #[test]
    fn the_factored_project_ov_matches_the_loop_nest_it_replaced() {
        for &(nao, no, nv) in &[(8usize, 3usize, 4usize), (24, 10, 12), (40, 17, 20)] {
            let orb = orbitals_for(nao, no, nv);
            let m = CMatrix {
                n: nao,
                re: filled(nao, nao, 29),
                im: filled(nao, nao, 31),
            };
            let want = project_ov_naive(&m, &orb);
            let got = project_ov(&m, &orb);
            let scale = want
                .re
                .iter()
                .chain(want.im.iter())
                .fold(0.0f64, |a, b| a.max(b.abs()));
            let mut worst = 0.0f64;
            for i in 0..want.re.len() {
                worst = worst.max((want.re[i] - got.re[i]).abs());
                worst = worst.max((want.im[i] - got.im[i]).abs());
            }
            eprintln!("    nao={nao:3} n_o={no:3} n_v={nv:3}: max |factored - nest| = {worst:.3e} of {scale:.3e}");
            assert!(worst < 1.0e-11 * scale.max(1.0), "disagreement {worst:.3e}");
        }
    }

    /// Multiply–accumulates each form issues, read off the shapes it hands its kernels.
    ///
    /// The nest rebuilds `(M C_o)` inside the virtual loop: `n_v n_o nao` outer steps each doing a
    /// `nao`-long complex dot, four real products apiece. The factored form does `M C_o`
    /// (`nao × nao × n_o`) and then `Cᵥ† T` (`n_v × nao × n_o`), four real matmuls each.
    fn project_ov_flops(nao: usize, no: usize, nv: usize) -> (f64, f64) {
        let (nao, no, nv) = (nao as f64, no as f64, nv as f64);
        (4.0 * nv * no * nao * nao, 4.0 * nao * no * (nao + nv))
    }

    /// The point of the rewrite: the cost exponent in `nao`, with `n_o` and `n_v` growing
    /// alongside it as they physically do.
    ///
    /// The nest recomputes `(M C_o)` for every virtual index, so it is `n_v n_o nao²`, i.e.
    /// `O(nao⁴)`. Factoring into `M C_o` then `Cᵥ† ·` makes it `O(nao³)`.
    ///
    /// # What is asserted, and what is only printed
    ///
    /// The claim is about an **exponent**, which is a count of operations and not a duration, so
    /// the count is what gets asserted — the same rule `DcResult::diagonalization_work` follows.
    /// Through 0.2.2 this asserted that the measured *speedup ratio* grew with `nao`, on the
    /// reasoning that a ratio divides out the machine's load. It does not: both halves of the
    /// ratio are wall clock, and the largest case has the most memory traffic and so suffers most.
    /// Idle, three consecutive runs gave 54/108/357, 46/137/385 and 48/94/376; with a build
    /// running alongside, the same test measured 61/258/**46** and failed.
    ///
    /// So the timings are printed as evidence and asserted only where wall clock can carry the
    /// weight: the factored form must be **much** faster at every size. Observed 46–385×, so a 5×
    /// floor is immune to any plausible load while still failing loudly if the rewrite is undone.
    /// That the two forms compute the same thing is `the_factored_projection_matches_the_nest`
    /// directly above; together they say "same answer, provably fewer operations, and measurably
    /// faster".
    #[test]
    fn factoring_project_ov_lowers_its_operation_count() {
        let sizes = [32usize, 64, 128];
        let mut ratios = Vec::new();
        let mut op_ratios = Vec::new();
        for &nao in &sizes {
            let (no, nv) = (nao / 2, nao / 2);
            let orb = orbitals_for(nao, no, nv);
            let m = CMatrix {
                n: nao,
                re: filled(nao, nao, 29),
                im: filled(nao, nao, 31),
            };
            // One untimed call each, so neither pays the first-touch page faults.
            let _ = project_ov_naive(&m, &orb);
            let _ = project_ov(&m, &orb);

            let reps = 3;
            let t0 = std::time::Instant::now();
            for _ in 0..reps {
                std::hint::black_box(project_ov_naive(&m, &orb));
            }
            let naive = t0.elapsed().as_secs_f64() / reps as f64;
            let t1 = std::time::Instant::now();
            for _ in 0..reps {
                std::hint::black_box(project_ov(&m, &orb));
            }
            let fast = t1.elapsed().as_secs_f64() / reps as f64;
            let (ops_naive, ops_fast) = project_ov_flops(nao, no, nv);
            eprintln!(
                "    nao={nao:4}: nest {:9.3} ms, factored {:9.3} ms, speedup {:6.2}x  |  \
                 flops {:.3e} vs {:.3e}, ratio {:6.1}x",
                naive * 1e3,
                fast * 1e3,
                naive / fast,
                ops_naive,
                ops_fast,
                ops_naive / ops_fast
            );
            ratios.push(naive / fast);
            op_ratios.push(ops_naive / ops_fast);
        }

        // The exponent, asserted where it is deterministic. With `n_o = n_v = nao/2` the ratio is
        // `n_v nao/(nao + n_v) = nao/3`, so doubling `nao` doubles it — one power of `nao` less
        // work, which is the whole claim. A tolerance of 1 % covers the integer division only.
        for (i, w) in op_ratios.windows(2).enumerate() {
            let growth = w[1] / w[0];
            assert!(
                (growth - 2.0).abs() < 0.02,
                "operation-count advantage grew {growth:.4}x from nao={} to nao={}, not 2x",
                sizes[i],
                sizes[i + 1]
            );
        }
        assert!(
            op_ratios[2] > 40.0,
            "at nao={} the nest should issue over 40x the operations, not {:.1}x",
            sizes[2],
            op_ratios[2]
        );

        // And wall clock, asserted only where it can carry the weight. See the doc comment: the
        // *trend* is printed, not asserted, because a loaded machine reverses it.
        for (i, r) in ratios.iter().enumerate() {
            assert!(
                *r > 5.0,
                "at nao={} the factored form is only {r:.2}x the nest; observed 46-385x idle",
                sizes[i]
            );
        }
    }
}

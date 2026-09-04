// SPDX-License-Identifier: GPL-3.0-or-later

//! NDDO SCF driver — restricted (RHF) closed shell and unrestricted (UHF) open shell.
//!
//! Because NDDO assumes an orthonormal AO basis, the working equations are the plain
//! eigenproblem `F C = C ε` (no `S`); overlap enters only the resonance term of `H_core`.
//! The initial density is a **superposition of atomic densities** ([`sad_density`]) — the
//! exact free-atom density in a minimal valence basis, far better than the bare-core guess —
//! and charge convergence is accelerated with the A-DIIS→CDIIS hybrid on the `[F,P]` commutator.

use crate::basis::Basis;
use crate::constants::{AU_DIPOLE_TO_DEBYE, EV_TO_KCAL};
use crate::error::{Am1Error, Result};
use crate::fock::{build_fock, build_fock_spin};
use crate::hamiltonian::CoreHamiltonian;
use crate::linalg::{symmetric_eigen, Matrix};
use crate::math::Vec3;
use crate::params::Am1Parameters;

use crate::system::Molecule;

/// Choice of SCF reference (spin treatment) — restricted vs unrestricted.
///
/// This is orthogonal to [`Am1Options::multiplicity`]: the multiplicity fixes the α/β electron
/// counts, while the reference chooses how the orbitals are solved. A closed-shell singlet can
/// therefore be run **either** restricted (RHF, the usual choice) **or** unrestricted (UHF, e.g.
/// as the starting point for a broken-symmetry singlet).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScfReference {
    /// RHF for a closed shell (equal α/β counts), UHF otherwise. The default; preserves the
    /// historical behavior.
    Auto,
    /// Restricted Hartree–Fock: doubly-occupied spatial orbitals. Requires a closed shell
    /// (equal α/β counts); an open-shell request is rejected (no ROHF).
    Restricted,
    /// Unrestricted Hartree–Fock: independent α/β orbitals, used even for a singlet.
    Unrestricted,
}

/// SCF charge-convergence accelerator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScfAccelerator {
    /// No extrapolation (plain iteration).
    None,
    /// Pulay CDIIS on the `[F,P]` commutator throughout.
    Cdiis,
    /// **A-DIIS** (Hu & Yang, *J. Chem. Phys.* **132**, 054109 (2010)) while far from
    /// convergence, switching to CDIIS once the commutator error drops below a threshold —
    /// the robust hybrid recommended for hard cases (radicals, small gaps, poor guesses).
    AdiisCdiis,
}

#[derive(Clone, Debug)]
pub struct Am1Options {
    pub charge: f64,
    pub multiplicity: usize,
    /// Restricted vs unrestricted reference (see [`ScfReference`]). Default [`ScfReference::Auto`].
    pub reference: ScfReference,
    pub max_scf: usize,
    pub e_tol: f64,
    pub p_tol: f64,
    /// Legacy flag: `false` forces [`ScfAccelerator::None`] regardless of `accelerator`.
    pub use_diis: bool,
    pub accelerator: ScfAccelerator,
    /// Commutator-error norm below which the ADIIS→CDIIS hybrid switches to CDIIS.
    pub adiis_switch: f64,
    /// Real-space cutoff (Bohr) for periodic image sums. Ignored for a molecule, where every
    /// pair is kept regardless — the NDDO two-centre integrals decay as `1/R`, so screening
    /// them by distance changes the answer rather than saving work that did not matter.
    ///
    /// Under a cell this truncation *is* an approximation: the same `1/R` tail makes the sum
    /// only conditionally convergent, so the result must be checked against the cutoff until
    /// the Ewald treatment replaces it.
    pub realspace_cutoff: f64,
    /// Sum the long-range monopole electrostatics by **Ewald summation** rather than by the
    /// real-space cutoff.
    ///
    /// Default `true`, and a no-op without a three-dimensional cell — a molecule has no lattice
    /// sum to correct, and a slab or a chain would need a reciprocal sum that is not implemented
    /// (asking for one there is an error rather than a different answer).
    ///
    /// This is what makes a **charged** cell meaningful. Without it the monopole sum
    /// `Σ_T Q²/|T|` diverges: a +1 water cell in an 8 Å cube measured −331 eV at a 20 Bohr
    /// cutoff and +72 eV at 130 Bohr. With it the energy is cutoff-independent, under the
    /// tin-foil boundary condition that a neutralizing background implies.
    ///
    /// The `R⁻³` part of the Klopman–Ohno kernel diverges separately and more weakly; that is
    /// [`Self::klopman_ohno_tail`], which is a different switch because it is a different sum.
    pub ewald: bool,
    /// Add the analytic **Klopman–Ohno `R⁻³` tail** beyond the pair list. Default `true`.
    ///
    /// [`Self::ewald`] corrects the `1/R` channel exactly. But the pair list summed the full NDDO
    /// kernel `γ_η(R) = e²/√(R² + η²)`, and `γ_η − 1/R = −η²/(2R³) + …` was left truncated —
    /// `Σ_T |T|⁻³` diverges logarithmically in three dimensions, so the total energy drifted with
    /// [`Self::realspace_cutoff`] and converged to nothing. `false` restores the 0.2.1 behaviour.
    ///
    /// The counterpart of [`crate::pbc::PbcOptions::klopman_ohno_tail`], and it has to be set the
    /// same way: the two paths are checked against each other at Γ (`tests/pbc_hessian.rs`), and
    /// leaving one tailed and the other not put the analytic Hessian 2.1e-3 eV/Bohr² from its
    /// finite difference.
    ///
    /// See `crate::pbc::ewald::klopman_ohno_tail_matrix` — which is `pub(crate)`, so the
    /// derivation is in the source rather than the API docs.
    pub klopman_ohno_tail: bool,
    /// Distance (Bohr) beyond which an **image** pair's exchange contribution is dropped.
    /// `None` keeps all of it, which is right for a molecule and wrong for a periodic cell.
    ///
    /// See [`crate::hamiltonian::PairIntegral::exchange_scale`]: NDDO's two-centre exchange
    /// integral decays as `1/R` and is finite only because the density matrix element it
    /// contracts against decays. At Γ-only sampling that element does not decay, so the image
    /// sum diverges. This is an explicit approximation standing in for the k-point sampling
    /// that would make the density matrix decay on its own.
    pub exchange_cutoff: Option<f64>,
    /// Separation (Bohr) beyond which a pair's electrostatics is treated as a **monopole**
    /// rather than through the full Dewar-Sabelli-Klopman multipole block.
    ///
    /// `None` (the default) keeps every pair exact, which is what every result in this crate
    /// was validated against. Setting it is an explicit accuracy-for-speed trade: the neglected
    /// dipole and quadrupole channels fall as `(d/R)^2`, so the error shrinks quadratically
    /// with the cutoff, and [`crate::farfield`] measures it rather than bounding it by
    /// argument.
    ///
    /// The reason it helps is that the all-pairs Fock build is the measured bottleneck of a
    /// large run -- 62 % of a 1029-atom divide-and-conquer calculation -- and a monopole pair
    /// costs about a hundredth of a full block.
    pub multipole_cutoff: Option<f64>,
    /// Uniform external **electric field**, in eV per (e·Bohr), or `None` for no field.
    ///
    /// The energy becomes `E(F) = E₀ − μ·F` with `μ` this model's own dipole; see
    /// [`crate::dipole`] for the operator and the full sign convention. The gradient and the
    /// analytic Hessian both account for it: the force on atom `a` gains `+Q_a F`, and because
    /// the dipole operator is *linear* in the nuclear positions the field contributes nothing to
    /// the fixed-density second derivative — it reaches the Hessian only through the CPHF
    /// response.
    ///
    /// **Molecules only.** A cell plus a field is an error rather than an approximation: `F·R`
    /// is unbounded along a periodic direction. The periodic analogue is the clamped-ion field
    /// response behind [`crate::pbc::dielectric_tensor`].
    pub electric_field: Option<Vec3>,
}

impl Am1Options {
    /// The subset of these options the core Hamiltonian build needs.
    ///
    /// One conversion, so that a path which builds `H_core` for itself — the Hessian, the
    /// divide-and-conquer driver — cannot quietly disagree with the SCF about which corrections
    /// are switched on. Getting that wrong is invisible: the calculation runs and returns a
    /// number from a slightly different Hamiltonian.
    pub fn core_build(&self) -> crate::hamiltonian::CoreBuildOptions {
        crate::hamiltonian::CoreBuildOptions {
            exchange_cutoff: self.exchange_cutoff,
            use_ewald: self.ewald,
            klopman_ohno_tail: self.klopman_ohno_tail.then_some(self.realspace_cutoff),
            multipole_cutoff: self.multipole_cutoff,
            electric_field: self.electric_field,
        }
    }
}

impl Default for Am1Options {
    fn default() -> Self {
        Self {
            charge: 0.0,
            multiplicity: 1,
            reference: ScfReference::Auto,
            max_scf: 200,
            e_tol: 1.0e-8,
            p_tol: 1.0e-7,
            use_diis: true,
            accelerator: ScfAccelerator::AdiisCdiis,
            adiis_switch: 0.1,
            realspace_cutoff: 40.0,
            ewald: true,
            klopman_ohno_tail: true,
            exchange_cutoff: None,
            multipole_cutoff: None,
            electric_field: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Am1Result {
    pub density: Matrix,
    /// Spin density `P_α − P_β` (open-shell UHF only; `None` for RHF).
    pub spin_density: Option<Matrix>,
    /// Orbital energies (eV). For UHF these are the **α** channel; see [`Am1Result::beta`].
    pub mo_energies: Vec<f64>,
    /// MO coefficients, columns are orbitals. For UHF these are the **α** channel.
    pub mo_coeff: Matrix,
    /// Number of occupied orbitals — doubly occupied for RHF, α-occupied for UHF.
    pub n_occ: usize,
    /// The **β** spin channel, present only for an unrestricted run.
    ///
    /// UHF solves two eigenproblems; before 0.2.1 only the α one survived into the result, so a
    /// spin-polarized wavefunction could not be written out and the β frontier orbitals could
    /// not be reported at all.
    pub beta: Option<BetaOrbitals>,
    pub electronic_ev: f64,
    pub core_ev: f64,
    /// The **nuclear** half of the external-field interaction, `−F · Σ_a Z_a R_a`, in eV; zero
    /// without a field.
    ///
    /// Only the nuclear half, because the electronic half `+Tr[P h^F]` is already inside
    /// `electronic_ev` — it enters through `H_core`, which is where the field operator is added.
    /// Reporting it separately is what makes `total_ev = electronic_ev + core_ev +
    /// field_nuclear_ev` add up without the field term being invisible.
    pub field_nuclear_ev: f64,
    pub total_ev: f64,
    pub heat_of_formation_kcal: f64,
    pub charges: Vec<f64>,
    pub dipole_debye: Vec3,
    pub dipole_magnitude: f64,
    pub homo_ev: Option<f64>,
    pub lumo_ev: Option<f64>,
    /// β-channel frontier orbital energies (eV), for an unrestricted run only.
    pub homo_beta_ev: Option<f64>,
    pub lumo_beta_ev: Option<f64>,
    pub iterations: usize,
    pub converged: bool,
    /// True when the UHF (open-shell) path was used.
    pub unrestricted: bool,
}

pub struct Am1Calculator {
    pub params: Am1Parameters,
    pub options: Am1Options,
}

impl Am1Calculator {
    pub fn new(params: Am1Parameters) -> Self {
        Self {
            params,
            options: Am1Options::default(),
        }
    }
    pub fn with_options(params: Am1Parameters, options: Am1Options) -> Self {
        Self { params, options }
    }
    pub fn calculate(&self, molecule: &Molecule) -> Result<Am1Result> {
        run_am1(molecule, &self.params, &self.options)
    }
}

struct ScfState {
    density: Matrix,
    spin_density: Option<Matrix>,
    mo_energies: Vec<f64>,
    mo_coeff: Matrix,
    n_occ: usize,
    /// The β channel, for an unrestricted run. `None` for RHF, where the two channels coincide.
    beta: Option<BetaOrbitals>,
    electronic_ev: f64,
    converged: bool,
    iterations: usize,
    unrestricted: bool,
}

/// The β spin channel's orbitals.
///
/// UHF solves two eigenproblems and used to return only the α one, so a spin-polarized
/// wavefunction could not be written out, a β HOMO could not be reported, and the β orbital
/// energies — which are what a spin-polarized ionization potential needs — were simply gone.
#[derive(Clone, Debug)]
pub struct BetaOrbitals {
    pub energies: Vec<f64>,
    pub coeff: Matrix,
    pub n_occ: usize,
}

pub fn run_am1(
    molecule: &Molecule,
    params: &Am1Parameters,
    options: &Am1Options,
) -> Result<Am1Result> {
    if options.multiplicity < 1 {
        return Err(Am1Error::InvalidInput(
            "multiplicity must be >= 1".to_string(),
        ));
    }
    crate::linalg::enable_parallelism();
    // One pair list serves both the core Hamiltonian and the core-core repulsion. For a
    // molecule it is every pair; for a periodic cell it is every pair within the real-space
    // cutoff, including an atom with its own images, and the assembly below is then the
    // Gamma-point Bloch sum because e^{ik·T} = 1 at k = 0.
    let neighbors = crate::neighbors::NeighborList::build_screened(
        molecule,
        options.realspace_cutoff,
        options.multipole_cutoff,
    );
    let (basis, core) = {
        let _t = crate::timing::Timer::start("basis+core");
        let basis = Basis::build(molecule, params)?;
        let core = crate::hamiltonian::build_core_with_neighbors(
            molecule,
            &basis,
            params,
            &neighbors,
            options.core_build(),
        )?;
        (basis, core)
    };

    let mut n_elec = 0.0;
    for atom in &molecule.atoms {
        n_elec += params.element(atom.z)?.core_charge;
    }
    n_elec -= options.charge;
    let n_elec_int = n_elec.round() as i64;
    if (n_elec - n_elec_int as f64).abs() > 1.0e-6 || n_elec_int < 0 {
        return Err(Am1Error::InvalidInput(format!(
            "invalid electron count {n_elec}"
        )));
    }
    let n_unpaired = (options.multiplicity - 1) as i64;
    if (n_elec_int - n_unpaired) < 0 || (n_elec_int - n_unpaired) % 2 != 0 {
        return Err(Am1Error::InvalidInput(format!(
            "electron count {n_elec_int} is incompatible with multiplicity {} (need same parity)",
            options.multiplicity
        )));
    }
    let n_alpha = ((n_elec_int + n_unpaired) / 2) as usize;
    let n_beta = ((n_elec_int - n_unpaired) / 2) as usize;
    let closed_shell = n_alpha == n_beta;

    // The reference (restricted/unrestricted) is chosen independently of the multiplicity:
    // `Auto` keeps the historical rule (RHF closed shell, UHF open shell); `Restricted` demands
    // a closed shell (no ROHF here); `Unrestricted` runs UHF even for a singlet.
    let use_uhf = match options.reference {
        ScfReference::Auto => !closed_shell,
        ScfReference::Unrestricted => true,
        ScfReference::Restricted => {
            if !closed_shell {
                return Err(Am1Error::InvalidInput(format!(
                    "restricted (RHF) reference requires a closed shell, but multiplicity {} \
                     gives {n_alpha} α and {n_beta} β electrons; use the unrestricted (UHF) \
                     reference for open-shell systems",
                    options.multiplicity
                )));
            }
            false
        }
    };

    let state = if !use_uhf {
        rhf_loop(molecule, &basis, params, &core, n_alpha, options)?
    } else {
        uhf_loop(molecule, &basis, params, &core, n_alpha, n_beta, options)?
    };

    let core_ev = crate::repulsion::core_core_energy_with_neighbors(molecule, params, &neighbors)?;
    // The field's nuclear half. Its electronic half rode in through `H_core`, so this is all
    // that is left to add. See `crate::dipole` for the sign convention.
    let field_nuclear_ev = match options.electric_field {
        Some(f) => crate::dipole::field_core_energy(molecule, params, f)?,
        None => 0.0,
    };
    let total_ev = state.electronic_ev + core_ev + field_nuclear_ev;

    let mut e_isol_sum = 0.0;
    let mut eheat_sum = 0.0;
    for atom in &molecule.atoms {
        let e = params.element(atom.z)?;
        e_isol_sum += e.e_isol;
        eheat_sum += e.eheat_ev;
    }
    let heat_of_formation_kcal = (total_ev - e_isol_sum + eheat_sum) * EV_TO_KCAL;

    if !state.converged {
        return Err(Am1Error::ScfNotConverged {
            iterations: state.iterations,
            error: f64::NAN,
        });
    }

    // Mulliken net charges from the total density.
    let mut charges = vec![0.0; molecule.atoms.len()];
    for (ia, atom) in molecule.atoms.iter().enumerate() {
        let off = basis.atom_offset[ia];
        let n = basis.atom_norb[ia];
        let mut pop = 0.0;
        for mu in 0..n {
            pop += state.density[(off + mu, off + mu)];
        }
        charges[ia] = params.element(atom.z)?.core_charge - pop;
    }

    // Dipole: point-charge term + s–p hybrid polarization (both in e·Bohr).
    let mut dip = Vec3::zero();
    for (ia, atom) in molecule.atoms.iter().enumerate() {
        dip += atom.position * charges[ia];
        let elem = params.element(atom.z)?;
        if elem.has_p() {
            let off = basis.atom_offset[ia];
            let hyb = -2.0 * elem.dd;
            dip += Vec3::new(
                hyb * state.density[(off, off + 1)],
                hyb * state.density[(off, off + 2)],
                hyb * state.density[(off, off + 3)],
            );
        }
    }
    let dipole_debye = dip * AU_DIPOLE_TO_DEBYE;
    let dipole_magnitude = dipole_debye.norm();

    let nao = basis.nao;
    let homo_ev = (state.n_occ >= 1).then(|| state.mo_energies[state.n_occ - 1]);
    let lumo_ev = (state.n_occ < nao).then(|| state.mo_energies[state.n_occ]);
    let (homo_beta_ev, lumo_beta_ev) = match &state.beta {
        Some(b) => (
            (b.n_occ >= 1).then(|| b.energies[b.n_occ - 1]),
            (b.n_occ < nao).then(|| b.energies[b.n_occ]),
        ),
        None => (None, None),
    };

    // No `timing::report` here. Reporting clears the accumulator, so a library function that
    // reported would truncate the measurement at its own boundary — profiling a gradient printed
    // only the SCF phases for exactly that reason. The top-level caller reports; see
    // `crate::timing`.

    Ok(Am1Result {
        density: state.density,
        spin_density: state.spin_density,
        mo_energies: state.mo_energies,
        mo_coeff: state.mo_coeff,
        n_occ: state.n_occ,
        beta: state.beta,
        electronic_ev: state.electronic_ev,
        core_ev,
        field_nuclear_ev,
        total_ev,
        heat_of_formation_kcal,
        charges,
        dipole_debye,
        dipole_magnitude,
        homo_ev,
        lumo_ev,
        homo_beta_ev,
        lumo_beta_ev,
        iterations: state.iterations,
        converged: state.converged,
        unrestricted: state.unrestricted,
    })
}

/// **Superposition of Atomic Densities (SAD)** initial guess: a block-diagonal density built
/// from each atom's spherically-averaged neutral valence configuration `s^{min(2,Zv)} p^{…}`.
/// In a minimal valence NDDO basis a free atom's density *is* diagonal (spherical symmetry), so
/// this superposition is the exact isolated-atom density and a far better SCF start than the
/// bare core (zero-density) guess — fewer iterations, more robust for larger systems.
fn sad_density(molecule: &Molecule, basis: &Basis, params: &Am1Parameters) -> Result<Matrix> {
    let nao = basis.nao;
    let mut p = Matrix::zeros(nao, nao);
    for (ia, atom) in molecule.atoms.iter().enumerate() {
        let elem = params.element(atom.z)?;
        let off = basis.atom_offset[ia];
        let n = basis.atom_norb[ia];
        let zv = elem.core_charge;
        let n_s = zv.min(2.0); // s shell fills first (up to 2)
        let n_p = (zv - n_s).max(0.0); // remaining valence electrons in the p shell
        p[(off, off)] = n_s;
        if n == 4 {
            let per_p = n_p / 3.0; // spherically averaged over px, py, pz
            for k in 1..4 {
                p[(off + k, off + k)] = per_p;
            }
        }
    }
    Ok(p)
}

/// Build a density `P = w Σ_{k<n_occ} c_k c_kᵀ` from MO coefficients (`w` = 2 for RHF, 1 for UHF).
///
/// Written as one `C_occ · C_occᵀ` product rather than the obvious triple loop. `Matrix` is
/// row-major, so accumulating over MO index `k` innermost walks memory with a stride of
/// `nao` — on a 798-AO system that is a cache miss per multiply, and profiling put this
/// single function at 6.1 s of a 9 s single point, well ahead of the Fock build (0.6 s) and
/// the eigendecomposition (1.3 s).
fn density_from_coeff(c: &Matrix, n_occ: usize, weight: f64) -> Matrix {
    let nao = c.rows;
    if n_occ == 0 || nao == 0 {
        return Matrix::zeros(nao, nao);
    }
    // Copy out the occupied columns so the product is over a compact block.
    let mut occ = Matrix::zeros(nao, n_occ);
    for mu in 0..nao {
        for k in 0..n_occ {
            occ[(mu, k)] = c[(mu, k)];
        }
    }
    // Transpose-free: materializing `occᵀ` is another `nao × n_occ` allocation and copy, and at
    // 1602 AOs that is 14 MB built and thrown away on every SCF iteration.
    let mut p = occ.matmul_transpose(&occ);
    if weight != 1.0 {
        for x in p.as_mut_slice() {
            *x *= weight;
        }
    }
    p
}

/// DIIS error `[F, P] = FP − PF`.
///
/// Both operands are symmetric, so `PF = (FP)ᵀ` and one matrix product suffices: the second
/// was recomputing a transpose the hard way, at O(nao³). NDDO's orthonormal AO basis is what
/// makes this the plain commutator — there is no `S` to sandwich.
fn commutator(f: &Matrix, p: &Matrix) -> Matrix {
    let mut e = f.matmul(p);
    let n = e.rows;
    debug_assert_eq!(n, e.cols);
    for i in 0..n {
        for j in (i + 1)..n {
            let (a, b) = (e[(i, j)], e[(j, i)]);
            e[(i, j)] = a - b;
            e[(j, i)] = b - a;
        }
        e[(i, i)] = 0.0;
    }
    e
}

fn rms_diff(a: &Matrix, b: &Matrix) -> f64 {
    let n = a.as_slice().len().max(1);
    (a.as_slice()
        .iter()
        .zip(b.as_slice())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        / n as f64)
        .sqrt()
}

fn rhf_loop(
    molecule: &Molecule,
    basis: &Basis,
    params: &Am1Parameters,
    core: &CoreHamiltonian,
    n_occ: usize,
    options: &Am1Options,
) -> Result<ScfState> {
    let nao = basis.nao;
    let mut density = sad_density(molecule, basis, params)?; // SAD initial guess
    let mut e_old = 0.0;
    let mut mo_energies = vec![0.0; nao];
    let mut mo_coeff = Matrix::zeros(nao, nao);
    let mut converged = false;
    let mut iterations = 0;
    // Packed triangles, not dense matrices: see `Tri`. At 1602 AOs the three depth-8 histories
    // go from 492 MB to 246 MB, and to 164 MB for a CDIIS-only run, where the density history
    // below is not built at all.
    let mut diis_f: Vec<Vec<f64>> = Vec::new();
    let mut diis_e: Vec<Vec<f64>> = Vec::new();
    let mut diis_d: Vec<Vec<f64>> = Vec::new();
    let max_diis = 8;
    let accel = if options.use_diis {
        options.accelerator
    } else {
        ScfAccelerator::None
    };
    // Only A-DIIS reads the density history. Keeping it for the other accelerators was a third
    // of the history for nothing.
    let needs_density_history = accel == ScfAccelerator::AdiisCdiis;

    for iter in 0..options.max_scf {
        iterations = iter + 1;
        let f = {
            let _t = crate::timing::Timer::start("scf:fock");
            build_fock(molecule, basis, params, core, &density)?
        };
        let e_elec = 0.5 * (density.frobenius_dot(&core.h_core) + density.frobenius_dot(&f));
        let err = {
            let _t = crate::timing::Timer::start("scf:commutator");
            commutator(&f, &density)
        };
        let err_norm = err.as_slice().iter().map(|x| x * x).sum::<f64>().sqrt();

        // History (Fock, commutator, density) for CDIIS / A-DIIS, packed.
        diis_f.push(Tri::Symmetric.pack(&f));
        diis_e.push(Tri::Antisymmetric.pack(&err));
        if needs_density_history {
            diis_d.push(Tri::Symmetric.pack(&density));
        }
        if diis_f.len() > max_diis {
            diis_f.remove(0);
            diis_e.remove(0);
            if !diis_d.is_empty() {
                diis_d.remove(0);
            }
        }

        let f_use = {
            let _t = crate::timing::Timer::start("scf:accel");
            match accel {
                ScfAccelerator::None => f,
                ScfAccelerator::Cdiis => {
                    diis_extrapolate_packed(&diis_f, &diis_e, nao).unwrap_or(f)
                }
                ScfAccelerator::AdiisCdiis => {
                    if err_norm > options.adiis_switch {
                        adiis_extrapolate_packed(&diis_d, &diis_f, nao).unwrap_or(f)
                    } else {
                        diis_extrapolate_packed(&diis_f, &diis_e, nao).unwrap_or(f)
                    }
                }
            }
        };
        let (eps, c) = {
            let _t = crate::timing::Timer::start("scf:eigen");
            symmetric_eigen(&f_use)?
        };
        let p_new = {
            let _t = crate::timing::Timer::start("scf:density");
            density_from_coeff(&c, n_occ, 2.0)
        };
        let dp = rms_diff(&p_new, &density);
        let de = (e_elec - e_old).abs();

        mo_energies = eps;
        mo_coeff = c;
        density = p_new;
        e_old = e_elec;
        if iter > 0 && de < options.e_tol && (dp < options.p_tol || err_norm < 1.0e-7) {
            converged = true;
            break;
        }
    }
    let f_final = build_fock(molecule, basis, params, core, &density)?;
    let electronic_ev = 0.5
        * (density.frobenius_dot(&core.h_core) + density.frobenius_dot(&f_final))
        + crate::fock::long_range_energy_term(molecule, basis, params, core, &density)?;

    Ok(ScfState {
        density,
        spin_density: None,
        mo_energies,
        mo_coeff,
        n_occ,
        beta: None,
        electronic_ev,
        converged,
        iterations,
        unrestricted: false,
    })
}

#[allow(clippy::too_many_arguments)]
fn uhf_loop(
    molecule: &Molecule,
    basis: &Basis,
    params: &Am1Parameters,
    core: &CoreHamiltonian,
    n_alpha: usize,
    n_beta: usize,
    options: &Am1Options,
) -> Result<ScfState> {
    let nao = basis.nao;
    // SAD guess split by spin population; the different α/β aufbau counts break spin symmetry.
    let sad = sad_density(molecule, basis, params)?;
    let n_tot = (n_alpha + n_beta).max(1) as f64;
    let (fa, fb) = (n_alpha as f64 / n_tot, n_beta as f64 / n_tot);
    let mut pa = sad.clone();
    for v in pa.as_mut_slice() {
        *v *= fa;
    }
    let mut pb = sad;
    for v in pb.as_mut_slice() {
        *v *= fb;
    }
    let mut e_old = 0.0;
    let mut eps_a = vec![0.0; nao];
    let mut c_a = Matrix::zeros(nao, nao);
    let mut eps_b = vec![0.0; nao];
    let mut c_b = Matrix::zeros(nao, nao);
    let mut converged = false;
    let mut iterations = 0;

    // Packed triangles, as in `rhf_loop` — see `Tri`. The error entry is the two spin
    // commutators' strict triangles laid end to end, which preserves the Frobenius product of
    // the stacked matrix exactly because the product of a block stack is the sum over blocks.
    let mut hist_fa: Vec<Vec<f64>> = Vec::new();
    let mut hist_fb: Vec<Vec<f64>> = Vec::new();
    let mut hist_err: Vec<Vec<f64>> = Vec::new();
    let max_diis = 8;

    for iter in 0..options.max_scf {
        iterations = iter + 1;
        let mut p_tot = pa.clone();
        for (t, b) in p_tot.as_mut_slice().iter_mut().zip(pb.as_slice()) {
            *t += *b;
        }
        let fa = build_fock_spin(molecule, basis, params, core, &p_tot, &pa)?;
        let fb = build_fock_spin(molecule, basis, params, core, &p_tot, &pb)?;

        let e_elec = 0.5
            * (p_tot.frobenius_dot(&core.h_core) + pa.frobenius_dot(&fa) + pb.frobenius_dot(&fb));

        // Combined DIIS error = [F_a,P_a] ⊕ [F_b,P_b], both antisymmetric.
        let ea = commutator(&fa, &pa);
        let eb = commutator(&fb, &pb);
        let err_norm = (ea.as_slice().iter().map(|x| x * x).sum::<f64>()
            + eb.as_slice().iter().map(|x| x * x).sum::<f64>())
        .sqrt();

        let (fa_use, fb_use) = if options.use_diis {
            let mut err = Tri::Antisymmetric.pack(&ea);
            err.extend_from_slice(&Tri::Antisymmetric.pack(&eb));
            hist_fa.push(Tri::Symmetric.pack(&fa));
            hist_fb.push(Tri::Symmetric.pack(&fb));
            hist_err.push(err);
            if hist_fa.len() > max_diis {
                hist_fa.remove(0);
                hist_fb.remove(0);
                hist_err.remove(0);
            }
            match diis_coeffs_packed(&hist_err, nao) {
                Some(coeffs) => (
                    combine_packed(&hist_fa, &coeffs, Tri::Symmetric, nao),
                    combine_packed(&hist_fb, &coeffs, Tri::Symmetric, nao),
                ),
                None => (fa, fb),
            }
        } else {
            (fa, fb)
        };

        let (ea_eps, ca) = symmetric_eigen(&fa_use)?;
        let (eb_eps, cb) = symmetric_eigen(&fb_use)?;
        let pa_new = density_from_coeff(&ca, n_alpha, 1.0);
        let pb_new = density_from_coeff(&cb, n_beta, 1.0);

        let dp = rms_diff(&pa_new, &pa) + rms_diff(&pb_new, &pb);
        let de = (e_elec - e_old).abs();
        eps_a = ea_eps;
        c_a = ca;
        eps_b = eb_eps;
        c_b = cb;
        pa = pa_new;
        pb = pb_new;
        e_old = e_elec;
        if iter > 0 && de < options.e_tol && (dp < options.p_tol || err_norm < 1.0e-7) {
            converged = true;
            break;
        }
    }

    let mut density = pa.clone();
    for (t, b) in density.as_mut_slice().iter_mut().zip(pb.as_slice()) {
        *t += *b;
    }
    let mut spin = pa.clone();
    for (s, b) in spin.as_mut_slice().iter_mut().zip(pb.as_slice()) {
        *s -= *b;
    }
    // Final energy.
    let fa = build_fock_spin(molecule, basis, params, core, &density, &pa)?;
    let fb = build_fock_spin(molecule, basis, params, core, &density, &pb)?;
    let electronic_ev = 0.5
        * (density.frobenius_dot(&core.h_core) + pa.frobenius_dot(&fa) + pb.frobenius_dot(&fb))
        + crate::fock::long_range_energy_term(molecule, basis, params, core, &density)?;

    Ok(ScfState {
        density,
        spin_density: Some(spin),
        mo_energies: eps_a,
        mo_coeff: c_a,
        n_occ: n_alpha,
        beta: Some(BetaOrbitals {
            energies: eps_b,
            coeff: c_b,
            n_occ: n_beta,
        }),
        electronic_ev,
        converged,
        iterations,
        unrestricted: true,
    })
}

/// A DIIS history entry stored as a packed triangle.
///
/// # Why this is not a `Vec<Matrix>`
///
/// `rhf_loop` keeps three depth-8 histories — Fock, commutator error, density — and at 1602 AOs
/// (an 801-atom water cluster) twenty-four dense `nao²` matrices are **492 MB**, against a
/// measured 877 MB peak for the whole calculation. It was the single largest term, and more than
/// half of it was redundant: every matrix in all three histories is either symmetric or
/// antisymmetric, so one triangle determines the other.
///
/// `F` and `P` are symmetric. The error `[F, P] = FP − PF` is **anti**symmetric — its transpose
/// is `PF − FP` — so its diagonal is identically zero and only the *strict* triangle is stored.
/// Nothing is approximated by any of this.
///
/// Packing halves the history exactly; gating the density history on the accelerator that
/// actually reads it (`AdiisCdiis`) removes a third of what remains.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Tri {
    /// Upper triangle including the diagonal, `n(n+1)/2` entries.
    Symmetric,
    /// Strict upper triangle, `n(n−1)/2` entries; the diagonal is known to be zero.
    Antisymmetric,
}

impl Tri {
    fn len(self, n: usize) -> usize {
        match self {
            Tri::Symmetric => n * (n + 1) / 2,
            Tri::Antisymmetric => n * (n - 1) / 2,
        }
    }

    /// Pack the upper triangle of `m` row by row.
    fn pack(self, m: &Matrix) -> Vec<f64> {
        let n = m.rows;
        let mut out = Vec::with_capacity(self.len(n));
        for i in 0..n {
            let start = if self == Tri::Symmetric { i } else { i + 1 };
            for j in start..n {
                out.push(m[(i, j)]);
            }
        }
        out
    }

    /// Rebuild the full matrix, mirroring with the sign this symmetry implies.
    fn unpack(self, packed: &[f64], n: usize) -> Matrix {
        let mut out = Matrix::zeros(n, n);
        let mut k = 0;
        for i in 0..n {
            let start = if self == Tri::Symmetric { i } else { i + 1 };
            for j in start..n {
                let v = packed[k];
                k += 1;
                out[(i, j)] = v;
                if i != j {
                    out[(j, i)] = if self == Tri::Symmetric { v } else { -v };
                }
            }
        }
        out
    }

    /// The full-matrix Frobenius product `Σ_ij A_ij B_ij` from the packed triangles.
    ///
    /// Off-diagonal entries stand for two elements of the full matrix, and for an antisymmetric
    /// pair `A_ji B_ji = (−A_ij)(−B_ij) = A_ij B_ij`, so the multiplicity is `2` in both cases.
    /// Only the diagonal, present for `Symmetric` only, counts once.
    fn dot(self, a: &[f64], b: &[f64], n: usize) -> f64 {
        let mut acc = 0.0;
        match self {
            Tri::Antisymmetric => {
                for (x, y) in a.iter().zip(b) {
                    acc += x * y;
                }
                2.0 * acc
            }
            Tri::Symmetric => {
                let mut diag = 0.0;
                let mut k = 0;
                for i in 0..n {
                    diag += a[k] * b[k]; // (i, i) is first in each row
                    k += 1;
                    for _ in (i + 1)..n {
                        acc += a[k] * b[k];
                        k += 1;
                    }
                }
                2.0 * acc + diag
            }
        }
    }
}

/// `Σ_i c_i H_i` over packed entries, unpacked once at the end.
fn combine_packed(history: &[Vec<f64>], coeffs: &[f64], tri: Tri, n: usize) -> Matrix {
    let mut acc = vec![0.0; tri.len(n)];
    for (h, ci) in history.iter().zip(coeffs) {
        for (o, v) in acc.iter_mut().zip(h) {
            *o += ci * v;
        }
    }
    tri.unpack(&acc, n)
}

/// Pulay DIIS coefficients from packed error vectors.
fn diis_coeffs_packed(es: &[Vec<f64>], n: usize) -> Option<Vec<f64>> {
    let m = es.len();
    if m < 2 {
        return None;
    }
    let dim = m + 1;
    let mut b = Matrix::zeros(dim, dim);
    for i in 0..m {
        for j in 0..m {
            b[(i, j)] = Tri::Antisymmetric.dot(&es[i], &es[j], n);
        }
        b[(i, m)] = -1.0;
        b[(m, i)] = -1.0;
    }
    let mut rhs = vec![0.0; dim];
    rhs[m] = -1.0;
    // The DIIS matrix (a small bordered saddle-point system) becomes singular near convergence
    // when the error vectors turn linearly dependent. A pivot-guarded Gaussian elimination
    // returns `None` there so the caller falls back to the plain Fock — faer's LU instead
    // returns a degenerate solution that derails DIIS. The heavy O(n³) eigendecomposition still
    // uses faer; only this tiny solve is bespoke.
    solve_bordered_small(&b, &rhs)
}

/// CDIIS on packed histories: coefficients from the errors, applied to the Focks.
fn diis_extrapolate_packed(fs: &[Vec<f64>], es: &[Vec<f64>], n: usize) -> Option<Matrix> {
    let coeffs = diis_coeffs_packed(es, n)?;
    Some(combine_packed(fs, &coeffs, Tri::Symmetric, n))
}

/// A-DIIS on packed histories. Same quadratic as [`adiis_extrapolate`], same expansion of the
/// differences through the Gram matrix; only the storage differs.
fn adiis_extrapolate_packed(
    densities: &[Vec<f64>],
    focks: &[Vec<f64>],
    n: usize,
) -> Option<Matrix> {
    let m = densities.len();
    if m < 2 {
        return None;
    }
    use rayon::prelude::*;
    let last = m - 1;
    let gram: Vec<Vec<f64>> = densities
        .par_iter()
        .map(|di| {
            focks
                .iter()
                .map(|fj| Tri::Symmetric.dot(di, fj, n))
                .collect()
        })
        .collect();
    let d: Vec<f64> = (0..m).map(|i| gram[i][last] - gram[last][last]).collect();
    let s: Vec<Vec<f64>> = (0..m)
        .map(|i| {
            (0..m)
                .map(|j| gram[i][j] - gram[i][last] - gram[last][j] + gram[last][last])
                .collect()
        })
        .collect();
    let c = solve_adiis_simplex(&d, &s);
    Some(combine_packed(focks, &c, Tri::Symmetric, n))
}

/// Gaussian elimination with partial pivoting for the small DIIS system; returns `None`
/// if the matrix is (near-)singular.
pub(crate) fn solve_bordered_small(a: &Matrix, b: &[f64]) -> Option<Vec<f64>> {
    let n = a.rows;
    let mut m = a.clone();
    let mut rhs = b.to_vec();
    for col in 0..n {
        let mut pivot = col;
        let mut best = m[(col, col)].abs();
        for row in (col + 1)..n {
            let v = m[(row, col)].abs();
            if v > best {
                best = v;
                pivot = row;
            }
        }
        if best < 1.0e-12 {
            return None;
        }
        if pivot != col {
            for j in 0..n {
                let t = m[(col, j)];
                m[(col, j)] = m[(pivot, j)];
                m[(pivot, j)] = t;
            }
            rhs.swap(col, pivot);
        }
        for row in (col + 1)..n {
            let factor = m[(row, col)] / m[(col, col)];
            if factor == 0.0 {
                continue;
            }
            for j in col..n {
                let v = m[(col, j)];
                m[(row, j)] -= factor * v;
            }
            rhs[row] -= factor * rhs[col];
        }
    }
    let mut x = vec![0.0; n];
    for col in (0..n).rev() {
        let mut sum = rhs[col];
        for j in (col + 1)..n {
            sum -= m[(col, j)] * x[j];
        }
        x[col] = sum / m[(col, col)];
    }
    Some(x)
}

/// Projected-gradient minimization of the A-DIIS quadratic on the probability simplex.
fn solve_adiis_simplex(d: &[f64], s: &[Vec<f64>]) -> Vec<f64> {
    let n = d.len();
    // Lipschitz estimate for the step size from (S + Sᵀ).
    let mut l: f64 = 1.0e-12;
    for i in 0..n {
        let mut row = 0.0;
        for j in 0..n {
            row += (s[i][j] + s[j][i]).abs();
        }
        l = l.max(row);
    }
    let lr = 1.0 / l;
    // Start from the latest point (all weight on the newest Fock/density).
    let mut c = vec![0.0; n];
    c[n - 1] = 1.0;
    for _ in 0..400 {
        // grad_k = 2 d_k + Σ_j (s_kj + s_jk) c_j
        let mut g = vec![0.0; n];
        for k in 0..n {
            let mut acc = 2.0 * d[k];
            for j in 0..n {
                acc += (s[k][j] + s[j][k]) * c[j];
            }
            g[k] = acc;
        }
        let trial: Vec<f64> = (0..n).map(|i| c[i] - lr * g[i]).collect();
        let proj = simplex_project(&trial);
        let mut delta = 0.0;
        for i in 0..n {
            delta += (proj[i] - c[i]).abs();
        }
        c = proj;
        if delta < 1.0e-12 {
            break;
        }
    }
    c
}

/// Euclidean projection of `v` onto the probability simplex `{c ≥ 0, Σc = 1}`.
///
/// Duchi *et al.*'s algorithm: sort descending, walk the prefix sums, and keep the threshold from
/// the last index where `u_j` still exceeds it. That index (`ρ` in the paper) is not needed once
/// the threshold is known, so only the threshold is carried.
fn simplex_project(v: &[f64]) -> Vec<f64> {
    let mut u = v.to_vec();
    // `total_cmp` rather than `partial_cmp(..).unwrap_or(Equal)`: it is a total order on every
    // `f64` including NaN, so a NaN cannot make the sort inconsistent — which for `sort_by` is a
    // contract violation, not merely an odd ordering.
    u.sort_by(|a, b| b.total_cmp(a));
    let mut css = 0.0;
    let mut theta = 0.0;
    for (j, &uj) in u.iter().enumerate() {
        css += uj;
        let t = (css - 1.0) / (j as f64 + 1.0);
        if uj - t > 0.0 {
            theta = t;
        }
    }
    v.iter().map(|&vi| (vi - theta).max(0.0)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(xyz: &str, charge: f64) -> Am1Result {
        let mol = Molecule::from_xyz_str(xyz, charge).unwrap();
        let params = Am1Parameters::standard().unwrap();
        run_am1(&mol, &params, &Am1Options::default()).unwrap()
    }

    fn seeded(n: usize, seed: u64) -> Matrix {
        let mut m = Matrix::zeros(n, n);
        let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        for v in m.as_mut_slice() {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *v = ((s >> 11) as f64 / (1u64 << 53) as f64) - 0.5;
        }
        m
    }

    /// The packed DIIS history must lose nothing.
    ///
    /// Halving the history's memory rests on two claims: that `F` and `P` are symmetric, and
    /// that `[F, P]` is antisymmetric *with a zero diagonal* — the packing drops that diagonal
    /// outright, so if `commutator` ever stopped zeroing it (`scf.rs`'s `e[(i, i)] = 0.0`) the
    /// DIIS error would silently lose a term. This checks the round trip and the Frobenius
    /// product against the dense original, which is what the extrapolation actually consumes.
    #[test]
    fn packing_a_diis_history_preserves_it_exactly() {
        let n = 9;
        // Symmetric operands, as F and P are.
        let raw = seeded(n, 5);
        let mut f = Matrix::zeros(n, n);
        let mut p = Matrix::zeros(n, n);
        let raw2 = seeded(n, 9);
        for i in 0..n {
            for j in 0..n {
                f[(i, j)] = raw[(i, j)] + raw[(j, i)];
                p[(i, j)] = raw2[(i, j)] + raw2[(j, i)];
            }
        }

        for m in [&f, &p] {
            let packed = Tri::Symmetric.pack(m);
            assert_eq!(packed.len(), Tri::Symmetric.len(n));
            let back = Tri::Symmetric.unpack(&packed, n);
            assert_eq!(&back, m, "symmetric round trip");
        }

        let e1 = commutator(&f, &p);
        let e2 = commutator(&p, &f);
        for e in [&e1, &e2] {
            for i in 0..n {
                assert_eq!(e[(i, i)], 0.0, "the commutator diagonal must be zero");
            }
            let packed = Tri::Antisymmetric.pack(e);
            assert_eq!(packed.len(), Tri::Antisymmetric.len(n));
            assert_eq!(&Tri::Antisymmetric.unpack(&packed, n), e, "anti round trip");
        }

        // The quantity DIIS actually forms: <e_i, e_j> and <D_i, F_j> over the full matrices.
        let (pe1, pe2) = (Tri::Antisymmetric.pack(&e1), Tri::Antisymmetric.pack(&e2));
        let want = e1.frobenius_dot(&e2);
        let got = Tri::Antisymmetric.dot(&pe1, &pe2, n);
        assert!(
            (want - got).abs() < 1.0e-12 * want.abs().max(1.0),
            "antisymmetric Frobenius product {got} against {want}"
        );

        let (pf, pp) = (Tri::Symmetric.pack(&f), Tri::Symmetric.pack(&p));
        let want = f.frobenius_dot(&p);
        let got = Tri::Symmetric.dot(&pf, &pp, n);
        assert!(
            (want - got).abs() < 1.0e-12 * want.abs().max(1.0),
            "symmetric Frobenius product {got} against {want}"
        );
    }

    fn run_mult(xyz: &str, charge: f64, mult: usize) -> Am1Result {
        let mol = Molecule::from_xyz_str(xyz, charge).unwrap();
        let params = Am1Parameters::standard().unwrap();
        let opts = Am1Options {
            charge,
            multiplicity: mult,
            ..Am1Options::default()
        };
        run_am1(&mol, &params, &opts).unwrap()
    }

    #[test]
    fn water_heat_of_formation() {
        let xyz =
            "3\nwater\nO 0.0000 0.0000 0.0000\nH 0.9584 0.0000 0.0000\nH -0.2400 0.9278 0.0000\n";
        let r = run(xyz, 0.0);
        eprintln!(
            "H2O: dHf={:.3} kcal/mol  elec={:.4} eV core={:.4} eV  dipole={:.3} D  charges={:?}  iters={}",
            r.heat_of_formation_kcal, r.electronic_ev, r.core_ev, r.dipole_magnitude, r.charges, r.iterations
        );
        assert!(r.converged);
        assert!((r.heat_of_formation_kcal - (-59.24)).abs() < 0.5);
        assert!((r.dipole_magnitude - 1.86).abs() < 0.15);
        let qsum: f64 = r.charges.iter().sum();
        assert!(qsum.abs() < 1e-6);
        assert!(r.charges[0] < 0.0 && r.charges[1] > 0.0);
    }

    #[test]
    fn water_no_diis_debug() {
        let xyz =
            "3\nwater\nO 0.0000 0.0000 0.0000\nH 0.9584 0.0000 0.0000\nH -0.2400 0.9278 0.0000\n";
        let mol = Molecule::from_xyz_str(xyz, 0.0).unwrap();
        let params = Am1Parameters::standard().unwrap();
        let opts = Am1Options {
            use_diis: false,
            max_scf: 500,
            ..Am1Options::default()
        };
        let r = run_am1(&mol, &params, &opts).unwrap();
        eprintln!(
            "H2O(no-diis): dHf={:.3} conv={} iters={}",
            r.heat_of_formation_kcal, r.converged, r.iterations
        );
    }

    #[test]
    fn methane_heat_of_formation() {
        let xyz = "5\nmethane\nC 0.0000 0.0000 0.0000\nH 0.6276 0.6276 0.6276\nH -0.6276 -0.6276 0.6276\nH -0.6276 0.6276 -0.6276\nH 0.6276 -0.6276 -0.6276\n";
        let r = run(xyz, 0.0);
        eprintln!(
            "CH4: dHf={:.3} kcal/mol dipole={:.3} D iters={}",
            r.heat_of_formation_kcal, r.dipole_magnitude, r.iterations
        );
        assert!(r.converged);
    }

    #[test]
    fn accelerators_agree_on_energy() {
        // A-DIIS→CDIIS, plain CDIIS, and no acceleration must reach the same converged
        // energy (same SCF fixed point); the hybrid should not need more iterations.
        let xyz =
            "4\nformaldehyde\nC 0.0 0.0 0.0\nO 0.0 0.0 1.21\nH 0.94 0.0 -0.54\nH -0.94 0.0 -0.54\n";
        let mol = Molecule::from_xyz_str(xyz, 0.0).unwrap();
        let params = Am1Parameters::standard().unwrap();
        let run = |acc: ScfAccelerator| {
            let opts = Am1Options {
                accelerator: acc,
                ..Am1Options::default()
            };
            run_am1(&mol, &params, &opts).unwrap()
        };
        let hybrid = run(ScfAccelerator::AdiisCdiis);
        let cdiis = run(ScfAccelerator::Cdiis);
        let none = run(ScfAccelerator::None);
        eprintln!(
            "iters: hybrid={} cdiis={} none={}  E(hybrid)={:.6}",
            hybrid.iterations, cdiis.iterations, none.iterations, hybrid.total_ev
        );
        assert!((hybrid.total_ev - cdiis.total_ev).abs() < 1e-6);
        assert!((hybrid.total_ev - none.total_ev).abs() < 1e-6);
        // Both accelerated paths should be far faster than plain iteration.
        assert!(hybrid.iterations < none.iterations);
    }

    #[test]
    fn methyl_radical_uhf() {
        // Planar CH3 radical (doublet): UHF must converge, be open-shell, and carry net spin.
        let xyz = "4\nmethyl\nC 0.0 0.0 0.0\nH 1.079 0.0 0.0\nH -0.5395 0.9344 0.0\nH -0.5395 -0.9344 0.0\n";
        let r = run_mult(xyz, 0.0, 2);
        eprintln!(
            "CH3.: dHf={:.3} kcal/mol unrestricted={} iters={}",
            r.heat_of_formation_kcal, r.unrestricted, r.iterations
        );
        assert!(r.converged);
        assert!(r.unrestricted);
        // Total spin population (∫ P_α − P_β) should be ≈ 1 unpaired electron.
        let spin = r.spin_density.as_ref().unwrap();
        let n_spin: f64 = (0..spin.rows).map(|i| spin[(i, i)]).sum();
        assert!((n_spin - 1.0).abs() < 1e-6, "net spin {n_spin}");
        // Published AM1 ΔHf(CH3·) ≈ +32 kcal/mol.
        assert!((r.heat_of_formation_kcal - 32.0).abs() < 6.0);
    }

    #[test]
    fn forced_uhf_singlet_matches_rhf() {
        // A closed-shell singlet run as UHF must take the unrestricted path yet converge to the
        // RHF energy (a symmetric guess yields zero spin density).
        let xyz =
            "3\nwater\nO 0.0000 0.0000 0.0000\nH 0.9584 0.0000 0.0000\nH -0.2400 0.9278 0.0000\n";
        let mol = Molecule::from_xyz_str(xyz, 0.0).unwrap();
        let params = Am1Parameters::standard().unwrap();
        let rhf = run_am1(&mol, &params, &Am1Options::default()).unwrap();
        let uhf = run_am1(
            &mol,
            &params,
            &Am1Options {
                reference: ScfReference::Unrestricted,
                ..Am1Options::default()
            },
        )
        .unwrap();
        assert!(!rhf.unrestricted);
        assert!(uhf.unrestricted, "forced reference did not select UHF");
        assert!(
            (rhf.total_ev - uhf.total_ev).abs() < 1e-6,
            "UHF singlet != RHF energy"
        );
        // Net spin of a (symmetric) singlet must be ≈ 0.
        let spin = uhf.spin_density.as_ref().unwrap();
        let n_spin: f64 = (0..spin.rows).map(|i| spin[(i, i)]).sum();
        assert!(n_spin.abs() < 1e-6, "unexpected net spin {n_spin}");
    }

    #[test]
    fn forced_rhf_singlet_matches_auto() {
        // Explicitly restricting a closed shell is identical to Auto.
        let xyz =
            "3\nwater\nO 0.0000 0.0000 0.0000\nH 0.9584 0.0000 0.0000\nH -0.2400 0.9278 0.0000\n";
        let mol = Molecule::from_xyz_str(xyz, 0.0).unwrap();
        let params = Am1Parameters::standard().unwrap();
        let auto = run_am1(&mol, &params, &Am1Options::default()).unwrap();
        let rhf = run_am1(
            &mol,
            &params,
            &Am1Options {
                reference: ScfReference::Restricted,
                ..Am1Options::default()
            },
        )
        .unwrap();
        assert!(!rhf.unrestricted);
        assert!((auto.total_ev - rhf.total_ev).abs() < 1e-12);
    }

    #[test]
    fn restricted_rejects_open_shell() {
        // RHF cannot represent an open shell (no ROHF): a doublet request must be an error.
        let xyz = "4\nmethyl\nC 0.0 0.0 0.0\nH 1.079 0.0 0.0\nH -0.5395 0.9344 0.0\nH -0.5395 -0.9344 0.0\n";
        let mol = Molecule::from_xyz_str(xyz, 0.0).unwrap();
        let params = Am1Parameters::standard().unwrap();
        let res = run_am1(
            &mol,
            &params,
            &Am1Options {
                multiplicity: 2,
                reference: ScfReference::Restricted,
                ..Am1Options::default()
            },
        );
        assert!(
            res.is_err(),
            "restricted open-shell request should be rejected"
        );
    }
}

// SPDX-License-Identifier: GPL-3.0-or-later

//! Divide-and-conquer SCF (Yang 1991; Yang & Lee 1995; Akama, Kobayashi & Nakai 2007).
//!
//! The cost of an SCF is dominated by one `O(N³)` diagonalization of an `N × N` Fock matrix.
//! Divide-and-conquer replaces it with many small ones: the atoms are split into disjoint
//! **core** regions, each core is padded with a **buffer** of nearby atoms, and each resulting
//! subsystem is diagonalized on its own. Because the subsystem size stays fixed as the molecule
//! grows, `Σ_α n_α³` grows linearly in `N` rather than cubically.
//!
//! Three things have to be right for that to be a method rather than a fragmentation heuristic.
//!
//! **The pieces have to add up to one density.** Each subsystem contributes to the global
//! density matrix with the Yang partition weight
//!
//! ```text
//! p^α_μν = ½ (d^α_μ + d^α_ν),     d^α_μ = 1 if μ is on a core atom of α, else 0
//! ```
//!
//! The cores are disjoint and cover every atom, so `Σ_α d^α_μ = 1` and the weights sum to
//! exactly one — *provided* every subsystem that owns half of a block also contains the other
//! half. That is not automatic, and how this module makes it automatic is the subject of
//! [`Subsystem`] below.
//!
//! **The subsystems have to share electrons.** Solved separately, each subsystem would keep
//! whatever electron count it started with, and charge could never flow between them. Instead
//! all subsystem levels are filled against a **single chemical potential**, bisected so that
//! the total electron count comes out right. Each level enters the count with the fraction of
//! it that lives on the core, `n^α_i = Σ_{μ ∈ core α} |c^α_{μi}|²` — no overlap matrix, because
//! NDDO's AO basis is orthonormal. Those fractions are not integers, so the filling is
//! genuinely fractional and [`crate::fermi`] does the work.
//!
//! **The approximation has to be the one being claimed.** The density matrix of a gapped system
//! decays with distance; divide-and-conquer is exactly the statement that it can be truncated.
//! This module truncates it *explicitly*, at the buffer radius, rather than letting the buffer
//! shape decide implicitly — see [`Subsystem`].
//!
//! # What is and is not linear here
//!
//! The diagonalization is `O(N)`. The rest of the SCF is not, and this module measures both
//! rather than asserting the good half. [`DcResult::diagonalization_work`] counts `Σ_α n_α³`
//! and [`DcResult::coulomb_work`] counts the two-centre Coulomb contractions in the Fock build,
//! which stay `O(N²)`: NDDO's two-centre integrals decay as `1/R`, so they cannot be dropped by
//! distance without changing the answer — that needs the Ewald/multipole split, not a cutoff.
//! The two-centre **exchange** does become `O(N)`, and exactly so rather than by approximation,
//! because the truncated density matrix it contracts against is identically zero beyond the
//! buffer radius ([`DcResult::exchange_work`]).
//!
//! `tests/divide_conquer.rs` asserts each of these separately, including the quadratic one.

use rayon::prelude::*;

use crate::basis::Basis;
use crate::error::{Am1Error, Result};
use crate::fermi::{fill, Filling, Level};
use crate::hamiltonian::{build_core_with_neighbors, CoreHamiltonian};
use crate::linalg::{symmetric_eigen, Matrix};
use crate::math::Vec3;
use crate::neighbors::NeighborList;
use crate::params::Am1Parameters;
use crate::scf::{Am1Options, ScfReference};
use crate::system::Molecule;

/// Controls for the divide-and-conquer SCF.
#[derive(Clone, Debug)]
pub struct DcOptions {
    /// Target number of atoms per **core** region. The partition splits until every region is
    /// at most this size, so regions are between roughly half this and this.
    pub core_size: usize,
    /// Buffer radius in **Bohr**: an atom joins a subsystem when it lies within this distance
    /// of any of that subsystem's core atoms. This is the method's one physical parameter — it
    /// is the distance beyond which the density matrix is taken to vanish — and increasing it
    /// must drive the answer monotonically towards the full SCF.
    pub buffer_radius: f64,
    /// How subsystem levels are filled against the common chemical potential.
    ///
    /// Fermi–Dirac by default, and not merely for robustness: the subsystem projections
    /// `n^α_i` are fractional, so the electron count essentially never lands on a level
    /// boundary, and sharp filling would have to hand the remainder to whichever level sorted
    /// first — a discontinuous function of geometry, which dynamics cannot use.
    pub filling: Filling,
    pub max_scf: usize,
    pub e_tol: f64,
    pub p_tol: f64,
    /// Linear mixing fraction applied to the density between iterations, under the DIIS.
    pub mixing: f64,
    /// Below this HOMO–LUMO gap (eV) the result carries a warning in
    /// [`DcResult::small_gap_warning`].
    ///
    /// Divide-and-conquer rests on the density matrix decaying with distance, which is a
    /// property of gapped systems. In a metal it decays algebraically and the buffer would have
    /// to grow with the system, so the linear scaling — and the accuracy — quietly stop being
    /// true. A warning is honest; silently returning a number is not.
    pub gap_warn_ev: f64,
}

impl Default for DcOptions {
    fn default() -> Self {
        Self {
            core_size: 8,
            buffer_radius: 12.0,
            filling: Filling::Fermi { kt: 0.1 },
            max_scf: 200,
            e_tol: 1.0e-7,
            p_tol: 1.0e-6,
            mixing: 0.4,
            gap_warn_ev: 0.5,
        }
    }
}

/// One subsystem: a set of core atoms plus the buffer around them.
///
/// # Why the density is truncated at the buffer radius
///
/// The Yang weights sum to one only if, whenever atom `a` is in the core of α and the block
/// `P_ab` is kept, atom `b` is also inside subsystem α — and symmetrically with the roles of
/// `a` and `b` exchanged, in the subsystem β that owns `b`. With a buffer defined as "within
/// `r_buf` of *any* core atom", that is **not** automatic: `b` can be close to some other core
/// atom of α while being far from `a`, and then β, whose core contains `b`, has no reason to
/// contain `a`. Half the weight for that block would simply go missing.
///
/// This module fixes it at the source: a block `P_ab` is kept **only when `|R_a − R_b| ≤
/// r_buf`**. Then `a` in the core of α forces `b` within `r_buf` of a core atom of α — namely
/// `a` — so `b ∈ α`; and the mirror argument puts `a ∈ β`. The sum rule becomes exact, for
/// every geometry, with no condition on how the partition happened to fall.
///
/// That truncation is not an extra approximation bolted on to make the bookkeeping work. It is
/// the *same* approximation divide-and-conquer already makes — that the density matrix is
/// short-ranged — written down explicitly instead of left to emerge from the buffer's shape.
/// Stating it explicitly is also what lets the exchange become exactly linear-scaling rather
/// than approximately so.
#[derive(Clone, Debug)]
pub struct Subsystem {
    /// Atoms owned by this subsystem, in global indexing. Disjoint across subsystems.
    pub core_atoms: Vec<usize>,
    /// Core plus buffer, sorted, in global indexing.
    pub atoms: Vec<usize>,
    /// Global AO indices of `atoms`, in the same order.
    pub aos: Vec<usize>,
    /// Whether each entry of `aos` is on a core atom. This, not an atom-level flag, is what the
    /// level projection `n^α_i = Σ_{μ ∈ core α} |c_{μi}|²` needs.
    ao_core_flag: Vec<bool>,
    /// `weight[p][q]` — the Yang weight for the atom-pair block, already zeroed outside the
    /// buffer radius. Local indexing into `atoms`.
    weight: Vec<Vec<f64>>,
}

impl Subsystem {
    /// Number of AOs, i.e. the dimension of this subsystem's eigenproblem.
    pub fn nao(&self) -> usize {
        self.aos.len()
    }
}

/// Result of a divide-and-conquer SCF.
#[derive(Clone, Debug)]
pub struct DcResult {
    pub density: Matrix,
    /// `P_α − P_β`, present only for an unrestricted run.
    pub spin_density: Option<Matrix>,
    pub charges: Vec<f64>,
    pub electronic_ev: f64,
    pub core_ev: f64,
    pub total_ev: f64,
    pub heat_of_formation_kcal: f64,
    /// Common chemical potential, eV. Unrestricted runs report the α channel here and both in
    /// [`DcResult::fermi_energies_ev`].
    pub fermi_energy_ev: f64,
    /// One chemical potential per spin channel: `[μ]` restricted, `[μ_α, μ_β]` unrestricted.
    ///
    /// Two levels, not one, for a fixed multiplicity: the α and β electron counts are each
    /// fixed by the multiplicity, so each channel is filled to its own count. A single shared
    /// level would let the multiplicity drift.
    pub fermi_energies_ev: Vec<f64>,
    /// `T·S` from the fractional occupations, eV.
    pub entropy_ev: f64,
    pub converged: bool,
    pub iterations: usize,
    pub unrestricted: bool,
    /// Smallest gap seen across the subsystems, eV, at the common chemical potential.
    pub homo_lumo_gap_ev: f64,
    /// Set when the gap fell below [`DcOptions::gap_warn_ev`]. See that field for why.
    pub small_gap_warning: Option<String>,

    // ---- scaling counters. Deliberately counters and not timers: a wall clock in a test is a
    // flaky test, and the point of these is to state precisely which parts became linear.
    /// Number of subsystems.
    pub subsystems: usize,
    /// Largest subsystem dimension, in AOs.
    pub largest_subsystem_aos: usize,
    /// `Σ_α n_α³` — the diagonalization work. Linear in the number of atoms.
    pub diagonalization_work: f64,
    /// Two-centre Coulomb contractions per Fock build. **Quadratic**, and stays quadratic.
    pub coulomb_work: usize,
    /// Two-centre exchange contractions per Fock build, i.e. pairs whose density block survives
    /// the truncation. Linear.
    pub exchange_work: usize,
    /// Atom-pair blocks of the global density that are not identically zero.
    pub retained_density_blocks: usize,
    /// AO pairs the DIIS history stores per entry, against `nao(nao+1)/2` for a dense triangle.
    ///
    /// **Linear**, and returned so that the claim can be checked rather than believed. The
    /// divide-and-conquer density is identically zero beyond the buffer radius, so the history
    /// only ever held zeros outside this set; storing the pattern instead makes the dominant
    /// memory term of a large run grow as `N · (atoms within the buffer)` rather than as `N²`.
    /// Compare against [`DcResult::dense_triangle_elements`].
    pub diis_pattern_elements: usize,
    /// `nao(nao+1)/2` — what the history would have cost dense, for comparison. Quadratic.
    pub dense_triangle_elements: usize,
}

impl DcResult {
    /// `E − TS`, the quantity that is variational at finite electronic temperature.
    pub fn free_energy_ev(&self) -> f64 {
        self.total_ev - self.entropy_ev
    }
}

/// Split the atoms into compact, disjoint core regions of about `core_size` atoms each.
///
/// Recursive bisection along whichever axis the group is currently widest in. Two properties
/// matter, and the second one is easy to get wrong.
///
/// **Compact**, because a subsystem's cost is set by how many atoms fall inside the buffer
/// around its core, and a ball-shaped core has a far smaller buffer than a slab-shaped one with
/// the same atom count.
///
/// **Uniform in size**, because the largest subsystem is what the cost is actually paid on.
/// Splitting each group into equal halves until every leaf fits under `core_size` cannot be
/// uniform: the leaf count is forced to a power of two, so 1536 atoms at `core_size = 12` gives
/// 128 cores of exactly 12 while 2187 gives 256 cores of 8.5, and the subsystem size then swings
/// by 20 % from one system size to the next rather than settling. Measured, before this was
/// fixed: largest subsystem 377, 382, 441, 355, 424 AOs across five increasing cluster sizes —
/// no trend, just noise from how the atom count happened to factor.
///
/// So each split is **proportional** instead: the group is divided into the number of parts it
/// will eventually need, and cut so that each side receives its share. Every leaf then holds
/// close to `core_size` atoms whatever the total is.
pub fn partition_atoms(molecule: &Molecule, core_size: usize) -> Vec<Vec<usize>> {
    let core_size = core_size.max(1);
    let mut out = Vec::new();
    let all: Vec<usize> = (0..molecule.atoms.len()).collect();
    if !all.is_empty() {
        bisect(molecule, all, core_size, &mut out);
    }
    out
}

fn bisect(molecule: &Molecule, mut idx: Vec<usize>, core_size: usize, out: &mut Vec<Vec<usize>>) {
    if idx.len() <= core_size {
        out.push(idx);
        return;
    }
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for &i in &idx {
        let p = molecule.atoms[i].position;
        for (axis, v) in [p.x, p.y, p.z].into_iter().enumerate() {
            lo[axis] = lo[axis].min(v);
            hi[axis] = hi[axis].max(v);
        }
    }
    // The longest extent, found by a fold rather than `max_by(..).unwrap()`: there are always
    // exactly three axes, so the `Option` was never `None`, and writing it this way makes that
    // structural instead of a claim the reader has to check.
    let mut axis = 0usize;
    for a in 1..3 {
        if hi[a] - lo[a] > hi[axis] - lo[axis] {
            axis = a;
        }
    }
    let key = |i: usize| -> f64 {
        let p = molecule.atoms[i].position;
        [p.x, p.y, p.z][axis]
    };
    idx.sort_by(|&a, &b| key(a).total_cmp(&key(b)).then(a.cmp(&b)));

    // How many cores this group will become, and how many go to the left. Cutting at
    // `len * left_parts / parts` rather than at the midpoint is what keeps the leaves uniform
    // when `parts` is odd.
    let parts = idx.len().div_ceil(core_size);
    let left_parts = parts / 2;
    let split_at = (idx.len() * left_parts / parts).clamp(1, idx.len() - 1);
    let right = idx.split_off(split_at);
    bisect(molecule, idx, core_size, out);
    bisect(molecule, right, core_size, out);
}

/// Build the subsystems: each core plus every atom within `buffer_radius` of one of its cores.
pub fn build_subsystems(
    molecule: &Molecule,
    basis: &Basis,
    cores: &[Vec<usize>],
    buffer_radius: f64,
) -> Vec<Subsystem> {
    let positions: Vec<Vec3> = molecule.atoms.iter().map(|a| a.position).collect();
    let r2 = buffer_radius * buffer_radius;

    // Under a cell, "how far apart" means the **minimum image** distance, not the difference of
    // the stored coordinates.
    //
    // This is what makes divide-and-conquer periodic rather than merely tolerant of a cell. A
    // subsystem near the cell boundary has neighbours on the other side of it, and measuring by
    // raw coordinates puts them at the far end of the cell instead — so the buffer misses them
    // and the truncation keeps blocks it should drop.
    //
    // The symptom is not a wrong-looking energy. Both the convergence-with-buffer test and the
    // Yang sum rule stay satisfied, because they only need the two uses of the distance to agree
    // with *each other*. What breaks is size consistency: measured before this, a doubled cell
    // cost 1.6e-4 eV more than twice the primitive cell, because the two describe the same
    // infinite crystal but disagree about which neighbours are within the buffer.
    let distance2 = |a: Vec3, b: Vec3| -> f64 {
        match molecule.cell {
            Some(cell) => cell.minimum_image(b - a).norm2(),
            None => (b - a).norm2(),
        }
    };

    cores
        .par_iter()
        .map(|core| {
            let mut in_core = vec![false; positions.len()];
            for &c in core {
                in_core[c] = true;
            }
            let mut atoms: Vec<usize> = (0..positions.len())
                .filter(|&j| {
                    in_core[j]
                        || core
                            .iter()
                            .any(|&c| distance2(positions[c], positions[j]) <= r2)
                })
                .collect();
            atoms.sort_unstable();

            let core_flag: Vec<bool> = atoms.iter().map(|&a| in_core[a]).collect();

            let mut aos = Vec::new();
            let mut ao_core_flag = Vec::new();
            for (local, &a) in atoms.iter().enumerate() {
                for k in 0..basis.atom_norb[a] {
                    aos.push(basis.atom_offset[a] + k);
                    ao_core_flag.push(core_flag[local]);
                }
            }

            // Yang weight per atom-pair block, already truncated at the buffer radius. See the
            // type documentation: the truncation is what makes the weights sum to exactly one.
            let n = atoms.len();
            let mut weight = vec![vec![0.0; n]; n];
            for p in 0..n {
                for q in 0..n {
                    let w = 0.5 * (f64::from(core_flag[p]) + f64::from(core_flag[q]));
                    if w > 0.0 && distance2(positions[atoms[p]], positions[atoms[q]]) <= r2 {
                        weight[p][q] = w;
                    }
                }
            }

            Subsystem {
                core_atoms: core.clone(),
                atoms,
                aos,
                ao_core_flag,
                weight,
            }
        })
        .collect()
}

/// Total Yang weight applied to each atom-pair block, summed over subsystems.
///
/// Exposed so the sum rule can be checked **directly** rather than inferred from the energy
/// agreeing with a full SCF. Those are different claims: an energy can agree because the errors
/// cancel, whereas this is the identity the method rests on. Entry `(a, b)` must be exactly 1
/// when `|R_a − R_b| ≤ buffer_radius` and exactly 0 otherwise.
pub fn partition_weight_sum(molecule: &Molecule, subsystems: &[Subsystem]) -> Matrix {
    let nat = molecule.atoms.len();
    let mut total = Matrix::zeros(nat, nat);
    for sub in subsystems {
        for (p, &a) in sub.atoms.iter().enumerate() {
            for (q, &b) in sub.atoms.iter().enumerate() {
                total[(a, b)] += sub.weight[p][q];
            }
        }
    }
    total
}

/// Divide-and-conquer SCF.
///
/// `options` supplies charge, multiplicity and reference exactly as for [`crate::run_am1`];
/// `dc` supplies the partitioning. Returns the assembled global density and the energy
/// evaluated from it with the ordinary (undivided) energy expression, so that a buffer large
/// enough to cover the whole system reproduces the full SCF to roundoff.
pub fn run_divide_conquer(
    molecule: &Molecule,
    params: &Am1Parameters,
    options: &Am1Options,
    dc: &DcOptions,
) -> Result<DcResult> {
    crate::linalg::enable_parallelism();
    let _t = crate::timing::Timer::start("dc:total");

    if dc.buffer_radius <= 0.0 {
        return Err(Am1Error::InvalidInput(
            "divide-and-conquer buffer_radius must be positive".into(),
        ));
    }
    let nat = molecule.atoms.len();
    if nat == 0 {
        return Err(Am1Error::InvalidInput("empty molecule".into()));
    }

    let basis = Basis::build(molecule, params)?;
    let _tcore = crate::timing::Timer::start("dc:core");
    let neighbors =
        NeighborList::build_screened(molecule, options.realspace_cutoff, options.multipole_cutoff);
    let core =
        build_core_with_neighbors(molecule, &basis, params, &neighbors, options.core_build())?;

    drop(_tcore);
    let cores = partition_atoms(molecule, dc.core_size);
    let subsystems = {
        let _t = crate::timing::Timer::start("dc:subsystems");
        build_subsystems(molecule, &basis, &cores, dc.buffer_radius)
    };

    // Electron counts, exactly as the full SCF derives them, so a charged or open-shell system
    // takes the same route here as there.
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
    let n_unpaired = (options.multiplicity.max(1) - 1) as i64;
    if (n_elec_int - n_unpaired) < 0 || (n_elec_int - n_unpaired) % 2 != 0 {
        return Err(Am1Error::InvalidInput(format!(
            "electron count {n_elec_int} is incompatible with multiplicity {} (need same parity)",
            options.multiplicity
        )));
    }
    let n_alpha = ((n_elec_int + n_unpaired) / 2) as f64;
    let n_beta = ((n_elec_int - n_unpaired) / 2) as f64;
    let closed_shell = (n_alpha - n_beta).abs() < 1.0e-12;
    let unrestricted = match options.reference {
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

    let state = if unrestricted {
        dc_loop_unrestricted(
            molecule,
            &basis,
            params,
            &core,
            &subsystems,
            n_alpha,
            n_beta,
            dc,
        )?
    } else {
        dc_loop_restricted(molecule, &basis, params, &core, &subsystems, n_elec, dc)?
    };

    let core_ev = crate::repulsion::core_core_energy_with_neighbors(molecule, params, &neighbors)?;
    // The external field's nuclear half, as `run_am1` does it. The electronic half is already in
    // `state.electronic_ev`, having entered through `H_core`.
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
    let heat_of_formation_kcal = (total_ev - e_isol_sum + eheat_sum) * crate::constants::EV_TO_KCAL;

    let mut charges = vec![0.0; nat];
    for (ia, atom) in molecule.atoms.iter().enumerate() {
        let off = basis.atom_offset[ia];
        let n = basis.atom_norb[ia];
        let mut pop = 0.0;
        for mu in 0..n {
            pop += state.density[(off + mu, off + mu)];
        }
        charges[ia] = params.element(atom.z)?.core_charge - pop;
    }

    let small_gap_warning = (state.gap_ev < dc.gap_warn_ev).then(|| {
        format!(
            "smallest subsystem gap is {:.3} eV, below the {:.3} eV threshold. Divide-and-conquer \
             assumes the density matrix decays with distance, which is a property of gapped \
             systems; in a small-gap or metallic system it decays algebraically, so the buffer \
             radius would have to grow with the system and neither the accuracy nor the linear \
             scaling holds. Check the result against a full SCF, or increase buffer_radius.",
            state.gap_ev, dc.gap_warn_ev
        )
    });

    let diagonalization_work: f64 = subsystems
        .iter()
        .map(|s| {
            let n = s.nao() as f64;
            n * n * n
        })
        .sum();
    let largest_subsystem_aos = subsystems.iter().map(|s| s.nao()).max().unwrap_or(0);
    let retained_density_blocks = count_retained_blocks(&basis, &state.density, nat);
    let diis_pattern_elements = retained_pattern(&basis, &subsystems, basis.nao).len();
    let dense_triangle_elements = basis.nao * (basis.nao + 1) / 2;

    // No `timing::report` here — see the note in `crate::timing` and in `run_am1`.

    Ok(DcResult {
        density: state.density,
        spin_density: state.spin_density,
        charges,
        electronic_ev: state.electronic_ev,
        core_ev,
        total_ev,
        heat_of_formation_kcal,
        fermi_energy_ev: state.fermi_energies[0],
        fermi_energies_ev: state.fermi_energies,
        entropy_ev: state.entropy_ev,
        converged: state.converged,
        iterations: state.iterations,
        unrestricted,
        homo_lumo_gap_ev: state.gap_ev,
        small_gap_warning,
        subsystems: subsystems.len(),
        largest_subsystem_aos,
        diagonalization_work,
        coulomb_work: core.pairs.len(),
        exchange_work: state.exchange_work,
        retained_density_blocks,
        diis_pattern_elements,
        dense_triangle_elements,
    })
}

struct DcState {
    density: Matrix,
    spin_density: Option<Matrix>,
    electronic_ev: f64,
    fermi_energies: Vec<f64>,
    entropy_ev: f64,
    gap_ev: f64,
    converged: bool,
    iterations: usize,
    exchange_work: usize,
}

/// Diagonalize every subsystem's block of `f` and return `(eigenvalues, eigenvectors)` per
/// subsystem, together with the core projection `n^α_i` of each level.
fn solve_subsystems(f: &Matrix, subsystems: &[Subsystem]) -> Result<Vec<SubSolution>> {
    let _t = crate::timing::Timer::start("dc:diagonalize");
    subsystems
        .par_iter()
        .map(|sub| {
            let n = sub.nao();
            let mut block = Matrix::zeros(n, n);
            for (p, &mu) in sub.aos.iter().enumerate() {
                for (q, &nu) in sub.aos.iter().enumerate() {
                    block[(p, q)] = f[(mu, nu)];
                }
            }
            let (eps, c) = symmetric_eigen(&block)?;
            // n^α_i = Σ_{μ ∈ core α} |c_{μi}|². No S: the NDDO AO basis is orthonormal, which is
            // exactly the simplification that makes the projection a plain column sum.
            let projection: Vec<f64> = (0..n)
                .map(|i| {
                    (0..n)
                        .filter(|&p| sub.ao_core_flag[p])
                        .map(|p| c[(p, i)] * c[(p, i)])
                        .sum()
                })
                .collect();
            Ok(SubSolution {
                energies: eps,
                coeff: c,
                projection,
            })
        })
        .collect()
}

struct SubSolution {
    energies: Vec<f64>,
    coeff: Matrix,
    projection: Vec<f64>,
}

/// Fill every subsystem level of every subsystem against one chemical potential.
///
/// `capacity` is 2 for a restricted run (a spatial orbital holds two electrons) and 1 for one
/// spin channel.
fn common_fermi(
    solutions: &[SubSolution],
    capacity: f64,
    electrons: f64,
    filling: Filling,
) -> Result<(Vec<Vec<f64>>, f64, f64)> {
    let mut levels = Vec::new();
    for sol in solutions {
        for (i, &e) in sol.energies.iter().enumerate() {
            levels.push(Level {
                energy: e,
                weight: capacity * sol.projection[i],
            });
        }
    }
    let occ = fill(&levels, electrons, filling)?;

    // Unflatten back into per-subsystem occupation fractions.
    let mut out = Vec::with_capacity(solutions.len());
    let mut cursor = 0;
    for sol in solutions {
        let n = sol.energies.len();
        out.push(occ.fractions[cursor..cursor + n].to_vec());
        cursor += n;
    }
    Ok((out, occ.fermi_energy, occ.ts))
}

/// Smallest gap straddling `mu` across all subsystems.
fn smallest_gap(solutions: &[SubSolution], mu: f64) -> f64 {
    let mut gap = f64::INFINITY;
    for sol in solutions {
        let below = sol
            .energies
            .iter()
            .copied()
            .filter(|&e| e <= mu)
            .fold(f64::NEG_INFINITY, f64::max);
        let above = sol
            .energies
            .iter()
            .copied()
            .filter(|&e| e > mu)
            .fold(f64::INFINITY, f64::min);
        if below.is_finite() && above.is_finite() {
            gap = gap.min(above - below);
        }
    }
    if gap.is_finite() {
        gap
    } else {
        0.0
    }
}

/// Assemble the global density from the subsystem solutions and the Yang weights.
fn assemble_density(
    nao: usize,
    basis: &Basis,
    subsystems: &[Subsystem],
    solutions: &[SubSolution],
    fractions: &[Vec<f64>],
    capacity: f64,
) -> Matrix {
    let _t = crate::timing::Timer::start("dc:assemble");

    // Per subsystem: D^α = Σ_i (capacity · f_i) c_i c_iᵀ, formed as one matrix product over the
    // occupied columns rather than an outer-product loop, then scattered with the Yang weights.
    let blocks: Vec<Matrix> = subsystems
        .par_iter()
        .zip(solutions.par_iter())
        .zip(fractions.par_iter())
        .map(|((sub, sol), frac)| {
            let n = sub.nao();
            let kept: Vec<usize> = (0..n).filter(|&i| frac[i] > 1.0e-14).collect();
            let mut scaled = Matrix::zeros(n, kept.len());
            for (col, &i) in kept.iter().enumerate() {
                let s = (capacity * frac[i]).sqrt();
                for p in 0..n {
                    scaled[(p, col)] = sol.coeff[(p, i)] * s;
                }
            }
            // Sequential and transpose-free: this closure is already a rayon task per
            // subsystem, and faer's own threads would contend with that outer pool.
            scaled.matmul_transpose_seq(&scaled)
        })
        .collect();

    let mut p = Matrix::zeros(nao, nao);
    for (sub, block) in subsystems.iter().zip(&blocks) {
        // AO offsets of each subsystem atom inside the subsystem's own indexing.
        let mut local_off = Vec::with_capacity(sub.atoms.len());
        let mut acc = 0;
        for &a in &sub.atoms {
            local_off.push(acc);
            acc += basis.atom_norb[a];
        }
        for (pa, &a) in sub.atoms.iter().enumerate() {
            for (qb, &b) in sub.atoms.iter().enumerate() {
                let w = sub.weight[pa][qb];
                if w == 0.0 {
                    continue;
                }
                let (ga, gb) = (basis.atom_offset[a], basis.atom_offset[b]);
                let (la, lb) = (local_off[pa], local_off[qb]);
                for i in 0..basis.atom_norb[a] {
                    for j in 0..basis.atom_norb[b] {
                        p[(ga + i, gb + j)] += w * block[(la + i, lb + j)];
                    }
                }
            }
        }
    }
    p
}

fn count_retained_blocks(basis: &Basis, p: &Matrix, nat: usize) -> usize {
    let mut count = 0;
    for a in 0..nat {
        for b in 0..nat {
            let (oa, ob) = (basis.atom_offset[a], basis.atom_offset[b]);
            let mut nonzero = false;
            'outer: for i in 0..basis.atom_norb[a] {
                for j in 0..basis.atom_norb[b] {
                    if p[(oa + i, ob + j)] != 0.0 {
                        nonzero = true;
                        break 'outer;
                    }
                }
            }
            if nonzero {
                count += 1;
            }
        }
    }
    count
}

/// Number of pairs whose same-spin density block survives, i.e. the exchange work per Fock
/// build. Counted here rather than inside the Fock build so that the Fock build stays a pure
/// function of its arguments.
fn count_exchange_work(core: &CoreHamiltonian, basis: &Basis, p_spin: &Matrix) -> usize {
    core.pairs
        .iter()
        .filter(|pair| {
            let (oa, ob) = (basis.atom_offset[pair.a], basis.atom_offset[pair.b]);
            let (na, nb) = (pair.te.norb_i, pair.te.norb_j);
            (0..na).any(|i| (0..nb).any(|j| p_spin[(oa + i, ob + j)] != 0.0))
        })
        .count()
}

/// Pulay DIIS on the density residual `r = P_out − P_in`.
///
/// Not the `[F, P]` commutator the full SCF uses: the divide-and-conquer density is assembled
/// from separately-diagonalized blocks, so it does not commute with the global Fock matrix even
/// at convergence and that residual would never reach zero. The density residual does.
/// The history is stored as **packed upper triangles**, and the depth is capped by a memory
/// budget rather than fixed.
///
/// Both matter at the sizes this method exists for. A depth-8 history of full `nao × nao`
/// densities and residuals is 16 dense matrices; at 1536 atoms (3072 AOs) that is 1.2 GB, which
/// is most of the peak footprint of the whole calculation and is quadratic in the atom count —
/// exactly the wrong shape for the one part of the code whose purpose is large systems.
///
/// Packing halves it exactly. The divide-and-conquer density is symmetric (the Yang weight
/// `½(d_μ + d_ν)` is symmetric and every subsystem block is), so is the residual as a difference
/// of two symmetric matrices, so nothing is approximated by storing one triangle. The cap
/// handles the rest: below the budget the depth is the full 8, and above it the history is
/// shortened rather than the calculation failing.
struct DensityDiis {
    /// History entries, gathered onto [`DensityDiis::pattern`].
    inputs: Vec<Vec<f64>>,
    residuals: Vec<Vec<f64>>,
    /// Flat row-major indices `row·nao + col`, `row ≤ col`, of the elements the divide-and-conquer
    /// density can be nonzero at — sorted, deduplicated, built once from the subsystem weights.
    ///
    /// This is what makes the history `O(N)` rather than `O(N²)`. The density is truncated at the
    /// buffer radius *exactly* — [`assemble_density`] never writes a block whose Yang weight is
    /// zero — so every element outside this set is identically zero at every iteration, and
    /// storing it was storing zeros. At 1029 atoms with a 12 Bohr buffer the dense packed triangle
    /// is 2.1 M elements and the pattern is about 0.2 M, and the gap widens linearly with the
    /// system: the pattern grows as `N · (atoms within the buffer)` while the triangle grows as
    /// `N²`.
    pattern: Vec<usize>,
    /// `2` for an off-diagonal element, `1` for a diagonal one — the multiplicity each entry
    /// stands for in the full symmetric matrix.
    multiplicity: Vec<f64>,
    /// `dots[i][j] = ⟨r_i, r_j⟩` for `j ≤ i` — the lower triangle of the DIIS B matrix, kept
    /// between iterations.
    ///
    /// These never change once computed: an entry depends only on two residuals that are already
    /// in the history, and a residual is never modified after it is pushed. Rebuilding the whole
    /// matrix every iteration was the largest single cost in a large run — see `push`.
    dots: Vec<Vec<f64>>,
    nao: usize,
    max: usize,
}

/// The AO index pairs the divide-and-conquer density can be nonzero at, as flat `row·nao + col`
/// indices with `row ≤ col`.
///
/// Read off the subsystem Yang weights, which are the same thing [`assemble_density`] consults
/// before writing a block — so this is the density's actual sparsity pattern rather than a
/// distance criterion that happens to agree with it.
fn retained_pattern(basis: &Basis, subsystems: &[Subsystem], nao: usize) -> Vec<usize> {
    let mut out: Vec<usize> = Vec::new();
    for sub in subsystems {
        for (pa, &a) in sub.atoms.iter().enumerate() {
            for (qb, &b) in sub.atoms.iter().enumerate() {
                if sub.weight[pa][qb] == 0.0 {
                    continue;
                }
                let (ga, gb) = (basis.atom_offset[a], basis.atom_offset[b]);
                for i in 0..basis.atom_norb[a] {
                    for j in 0..basis.atom_norb[b] {
                        let (r, c) = (ga + i, gb + j);
                        let (r, c) = if r <= c { (r, c) } else { (c, r) };
                        out.push(r * nao + c);
                    }
                }
            }
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// Memory the DIIS history is allowed, in bytes. 512 MB is generous for a workstation and still
/// bounds the depth at the sizes where the quadratic growth would otherwise dominate.
const DIIS_MEMORY_BUDGET: usize = 512 << 20;

impl DensityDiis {
    /// Gather a symmetric matrix onto the retained pattern.
    ///
    /// A gather rather than the contiguous row copies the dense version used: the pattern is
    /// sorted, so this still walks `m` forwards, but it touches only the elements that can be
    /// nonzero. That is the whole saving — a tenth of the memory traffic at 1029 atoms, and a
    /// falling fraction as the system grows.
    fn gather(&self, m: &Matrix) -> Vec<f64> {
        debug_assert_eq!(m.rows, self.nao);
        debug_assert_eq!(m.cols, self.nao, "the density is square");
        let src = m.as_slice();
        self.pattern.iter().map(|&k| src[k]).collect()
    }

    fn new(max: usize, nao: usize, pattern: Vec<usize>) -> Self {
        // Two arrays (input and residual) per history entry.
        let per_entry = 2 * pattern.len().max(1) * std::mem::size_of::<f64>();
        let affordable = (DIIS_MEMORY_BUDGET / per_entry.max(1)).max(2);
        let multiplicity = pattern
            .iter()
            .map(|&k| if k / nao == k % nao { 1.0 } else { 2.0 })
            .collect();
        Self {
            inputs: Vec::new(),
            residuals: Vec::new(),
            dots: Vec::new(),
            pattern,
            multiplicity,
            nao,
            max: max.min(affordable),
        }
    }

    fn push(&mut self, input: &Matrix, residual: &Matrix) {
        let (i_packed, r_packed) = (self.gather(input), self.gather(residual));
        self.inputs.push(i_packed);
        self.residuals.push(r_packed);

        // Only the new residual's row of the B matrix is new; every other entry is already known.
        //
        // This is why the cache exists. `extrapolate` used to call `residual_dot` for all `n²`
        // ordered pairs — both triangles — on every iteration. At 1029 atoms a packed residual is
        // 16.9 MB, so a depth-8 history meant 64 dot products over 2 × 16.9 MB: 2.2 GB of memory
        // traffic per SCF iteration, none of which fits in any cache. It was invisible because it
        // sat *between* the phase timers rather than inside one, which is why the labelled phases
        // summed to 8.8 s of a 16.1 s run.
        //
        // One row instead of the full matrix turns 64 dot products per iteration into at most 8.
        //
        // In parallel, because the entries are independent and each streams two 16.9 MB arrays:
        // the loop is memory-bound, and several cores pull more bandwidth than one.
        let i = self.residuals.len() - 1;
        let row: Vec<f64> = {
            let this: &Self = self;
            (0..=i)
                .into_par_iter()
                .map(|j| this.residual_dot(i, j))
                .collect()
        };
        self.dots.push(row);

        if self.inputs.len() > self.max {
            self.inputs.remove(0);
            self.residuals.remove(0);
            // The evicted residual has to leave both axes of the cache, not just one.
            self.dots.remove(0);
            for row in &mut self.dots {
                row.remove(0);
            }
        }
    }

    /// `⟨rᵢ, rⱼ⟩` over the full matrices, from the gathered triangles.
    ///
    /// Each stored element stands for one or two elements of the symmetric matrix, so it is
    /// weighted by its multiplicity. A flat three-way `zip` over contiguous memory, which
    /// vectorizes; the summation order differs from a walk over the full matrix, so the last bit
    /// can move, which is far below anything asserted anywhere.
    fn residual_dot(&self, i: usize, j: usize) -> f64 {
        let (a, b) = (&self.residuals[i], &self.residuals[j]);
        a.iter()
            .zip(b.iter())
            .zip(&self.multiplicity)
            .map(|((x, y), w)| w * x * y)
            .sum()
    }

    fn extrapolate(&self) -> Option<Matrix> {
        let n = self.inputs.len();
        if n < 2 {
            return None;
        }
        let dim = n + 1;
        let mut b = Matrix::zeros(dim, dim);
        // Normalize by the largest entry: the residuals span many orders of magnitude over an
        // SCF, and an unscaled B matrix becomes numerically singular long before the history is
        // actually redundant.
        let mut scale = 0.0_f64;
        for i in 0..n {
            // Read from the cache `push` filled in, and mirror: the B matrix is a Gram matrix, so
            // the upper triangle is the lower one transposed and there is nothing to recompute.
            for j in 0..=i {
                let v = self.dots[i][j];
                b[(i, j)] = v;
                b[(j, i)] = v;
                scale = scale.max(v.abs());
            }
            b[(i, n)] = -1.0;
            b[(n, i)] = -1.0;
        }
        if scale <= 0.0 {
            return None;
        }
        for i in 0..n {
            for j in 0..n {
                b[(i, j)] /= scale;
            }
            // Tikhonov ridge, for the same reason.
            b[(i, i)] += 1.0e-10;
        }
        let mut rhs = vec![0.0; dim];
        rhs[n] = -1.0;
        let coeff = crate::linalg::solve_linear(&b, &rhs).ok()?;
        if coeff.iter().take(n).any(|c| !c.is_finite()) {
            return None;
        }

        // Accumulate the combination in gathered form, then scatter once.
        //
        // The obvious loop writes `out[(row, col)]` and `out[(col, row)]` for every history entry,
        // so a depth-8 history makes eight scattered passes over an `nao × nao` matrix. Summing
        // the gathered triangles first is eight flat passes over the pattern and exactly one
        // scatter.
        //
        // Everything outside the pattern stays zero, which is not an approximation: the
        // divide-and-conquer density is identically zero there at every iteration, so any linear
        // combination of history entries is too.
        let mut packed = vec![0.0; self.pattern.len()];
        for (i, c) in coeff.iter().take(n).enumerate() {
            let (src, res) = (&self.inputs[i], &self.residuals[i]);
            for ((p, s), r) in packed.iter_mut().zip(src).zip(res) {
                *p += c * (s + r);
            }
        }
        let mut out = Matrix::zeros(self.nao, self.nao);
        let nao = self.nao;
        {
            let dst = out.as_mut_slice();
            for (&k, v) in self.pattern.iter().zip(&packed) {
                let (row, col) = (k / nao, k % nao);
                dst[k] = *v;
                dst[col * nao + row] = *v;
            }
        }
        Some(out)
    }
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

#[allow(clippy::too_many_arguments)]
fn dc_loop_restricted(
    molecule: &Molecule,
    basis: &Basis,
    params: &Am1Parameters,
    core: &CoreHamiltonian,
    subsystems: &[Subsystem],
    n_elec: f64,
    dc: &DcOptions,
) -> Result<DcState> {
    let nao = basis.nao;
    let mut density = initial_density(molecule, basis, params, subsystems, dc)?;
    let mut diis = DensityDiis::new(8, nao, retained_pattern(basis, subsystems, nao));
    let mut e_old = 0.0;
    let mut converged = false;
    let mut iterations = 0;
    let mut fermi = 0.0;
    let mut entropy = 0.0;
    let mut gap = 0.0;

    for iter in 0..dc.max_scf {
        iterations = iter + 1;
        let f = {
            let _t = crate::timing::Timer::start("dc:fock");
            crate::fock::build_fock(molecule, basis, params, core, &density)?
        };
        let e_elec = {
            let _t = crate::timing::Timer::start("dc:energy");
            0.5 * (density.frobenius_dot(&core.h_core) + density.frobenius_dot(&f))
        };

        let solutions = solve_subsystems(&f, subsystems)?;
        let (fractions, mu, ts) = {
            let _t = crate::timing::Timer::start("dc:fermi");
            common_fermi(&solutions, 2.0, n_elec, dc.filling)?
        };
        let p_out = assemble_density(nao, basis, subsystems, &solutions, &fractions, 2.0);

        fermi = mu;
        entropy = ts;
        gap = smallest_gap(&solutions, mu);

        let dp = rms_diff(&p_out, &density);
        let de = (e_elec - e_old).abs();
        e_old = e_elec;
        if iter > 0 && de < dc.e_tol && dp < dc.p_tol {
            density = p_out;
            converged = true;
            break;
        }

        // Timed as one block: every step from here to the next density is a pass over an
        // `nao × nao` matrix, and they are only worth separating if the block itself is large.
        let _t = crate::timing::Timer::start("dc:diis");
        let mut residual = p_out.clone();
        for (r, d) in residual.as_mut_slice().iter_mut().zip(density.as_slice()) {
            *r -= d;
        }
        diis.push(&density, &residual);
        density = match diis.extrapolate() {
            Some(next) => next,
            None => {
                let mut mixed = density.clone();
                for (m, o) in mixed.as_mut_slice().iter_mut().zip(p_out.as_slice()) {
                    *m += dc.mixing * (o - *m);
                }
                mixed
            }
        };
    }

    let f_final = {
        let _t = crate::timing::Timer::start("dc:fock");
        crate::fock::build_fock(molecule, basis, params, core, &density)?
    };
    let electronic_ev = {
        let _t = crate::timing::Timer::start("dc:energy");
        0.5 * (density.frobenius_dot(&core.h_core) + density.frobenius_dot(&f_final))
            + crate::fock::long_range_energy_term(molecule, basis, params, core, &density)?
    };
    let mut half = density.clone();
    for v in half.as_mut_slice() {
        *v *= 0.5;
    }
    let exchange_work = count_exchange_work(core, basis, &half);

    Ok(DcState {
        density,
        spin_density: None,
        electronic_ev,
        fermi_energies: vec![fermi],
        entropy_ev: entropy,
        gap_ev: gap,
        converged,
        iterations,
        exchange_work,
    })
}

#[allow(clippy::too_many_arguments)]
fn dc_loop_unrestricted(
    molecule: &Molecule,
    basis: &Basis,
    params: &Am1Parameters,
    core: &CoreHamiltonian,
    subsystems: &[Subsystem],
    n_alpha: f64,
    n_beta: f64,
    dc: &DcOptions,
) -> Result<DcState> {
    let nao = basis.nao;
    let total = initial_density(molecule, basis, params, subsystems, dc)?;
    // Split the SAD guess *in proportion to the α/β electron counts*, exactly as the full
    // unrestricted SCF does. The proportion matters twice over: it is what breaks spin symmetry
    // for an open shell, and — because the two fractions sum to one — it keeps the initial total
    // density holding the right number of electrons. A guess that does not is not merely
    // inelegant; it starts the iteration in a different basin, and unrestricted Hartree–Fock
    // has more than one solution to be caught by.
    let n_tot = (n_alpha + n_beta).max(1.0);
    let mut pa = total.clone();
    let mut pb = total;
    for v in pa.as_mut_slice() {
        *v *= n_alpha / n_tot;
    }
    for v in pb.as_mut_slice() {
        *v *= n_beta / n_tot;
    }

    let pattern = retained_pattern(basis, subsystems, nao);
    let mut diis_a = DensityDiis::new(8, nao, pattern.clone());
    let mut diis_b = DensityDiis::new(8, nao, pattern);
    let mut e_old = 0.0;
    let mut converged = false;
    let mut iterations = 0;
    let mut fermi = vec![0.0, 0.0];
    let mut entropy = 0.0;
    let mut gap = 0.0;

    for iter in 0..dc.max_scf {
        iterations = iter + 1;
        let mut p_tot = pa.clone();
        for (t, b) in p_tot.as_mut_slice().iter_mut().zip(pb.as_slice()) {
            *t += b;
        }
        let fa = crate::fock::build_fock_spin(molecule, basis, params, core, &p_tot, &pa)?;
        let fb = crate::fock::build_fock_spin(molecule, basis, params, core, &p_tot, &pb)?;
        let e_elec = 0.5
            * (p_tot.frobenius_dot(&core.h_core) + pa.frobenius_dot(&fa) + pb.frobenius_dot(&fb));

        // Each channel gets its own chemical potential, because the multiplicity fixes each
        // channel's electron count separately. One shared level would let the two exchange
        // electrons and the multiplicity would drift away from what was asked for.
        let sol_a = solve_subsystems(&fa, subsystems)?;
        let sol_b = solve_subsystems(&fb, subsystems)?;
        let (frac_a, mu_a, ts_a) = common_fermi(&sol_a, 1.0, n_alpha, dc.filling)?;
        let (frac_b, mu_b, ts_b) = common_fermi(&sol_b, 1.0, n_beta, dc.filling)?;
        let pa_out = assemble_density(nao, basis, subsystems, &sol_a, &frac_a, 1.0);
        let pb_out = assemble_density(nao, basis, subsystems, &sol_b, &frac_b, 1.0);

        fermi = vec![mu_a, mu_b];
        entropy = ts_a + ts_b;
        gap = smallest_gap(&sol_a, mu_a).min(smallest_gap(&sol_b, mu_b));

        let dp = rms_diff(&pa_out, &pa).max(rms_diff(&pb_out, &pb));
        let de = (e_elec - e_old).abs();
        e_old = e_elec;
        if iter > 0 && de < dc.e_tol && dp < dc.p_tol {
            pa = pa_out;
            pb = pb_out;
            converged = true;
            break;
        }

        for (p, p_out, diis) in [
            (&mut pa, &pa_out, &mut diis_a),
            (&mut pb, &pb_out, &mut diis_b),
        ] {
            let mut residual = p_out.clone();
            for (r, d) in residual.as_mut_slice().iter_mut().zip(p.as_slice()) {
                *r -= d;
            }
            diis.push(p, &residual);
            *p = match diis.extrapolate() {
                Some(next) => next,
                None => {
                    let mut mixed = p.clone();
                    for (m, o) in mixed.as_mut_slice().iter_mut().zip(p_out.as_slice()) {
                        *m += dc.mixing * (o - *m);
                    }
                    mixed
                }
            };
        }
    }

    let mut p_tot = pa.clone();
    for (t, b) in p_tot.as_mut_slice().iter_mut().zip(pb.as_slice()) {
        *t += b;
    }
    let fa = crate::fock::build_fock_spin(molecule, basis, params, core, &p_tot, &pa)?;
    let fb = crate::fock::build_fock_spin(molecule, basis, params, core, &p_tot, &pb)?;
    let electronic_ev = 0.5
        * (p_tot.frobenius_dot(&core.h_core) + pa.frobenius_dot(&fa) + pb.frobenius_dot(&fb))
        + crate::fock::long_range_energy_term(molecule, basis, params, core, &p_tot)?;

    let mut spin = pa.clone();
    for (s, b) in spin.as_mut_slice().iter_mut().zip(pb.as_slice()) {
        *s -= b;
    }
    let exchange_work = count_exchange_work(core, basis, &pa);

    Ok(DcState {
        density: p_tot,
        spin_density: Some(spin),
        electronic_ev,
        fermi_energies: fermi,
        entropy_ev: entropy,
        gap_ev: gap,
        converged,
        iterations,
        exchange_work,
    })
}

/// Superposition of atomic densities, truncated to the blocks divide-and-conquer will keep.
///
/// Truncating the *guess* the same way as the result matters: an untruncated guess makes the
/// first Fock build do the full quadratic exchange, and — worse for a test — makes the first
/// iteration's exchange work depend on the system size in a way the converged one does not.
fn initial_density(
    molecule: &Molecule,
    basis: &Basis,
    params: &Am1Parameters,
    _subsystems: &[Subsystem],
    _dc: &DcOptions,
) -> Result<Matrix> {
    let nao = basis.nao;
    let mut p = Matrix::zeros(nao, nao);
    for (ia, atom) in molecule.atoms.iter().enumerate() {
        let elem = params.element(atom.z)?;
        let off = basis.atom_offset[ia];
        let n = basis.atom_norb[ia];
        let zv = elem.core_charge;
        let n_s = zv.min(2.0);
        let n_p = (zv - n_s).max(0.0);
        p[(off, off)] = n_s;
        if n == 4 {
            let per_p = n_p / 3.0;
            for k in 1..4 {
                p[(off + k, off + k)] = per_p;
            }
        }
    }
    Ok(p)
}

/// Analytic stress tensor (eV/Bohr³) at the converged divide-and-conquer density.
///
/// `σ_αβ = (1/Ω) ∂E/∂ε_αβ` with `Ω` the cell measure — volume in 3D, area in 2D, length in 1D.
///
/// # What it is and is not
///
/// Every term in this model is a function of pair separations alone, so the electronic and
/// core–core stresses are pair virials `Σ f_α δ_β` over the same image-aware pair list the
/// energy used; the long-range correction contributes its own strain derivative. All three are
/// taken from the **same pass** as the gradient rather than from a second loop, because a second
/// loop is a second chance to disagree about which pairs exist.
///
/// Like [`divide_conquer_gradient`], this is the fixed-density (Hellmann–Feynman) expression.
/// The divide-and-conquer density is not variational, so the `(∂E/∂P)(∂P/∂ε)` term does not
/// vanish and this is exact only in the limit where the buffer covers the cell —
/// `tests/dc_periodic.rs` measures that residual rather than asserting a tolerance.
///
/// A component touching a non-periodic axis is exactly zero: there is no cell length there to
/// differentiate with respect to.
pub fn divide_conquer_stress(
    molecule: &Molecule,
    params: &Am1Parameters,
    options: &Am1Options,
    result: &DcResult,
) -> Result<crate::math::Mat3> {
    let cell = molecule.cell.ok_or_else(|| {
        Am1Error::InvalidInput("a divide-and-conquer stress needs a periodic cell".into())
    })?;
    let basis = Basis::build(molecule, params)?;
    let neighbors =
        NeighborList::build_screened(molecule, options.realspace_cutoff, options.multipole_cutoff);

    let (_, mut virial) =
        crate::repulsion::core_core_gradient_and_virial(molecule, params, &neighbors)?;

    // The electronic virial contracts against whichever density the run produced: the total for
    // a restricted run, and the same total plus the reconstructed spin channels for an
    // unrestricted one, exactly as the gradient does.
    //
    // The open-shell branch used to be a refusal — the spin-resolved pair virial did not exist,
    // so the alternative was to contract the restricted expression against an open-shell density
    // and be wrong in the exchange channel by the spin split. It exists now
    // (`electronic_gradient_and_virial_fixed_density_spin`), and it is the same loop as the
    // restricted one with the exchange coefficient reading `Pα`/`Pβ` instead of half the total.
    let electronic = match &result.spin_density {
        None => {
            crate::gradient::electronic_gradient_and_virial_fixed_density(
                molecule,
                params,
                &basis,
                &neighbors,
                options.exchange_cutoff,
                &result.density,
            )?
            .1
        }
        Some(spin) => {
            // `Pα = (P_tot + S)/2`, `Pβ = (P_tot − S)/2`, as the unrestricted gradient does.
            let mut pa = result.density.clone();
            let mut pb = result.density.clone();
            {
                let (pas, pbs) = (pa.as_mut_slice(), pb.as_mut_slice());
                let (pts, ss) = (result.density.as_slice(), spin.as_slice());
                for i in 0..pts.len() {
                    pas[i] = 0.5 * (pts[i] + ss[i]);
                    pbs[i] = 0.5 * (pts[i] - ss[i]);
                }
            }
            crate::gradient::electronic_gradient_and_virial_fixed_density_spin(
                molecule,
                params,
                &basis,
                &neighbors,
                options.exchange_cutoff,
                &result.density,
                &pa,
                &pb,
            )?
            .1
        }
    };
    for (a, row) in virial.iter_mut().enumerate() {
        for (b, v) in row.iter_mut().enumerate() {
            *v += electronic[a][b];
        }
    }

    // Far-field monopole pairs, when the pair list was screened. Their virial is a pair virial
    // like the near ones, just over a cheaper kernel — but it lives here rather than in the
    // electronic pass because those pairs are not in the pair list at all.
    if let Some(far) =
        crate::farfield::FarField::new(molecule, params, options.multipole_cutoff.unwrap_or(0.0))?
    {
        let charges = crate::pbc::ewald::net_charges(molecule, &basis, params, &result.density)?;
        let far_virial = far.virial(&charges);
        for (a, row) in virial.iter_mut().enumerate() {
            for (b, v) in row.iter_mut().enumerate() {
                *v += far_virial[a][b];
            }
        }
    }

    // Long-range monopole correction: its own strain derivative, not a pair virial. Built *with*
    // the Klopman–Ohno tail, because the SCF this differentiates was converged with it — and the
    // tail's strain derivative is a separate term, since it is not a function of pair separation.
    if let Some((monopole, kernel)) = crate::pbc::ewald::LongRangeMonopole::for_molecule_with(
        molecule,
        options
            .klopman_ohno_tail
            .then_some((params, options.realspace_cutoff)),
        &neighbors,
        options.ewald,
    )? {
        let charges = crate::pbc::ewald::net_charges(molecule, &basis, params, &result.density)?;
        let strain = crate::pbc::ewald::LongRangeMonopole::energy_strain(
            molecule, &neighbors, &kernel, &charges,
        )?;
        let tail = monopole.klopman_ohno_strain(&charges);
        for (a, row) in virial.iter_mut().enumerate() {
            for (b, v) in row.iter_mut().enumerate() {
                *v += strain[a][b] + tail[a][b];
            }
        }
    }

    let measure = cell.measure();
    let mut stress = crate::math::Mat3::zero();
    for alpha in 0..3 {
        for beta in 0..3 {
            // A non-periodic direction has no cell length to differentiate.
            let v = if cell.periodic[alpha] && cell.periodic[beta] {
                virial[alpha][beta] / measure
            } else {
                0.0
            };
            let col = &mut stress.col[beta];
            match alpha {
                0 => col.x = v,
                1 => col.y = v,
                _ => col.z = v,
            }
        }
    }
    Ok(stress)
}

/// Nuclear gradient at the converged divide-and-conquer density.
///
/// This is the Hellmann–Feynman (fixed-density) gradient — the same expression the full SCF
/// uses, evaluated with the divide-and-conquer density in place of the variational one.
///
/// For a full SCF that expression is the *exact* derivative, because the energy is stationary
/// with respect to the density and the `(∂E/∂P)(dP/dR)` term vanishes identically. The
/// divide-and-conquer density is **not** stationary — it is assembled from separately
/// diagonalized blocks — so that term does not vanish, and this gradient is exact only in the
/// limit where the buffer covers the system.
///
/// `tests/divide_conquer.rs` measures the residual against a finite difference of the
/// divide-and-conquer energy itself and reports it as a function of buffer radius, rather than
/// asserting a tolerance that would only be describing one test system.
pub fn divide_conquer_gradient(
    molecule: &Molecule,
    params: &Am1Parameters,
    options: &Am1Options,
    result: &DcResult,
) -> Result<Vec<Vec3>> {
    let basis = Basis::build(molecule, params)?;
    // The same pair list the divide-and-conquer SCF used, so the force differentiates the energy
    // that was actually reported. Passing a molecular list here — which this did until the
    // periodic path was tested — silently produces a molecular gradient for a periodic cell.
    let neighbors =
        NeighborList::build_screened(molecule, options.realspace_cutoff, options.multipole_cutoff);
    let mut gradient =
        crate::repulsion::core_core_gradient_with_neighbors(molecule, params, &neighbors)?;

    let elec = match &result.spin_density {
        None => crate::gradient::electronic_gradient_fixed_density(
            molecule,
            params,
            &basis,
            &neighbors,
            options.exchange_cutoff,
            &result.density,
        )?,
        Some(spin) => {
            let mut pa = result.density.clone();
            let mut pb = result.density.clone();
            for (a, s) in pa.as_mut_slice().iter_mut().zip(spin.as_slice()) {
                *a = 0.5 * (*a + s);
            }
            for (b, s) in pb.as_mut_slice().iter_mut().zip(spin.as_slice()) {
                *b = 0.5 * (*b - s);
            }
            crate::gradient::electronic_gradient_fixed_density_spin(
                molecule,
                params,
                &basis,
                &neighbors,
                options.exchange_cutoff,
                &result.density,
                &pa,
                &pb,
            )?
        }
    };
    for (g, e) in gradient.iter_mut().zip(&elec) {
        *g += *e;
    }
    crate::gradient::add_long_range_force(
        molecule,
        params,
        options,
        &result.density,
        &mut gradient,
    )?;
    // The external field's force. `run_divide_conquer` already puts the field into `H_core`
    // through `options.core_build()`, so the energy carries it; without this the force would
    // differentiate a different energy from the one reported.
    crate::gradient::add_external_field_force(
        molecule,
        params,
        options,
        &result.density,
        &mut gradient,
    )?;
    Ok(gradient)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::system::Atom;

    fn chain(n: usize) -> Molecule {
        // A line of hydrogens, spaced so the partition has something to separate.
        Molecule::new(
            (0..n)
                .map(|i| Atom {
                    z: 1,
                    position: Vec3::new(i as f64 * 1.6, 0.0, 0.0),
                })
                .collect(),
        )
    }

    #[test]
    fn the_partition_is_disjoint_and_covers_every_atom() {
        let m = chain(37);
        let cores = partition_atoms(&m, 5);
        let mut seen = vec![0usize; 37];
        for core in &cores {
            assert!(core.len() <= 5, "a core exceeded the requested size");
            for &a in core {
                seen[a] += 1;
            }
        }
        assert!(
            seen.iter().all(|&c| c == 1),
            "every atom must belong to exactly one core, got {seen:?}"
        );
    }

    #[test]
    fn a_single_core_reproduces_the_whole_system() {
        let m = chain(6);
        let cores = partition_atoms(&m, 100);
        assert_eq!(cores.len(), 1);
        assert_eq!(cores[0].len(), 6);
    }
}

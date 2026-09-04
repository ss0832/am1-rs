// SPDX-License-Identifier: GPL-3.0-or-later

//! Far-field monopole treatment of the NDDO electrostatics.
//!
//! # What this replaces, and why it is allowed
//!
//! NDDO's two-centre electrostatics is three `1/R`-like pieces — electron–core attraction,
//! electron–electron Coulomb, core–core repulsion — each carrying a full `4 × 4 × 4 × 4` block
//! of Dewar–Sabelli–Klopman multipole integrals. They cannot be dropped by distance, because
//! `1/R` does not decay: screening them by a cutoff changes the answer rather than saving work
//! that did not matter. That is why the pair loop runs over **every** pair, and why it is the
//! measured bottleneck of a large calculation — 62 % of a 1029-atom divide-and-conquer run.
//!
//! What *can* be dropped is the multipole **structure**, not the interaction. Beyond a few atomic
//! diameters the whole block collapses onto its monopole term:
//!
//! ```text
//! Σ_μν∈a Σ_λσ∈b P_μν P_λσ (μν|λσ)  →  Q_a Q_b γ_ab(R),   γ_η(R) = e²/√(R² + η²)
//! ```
//!
//! with corrections of relative order `(d/R)²`, `d` the charge-separation parameter of the
//! multipole expansion (about 1 Bohr). So the interaction is kept in full and only its shape is
//! simplified — which turns a hundred-flop block into about ten flops.
//!
//! # How it enters
//!
//! Through the **net** charges, `V_a = Σ_{b far} γ_ab Q_b` with `Q_b = Z_b − p_b`, exactly as the
//! Ewald correction does and for the same reason: the electron–core and Coulomb pieces are
//! individually large and cancel, and applying them separately destroys the conditioning of the
//! SCF. See [`crate::fock::long_range_potential`] for the derivation of the matching
//! `+½ Σ_a Z_a V_a` energy term.
//!
//! # What it costs, stated
//!
//! This is an **approximation**, controlled by one parameter and off by default. The neglected
//! terms are the dipole and quadrupole channels beyond the cutoff, which fall as `(d/R)²`, so the
//! error shrinks quadratically as the cutoff grows. `tests/farfield.rs` measures it against the
//! unscreened calculation rather than bounding it by argument.
//!
//! The pair sum here is still `O(N²)` — it visits every distant pair — but with a per-pair cost
//! roughly a hundred times smaller. That moves the crossover; it does not remove the exponent.
//!
//! # Why there is no tree here, measured rather than argued
//!
//! A Barnes–Hut or particle-mesh evaluation of `V_a` would remove the exponent, and this module
//! is the natural place for one: by this point `V_a` is an ordinary classical potential of point
//! charges. It was not built, because the profile says it would be optimizing nothing.
//!
//! On a 1029-atom divide-and-conquer run with the far field **on** (`AM1_TIMING=1`, see
//! `tests/dc_where_the_time_goes.rs`), re-taken on the 0.2.1 code — the earlier table here was
//! measured before the sparse DIIS history, and `dc:diis` has since fallen by more than half:
//!
//! ```text
//!   dc:total           7.442 s   (wall)
//!   dc:diagonalize     2.790 s   37 %
//!   dc:fock            1.793 s   24 %
//!   dc:assemble        0.876 s   12 %
//!   dc:diis            0.689 s    9 %
//!   farfield:potential 0.104 s    1.4 %
//! ```
//!
//! The whole far-field sum is around one percent of the run, because a monopole pair costs about
//! ten flops and the loop is embarrassingly parallel. Making it `O(N log N)` would save about one
//! percent, and would add an acceptance-angle discontinuity to the energy surface to do it.
//!
//! The `O(N²)` term does eventually win — it grows quadratically while the rest of the run grows
//! roughly linearly — but the crossover is around `10⁴–10⁵` atoms, far outside what this code is
//! used for. This is precisely the trap `docs/divide-conquer.md` warns about: an `O(N²)` term
//! with a small prefactor sitting far below an `O(N)` term with a large one across the whole
//! range anybody runs. If that range changes, the measurement is the thing to redo first.

use crate::constants::AM1_EV;
use crate::error::Result;
use crate::math::Vec3;
use crate::params::Am1Parameters;
use crate::system::Molecule;

/// The atoms' positions and monopole parameters, prepared for the far-field sum.
#[derive(Clone, Debug)]
pub struct FarField {
    positions: Vec<Vec3>,
    /// `ρ⁰`, the Klopman–Ohno screening length of each atom (Bohr).
    rho0: Vec<f64>,
    /// Core charge `Z_a`.
    pub core_charges: Vec<f64>,
    /// Separation beyond which a pair is treated as a monopole (Bohr).
    pub cutoff: f64,
    /// Minimum-image cell, when the system is periodic.
    cell: Option<crate::lattice::Lattice>,
}

impl FarField {
    /// Prepare the far-field sum, or `None` when it does not apply.
    ///
    /// Returns `None` for `cutoff <= 0`, which is how a caller asks for the exact (unscreened)
    /// treatment.
    pub fn new(molecule: &Molecule, params: &Am1Parameters, cutoff: f64) -> Result<Option<Self>> {
        // Negated `>` rather than `<=`, deliberately: a NaN cutoff must disable the far field,
        // and `NaN <= 0.0` is false while `!(NaN > 0.0)` is true. Clippy flags the idiom because
        // the two differ on unordered values, which is exactly why it is the one used.
        #[allow(clippy::neg_cmp_op_on_partial_ord)]
        if !(cutoff > 0.0) {
            return Ok(None);
        }
        let mut rho0 = Vec::with_capacity(molecule.atoms.len());
        let mut core_charges = Vec::with_capacity(molecule.atoms.len());
        for atom in &molecule.atoms {
            let e = params.element(atom.z)?;
            rho0.push(e.rho0);
            core_charges.push(e.core_charge);
        }
        Ok(Some(Self {
            positions: molecule.atoms.iter().map(|a| a.position).collect(),
            rho0,
            core_charges,
            cutoff,
            cell: molecule.cell,
        }))
    }

    /// Separation between two atoms, honouring the minimum image under a cell.
    #[inline]
    fn separation(&self, a: usize, b: usize) -> Vec3 {
        let d = self.positions[b] - self.positions[a];
        match &self.cell {
            Some(cell) => cell.minimum_image(d),
            None => d,
        }
    }

    /// Build a Barnes–Hut tree over the atoms, for an `O(N log N)` far field.
    ///
    /// `theta` is the acceptance angle: a node is used whole when its radius over its distance is
    /// below it. Smaller is more accurate and slower; `0` would open every node and reduce to the
    /// direct sum. See [`FarFieldTree`] for what the approximation is and what it costs.
    ///
    /// Returns `None` when there is nothing to build a tree over.
    pub fn tree(&self, theta: f64) -> Option<FarFieldTree> {
        FarFieldTree::build(self, theta)
    }

    /// `V_a = Σ_{b : |R_ab| > cutoff} γ_ab(R) Q_b`, in eV per unit charge.
    ///
    /// The self term is excluded: an atom does not interact with itself through this channel.
    pub fn potential(&self, charges: &[f64]) -> Vec<f64> {
        use rayon::prelude::*;
        let _t = crate::timing::Timer::start("farfield:potential");
        let n = self.positions.len();
        let cut2 = self.cutoff * self.cutoff;
        (0..n)
            .into_par_iter()
            .map(|a| {
                let mut acc = 0.0;
                for b in 0..n {
                    if b == a {
                        continue;
                    }
                    let d = self.separation(a, b);
                    let r2 = d.norm2();
                    if r2 <= cut2 {
                        continue;
                    }
                    let eta = self.rho0[a] + self.rho0[b];
                    acc += AM1_EV / (r2 + eta * eta).sqrt() * charges[b];
                }
                acc
            })
            .collect()
    }

    /// `∂/∂R_c` of `½ Σ_ab Q_a Q_b γ_ab`, restricted to far pairs — eV/Bohr, per atom.
    ///
    /// Hellmann–Feynman at the converged density: the explicit position derivative only, because
    /// the density-response term vanishes at a stationary point.
    pub fn gradient(&self, charges: &[f64]) -> Vec<Vec3> {
        use rayon::prelude::*;
        let n = self.positions.len();
        let cut2 = self.cutoff * self.cutoff;
        (0..n)
            .into_par_iter()
            .map(|a| {
                let mut acc = Vec3::zero();
                for b in 0..n {
                    if b == a {
                        continue;
                    }
                    let d = self.separation(a, b);
                    let r2 = d.norm2();
                    if r2 <= cut2 {
                        continue;
                    }
                    let eta = self.rho0[a] + self.rho0[b];
                    let s = (r2 + eta * eta).sqrt();
                    // d/dR_a [ 1/s ] = +d/s³ with d = R_b − R_a.
                    acc += d * (AM1_EV / (s * s * s) * charges[b]);
                }
                acc * charges[a]
            })
            .collect()
    }

    /// `∂/∂ε_αβ` of `½ Σ_ab Q_a Q_b γ_ab` over far pairs — the virial the periodic stress needs.
    pub fn virial(&self, charges: &[f64]) -> [[f64; 3]; 3] {
        let n = self.positions.len();
        let cut2 = self.cutoff * self.cutoff;
        let mut virial = [[0.0_f64; 3]; 3];
        for a in 0..n {
            for b in 0..n {
                if b == a {
                    continue;
                }
                let d = self.separation(a, b);
                let r2 = d.norm2();
                if r2 <= cut2 {
                    continue;
                }
                let eta = self.rho0[a] + self.rho0[b];
                let s = (r2 + eta * eta).sqrt();
                // ½ from the double count over ordered pairs; f = dE/d(delta).
                let w = -0.5 * AM1_EV / (s * s * s) * charges[a] * charges[b];
                let dv = [d.x, d.y, d.z];
                for (alpha, row) in virial.iter_mut().enumerate() {
                    for (beta, v) in row.iter_mut().enumerate() {
                        *v += w * dv[alpha] * dv[beta];
                    }
                }
            }
        }
        virial
    }

    /// How many pairs the cutoff sends to the monopole treatment, and how many stay exact.
    ///
    /// Reported so a caller can see what the approximation is actually doing on their system
    /// rather than inferring it from the cutoff.
    pub fn pair_counts(&self) -> (usize, usize) {
        let n = self.positions.len();
        let cut2 = self.cutoff * self.cutoff;
        let mut far = 0;
        let mut near = 0;
        for a in 0..n {
            for b in (a + 1)..n {
                if self.separation(a, b).norm2() > cut2 {
                    far += 1;
                } else {
                    near += 1;
                }
            }
        }
        (near, far)
    }
}

/// A Barnes–Hut tree over the far field, which is what makes the NDDO Coulomb `O(N log N)`.
///
/// # What was `O(N²)` and why this removes it
///
/// [`FarField`] keeps the interaction in full and simplifies only its *shape*, so it still visits
/// every distant pair: the prefactor falls about a hundredfold and the exponent does not move.
/// `docs/scope.md` recorded that as "linear-scaling Coulomb ⛔ — stays `O(N²)` by construction",
/// and it was true.
///
/// By the time the far field is reached, `V_a = Σ_b γ_ab Q_b` is an ordinary classical potential of
/// point charges, so the standard treatment applies: group distant atoms into cells and interact
/// with a cell rather than with its members.
///
/// # The cluster is a pseudo-atom, not a multipole expansion
///
/// The obvious implementation expands `γ` in `1/R` and carries moments. That does not work cleanly
/// here, because `γ_ab = e²/√(R² + (ρ_a + ρ_b)²)` depends on **both** atoms' screening lengths, so
/// a cluster's contribution is not a function of its charge and position alone.
///
/// Instead each accepted node is replaced by a single **pseudo-atom**: charge `Σ_b Q_b`, position
/// the charge-weighted centre, and screening length the `|Q|`-weighted mean `Σ|Q_b|ρ_b / Σ|Q_b|`
/// (weighted by modulus so a near-neutral cluster still has a defined length). Every consumer then
/// evaluates *the same pair kernel it always did*, against a shorter list of partners.
///
/// That is the property worth having: [`Self::potential`], [`Self::gradient`] and
/// [`Self::virial`] cannot drift apart, because they are the untouched pair expressions applied to
/// the same substitution. An expansion carried to different orders in three places is how an energy
/// and its forces stop matching.
///
/// # What it costs
///
/// Two approximations, both of order `(size/R)²`: the positions inside a node collapse to one
/// point, and the spread of `ρ` inside it collapses to one value. The far field **already** drops
/// terms of that order — it is what "keep the interaction, simplify the shape" means — so the tree
/// adds no new order of error, only a new prefactor on the same one. `theta` controls it and
/// `tests/farfield_tree.rs` measures the error against the direct sum rather than bounding it.
///
/// A node is accepted only when it lies **entirely** beyond the far-field cutoff
/// (`distance − radius > cutoff`). Nodes straddling it are opened to their leaves, so the set of
/// pairs treated as near is exactly the set the direct sum treats as near — the tree changes how
/// the far pairs are summed and never which pairs are far.
///
/// # The discontinuity, stated
///
/// An acceptance angle makes the energy a discontinuous function of the geometry: an atom crossing
/// the acceptance boundary switches between a cluster and its members. The jump is of the order of
/// the truncation error, but it is a jump, and it is why this is **opt-in** rather than the
/// default. For molecular dynamics either leave it off or accept a force that is not exactly the
/// gradient of the reported energy at those points.
#[derive(Clone, Debug)]
pub struct FarFieldTree {
    nodes: Vec<Node>,
    /// Atom indices, permuted so each node owns a contiguous range.
    order: Vec<usize>,
    theta: f64,
}

#[derive(Clone, Debug)]
struct Node {
    /// Half-extent of the bounding box, as a radius about `centre_geometric`.
    radius: f64,
    centre_geometric: Vec3,
    /// `[start, end)` into `order`.
    start: usize,
    end: usize,
    /// Child node indices, or `None` for a leaf.
    children: Option<[usize; 8]>,
}

/// One accepted cluster, as the pair kernel sees it.
#[derive(Clone, Copy, Debug)]
pub struct PseudoAtom {
    pub position: Vec3,
    pub charge: f64,
    pub rho0: f64,
}

/// Atoms below this per node are kept as a leaf: an octree of singletons costs more in traversal
/// than the pairs it saves.
const LEAF_SIZE: usize = 8;

impl FarFieldTree {
    fn build(field: &FarField, theta: f64) -> Option<Self> {
        let n = field.positions.len();
        if n == 0 {
            return None;
        }
        let mut order: Vec<usize> = (0..n).collect();
        let mut nodes = Vec::new();
        Self::split(&field.positions, &mut order, 0, n, &mut nodes);
        Some(Self {
            nodes,
            order,
            theta: theta.max(0.0),
        })
    }

    /// Recursively bisect `[start, end)` of `order` into an octree node, returning its index.
    fn split(
        positions: &[Vec3],
        order: &mut [usize],
        start: usize,
        end: usize,
        nodes: &mut Vec<Node>,
    ) -> usize {
        let mut lo = Vec3::new(f64::MAX, f64::MAX, f64::MAX);
        let mut hi = Vec3::new(f64::MIN, f64::MIN, f64::MIN);
        for &i in &order[start..end] {
            let p = positions[i];
            lo = Vec3::new(lo.x.min(p.x), lo.y.min(p.y), lo.z.min(p.z));
            hi = Vec3::new(hi.x.max(p.x), hi.y.max(p.y), hi.z.max(p.z));
        }
        let centre = (lo + hi) * 0.5;
        let radius = ((hi - lo) * 0.5).norm();

        let index = nodes.len();
        nodes.push(Node {
            radius,
            centre_geometric: centre,
            start,
            end,
            children: None,
        });
        if end - start <= LEAF_SIZE {
            return index;
        }

        // Partition into octants about the centre, in place.
        let mut buckets: [Vec<usize>; 8] = Default::default();
        for &i in &order[start..end] {
            let p = positions[i];
            let octant = usize::from(p.x >= centre.x)
                | (usize::from(p.y >= centre.y) << 1)
                | (usize::from(p.z >= centre.z) << 2);
            buckets[octant].push(i);
        }
        // A degenerate split — every atom in one octant, which happens for coincident points —
        // would recurse forever. Keep it a leaf instead.
        if buckets.iter().any(|b| b.len() == end - start) {
            return index;
        }
        let mut cursor = start;
        let mut ranges = [(0usize, 0usize); 8];
        for (o, bucket) in buckets.iter().enumerate() {
            let s = cursor;
            for &i in bucket {
                order[cursor] = i;
                cursor += 1;
            }
            ranges[o] = (s, cursor);
        }
        let mut children = [usize::MAX; 8];
        for (o, (s, e)) in ranges.iter().copied().enumerate() {
            children[o] = if e > s {
                Self::split(positions, order, s, e, nodes)
            } else {
                usize::MAX
            };
        }
        nodes[index].children = Some(children);
        index
    }

    /// The partners atom `a` should interact with: exact atoms for the near ones and inside opened
    /// nodes, pseudo-atoms for accepted clusters.
    ///
    /// Pairs closer than the far-field cutoff are **excluded** here, exactly as the direct sum
    /// excludes them — they are the near field and belong to the pair loop.
    fn partners(&self, field: &FarField, charges: &[f64], a: usize, out: &mut Vec<PseudoAtom>) {
        out.clear();
        let mut stack = vec![0usize];
        let cut = field.cutoff;
        while let Some(index) = stack.pop() {
            let node = &self.nodes[index];
            if node.end == node.start {
                continue;
            }
            let d = match &field.cell {
                Some(cell) => cell.minimum_image(node.centre_geometric - field.positions[a]),
                None => node.centre_geometric - field.positions[a],
            };
            let dist = d.norm();
            // Accept only if the whole node is beyond the cutoff *and* it subtends a small enough
            // angle. Both conditions, because either alone admits a node that is partly near.
            let separated = dist - node.radius > cut;
            let small = node.radius < self.theta * dist;
            if separated && small && node.end - node.start > 2 {
                // **Two** pseudo-atoms, not one: the positive charge at its centroid and the
                // negative at its own.
                //
                // One would be a monopole expansion, and a monopole expansion of a molecular
                // cluster is worthless — the clusters here are made of *neutral molecules*, so the
                // net charge is around zero and the whole interaction is dipolar. The first draft
                // did exactly that and the error against the direct sum was 64 % and did not
                // shrink with the acceptance angle, because there was no monopole for the angle to
                // resolve.
                //
                // Splitting by sign carries the dipole exactly to leading order while keeping the
                // property that makes this design safe: every consumer still evaluates the
                // ordinary pair kernel against a shorter list, so the potential, the gradient and
                // the virial cannot drift apart.
                let mut acc = [(0.0_f64, 0.0_f64, Vec3::zero(), 0.0_f64); 2];
                for &b in &self.order[node.start..node.end] {
                    let q = charges[b];
                    let slot = usize::from(q < 0.0);
                    let w = q.abs();
                    acc[slot].0 += q;
                    acc[slot].1 += w;
                    acc[slot].2 += field.positions[b] * w;
                    acc[slot].3 += w * field.rho0[b];
                }
                for (charge, weight, centre, rho) in acc {
                    if weight > 1.0e-30 {
                        out.push(PseudoAtom {
                            position: centre * (1.0 / weight),
                            charge,
                            rho0: rho / weight,
                        });
                    }
                }
                continue;
            }
            match node.children {
                Some(children) => {
                    for c in children {
                        if c != usize::MAX {
                            stack.push(c);
                        }
                    }
                }
                None => {
                    for &b in &self.order[node.start..node.end] {
                        if b == a {
                            continue;
                        }
                        let sep = field.separation(a, b);
                        if sep.norm2() <= cut * cut {
                            continue; // near field: the pair loop owns it
                        }
                        out.push(PseudoAtom {
                            position: field.positions[a] + sep,
                            charge: charges[b],
                            rho0: field.rho0[b],
                        });
                    }
                }
            }
        }
    }

    /// [`FarField::potential`], evaluated through the tree.
    pub fn potential(&self, field: &FarField, charges: &[f64]) -> Vec<f64> {
        use rayon::prelude::*;
        let _t = crate::timing::Timer::start("farfield:potential_tree");
        (0..field.positions.len())
            .into_par_iter()
            .map_with(Vec::new(), |scratch, a| {
                self.partners(field, charges, a, scratch);
                let mut acc = 0.0;
                for p in scratch.iter() {
                    let r2 = (p.position - field.positions[a]).norm2();
                    let eta = field.rho0[a] + p.rho0;
                    acc += AM1_EV / (r2 + eta * eta).sqrt() * p.charge;
                }
                acc
            })
            .collect()
    }

    /// [`FarField::gradient`], evaluated through the tree.
    ///
    /// The **same** pair expression the direct sum uses, applied to the pseudo-atoms — which is
    /// what keeps it the gradient of [`Self::potential`]'s energy rather than a second derivation.
    pub fn gradient(&self, field: &FarField, charges: &[f64]) -> Vec<Vec3> {
        use rayon::prelude::*;
        (0..field.positions.len())
            .into_par_iter()
            .map_with(Vec::new(), |scratch, a| {
                self.partners(field, charges, a, scratch);
                let mut acc = Vec3::zero();
                for p in scratch.iter() {
                    let d = p.position - field.positions[a];
                    let r2 = d.norm2();
                    let eta = field.rho0[a] + p.rho0;
                    let s = (r2 + eta * eta).sqrt();
                    // ∂/∂R_a of γ(|R_b − R_a|) Q_a Q_b, summed over the partners of `a`.
                    acc += d * (AM1_EV * charges[a] * p.charge / (s * s * s));
                }
                acc
            })
            .collect()
    }

    /// How many partner evaluations the tree performs, summed over atoms.
    ///
    /// The operation count the scaling claim rests on. Reported rather than timed, because on a
    /// loaded machine a stopwatch measures the load — the same discipline the divide-and-conquer
    /// counters use.
    pub fn partner_evaluations(&self, field: &FarField, charges: &[f64]) -> usize {
        let mut scratch = Vec::new();
        let mut total = 0;
        for a in 0..field.positions.len() {
            self.partners(field, charges, a, &mut scratch);
            total += scratch.len();
        }
        total
    }
}

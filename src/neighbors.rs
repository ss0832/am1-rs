// SPDX-License-Identifier: GPL-3.0-or-later

//! Image-aware pair list.
//!
//! The molecular code enumerates pairs as `for i { for j in i+1.. }`, which is fine when
//! every pair exists once. Under a periodic cell an atom also interacts with *its own*
//! images, and each physical pair `(i, j, T)` has a mirror `(j, i, −T)` describing the same
//! interaction. Getting that bookkeeping wrong double-counts the energy in a way that looks
//! like a parameter error.
//!
//! So pairs are enumerated once each, by a canonical rule (see [`is_canonical`]), and the
//! molecular case falls out as the special case of a single origin translation — the same
//! list type serves both, which is what keeps the periodic and molecular paths from drifting
//! apart.

use rayon::prelude::*;

use crate::lattice::{ImageOffset, Lattice};
use crate::math::Vec3;
use crate::system::Molecule;

/// One directed interaction: home-cell atom `i` with atom `j` displaced by lattice
/// translation `offset`.
#[derive(Clone, Copy, Debug)]
pub struct PairImage {
    pub i: usize,
    pub j: usize,
    pub offset: ImageOffset,
    /// `r_j + T − r_i`, Bohr.
    pub delta: Vec3,
    /// `|delta|`, Bohr.
    pub r: f64,
}

impl PairImage {
    /// Whether this is a self-interaction with one of the atom's own periodic images.
    #[inline]
    pub fn is_self_image(&self) -> bool {
        self.i == self.j && !self.offset.is_origin()
    }
}

/// Whether `(i, j, T)` is the representative of its physical pair.
///
/// Each pair appears twice in a naive enumeration — as `(i, j, T)` and as `(j, i, −T)` — so
/// exactly one must be kept. The rule: keep it if the translation is lexicographically
/// positive; if the translation is zero, keep it when `i < j`. For a self-image pair
/// `(i, i, ±T)` this keeps precisely one of the two, which is what makes an atom's
/// interaction with its own image count once rather than twice.
#[inline]
pub fn is_canonical(i: usize, j: usize, offset: ImageOffset) -> bool {
    for c in offset.n {
        if c > 0 {
            return true;
        }
        if c < 0 {
            return false;
        }
    }
    i < j
}

/// Every physical pair within `cutoff`, each listed once.
#[derive(Clone, Debug)]
pub struct NeighborList {
    pub pairs: Vec<PairImage>,
    pub cutoff: f64,
    /// Translations that were considered. `[origin]` for a molecule.
    pub translations: Vec<ImageOffset>,
}

impl NeighborList {
    /// Build the list for `molecule`. Uses its cell when it has one; otherwise every pair is
    /// included regardless of `cutoff`, which keeps the molecular result exact.
    pub fn build(molecule: &Molecule, cutoff: f64) -> Self {
        match molecule.cell {
            Some(cell) if cell.n_periodic() > 0 => Self::periodic(molecule, &cell, cutoff),
            _ => Self::molecular(molecule),
        }
    }

    /// [`Self::build`], with pairs beyond `far_cutoff` removed.
    ///
    /// Removing them is only legitimate when something else accounts for their interaction —
    /// [`crate::farfield::FarField`] does, as a monopole — which is why this is a separate
    /// constructor rather than an argument to [`Self::build`]. Calling it without the matching
    /// far-field term silently discards `1/R` interactions and changes the answer.
    ///
    /// `None` keeps every pair, which is [`Self::build`]'s behaviour.
    pub fn build_screened(molecule: &Molecule, cutoff: f64, far_cutoff: Option<f64>) -> Self {
        let mut list = Self::build(molecule, cutoff);
        let Some(far) = far_cutoff else {
            return list;
        };
        // Negated `>` rather than `<=` so that a NaN radius means "no screening" instead of
        // silently keeping every pair; see the same idiom in `farfield`.
        #[allow(clippy::neg_cmp_op_on_partial_ord)]
        if !(far > 0.0) {
            return list;
        }
        list.pairs.retain(|p| p.r <= far);
        list
    }

    /// All `i < j` pairs, no cutoff.
    ///
    /// Deliberately not screened by distance: the NDDO two-centre two-electron integrals decay
    /// as `1/R`, so discarding a distant pair changes the energy rather than saving work that
    /// did not matter. Screening those needs the multipole/Ewald split, not a cutoff.
    pub fn molecular(molecule: &Molecule) -> Self {
        let n = molecule.atoms.len();
        let mut pairs = Vec::with_capacity(n * n / 2);
        for i in 0..n {
            for j in (i + 1)..n {
                let delta = molecule.atoms[j].position - molecule.atoms[i].position;
                pairs.push(PairImage {
                    i,
                    j,
                    offset: ImageOffset::origin(),
                    delta,
                    r: delta.norm(),
                });
            }
        }
        Self {
            pairs,
            cutoff: f64::INFINITY,
            translations: vec![ImageOffset::origin()],
        }
    }

    /// Periodic list, truncated **by lattice translation** rather than by pair distance.
    ///
    /// This distinction is the whole correctness of the sum, not a tuning choice.
    ///
    /// The NDDO electrostatics is three `1/R` pieces that cancel: the electron–core
    /// attraction, the electron–electron Coulomb, and the core–core repulsion. For a neutral
    /// cell their monopole parts sum to `Σ_ab Q_a Q_b γ` with `Σ_a Q_a = 0`, so the leading
    /// term vanishes and what remains is a rapidly-decaying multipole series. That
    /// cancellation is a statement about a **whole image**: it needs every atom pair of a
    /// given translation, or nothing from it.
    ///
    /// A cutoff on the pair distance slices through image shells — it keeps an oxygen's
    /// attraction to a distant image while dropping a hydrogen's — and destroys the
    /// cancellation. Measured, on one water in a cubic cell with a pair-distance cutoff, the
    /// energy came out 0.5 Hartree below the molecular value at a 30 Bohr cell and diverged
    /// by 24 Hartree at 12 Bohr, where the true interaction is millihartrees.
    ///
    /// Cutting on `|T|` keeps each image intact, so a neutral cell stays neutral and the
    /// cancellation survives.
    fn periodic(molecule: &Molecule, cell: &Lattice, cutoff: f64) -> Self {
        let n = molecule.atoms.len();
        let translations = cell.image_offsets(cutoff);

        let pairs: Vec<PairImage> = translations
            .par_iter()
            .flat_map_iter(|&offset| {
                let t = cell.translation(offset);
                let mut local = Vec::new();
                for i in 0..n {
                    for j in 0..n {
                        if !is_canonical(i, j, offset) {
                            continue;
                        }
                        let delta = molecule.atoms[j].position + t - molecule.atoms[i].position;
                        let r = delta.norm();
                        // The only rejection is a coincident pair, which would be an atom on
                        // top of its own image and is a broken structure, not a far neighbour.
                        if r > 1.0e-8 {
                            local.push(PairImage {
                                i,
                                j,
                                offset,
                                delta,
                                r,
                            });
                        }
                    }
                }
                local
            })
            .collect();

        Self {
            pairs,
            cutoff,
            translations,
        }
    }

    pub fn len(&self) -> usize {
        self.pairs.len()
    }
    pub fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }

    /// Largest pair separation actually included, Bohr.
    pub fn max_distance(&self) -> f64 {
        self.pairs.iter().map(|p| p.r).fold(0.0, f64::max)
    }

    /// How many of the pairs are an atom with one of its own images.
    pub fn self_image_count(&self) -> usize {
        self.pairs.iter().filter(|p| p.is_self_image()).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::system::Atom;

    fn atoms_at(points: &[[f64; 3]]) -> Vec<Atom> {
        points
            .iter()
            .map(|p| Atom {
                z: 1,
                position: Vec3::new(p[0], p[1], p[2]),
            })
            .collect()
    }

    #[test]
    fn the_canonical_rule_keeps_each_physical_pair_exactly_once() {
        // For every (i, j, T) exactly one of it and its mirror (j, i, -T) is canonical.
        for i in 0..3usize {
            for j in 0..3usize {
                for a in -1..=1 {
                    for b in -1..=1 {
                        for c in -1..=1 {
                            let t = ImageOffset { n: [a, b, c] };
                            if i == j && t.is_origin() {
                                continue; // not a pair at all
                            }
                            let forward = is_canonical(i, j, t);
                            let mirror = is_canonical(j, i, t.negated());
                            assert!(
                                forward ^ mirror,
                                "(i={i}, j={j}, T={:?}) and its mirror are both {} canonical",
                                t.n,
                                if forward { "" } else { "not" }
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn a_molecule_gets_every_pair_and_no_images() {
        let m = Molecule::new(atoms_at(&[
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
        ]));
        let nl = NeighborList::build(&m, 10.0);
        assert_eq!(nl.len(), 3, "three atoms give three pairs");
        assert!(nl.pairs.iter().all(|p| p.offset.is_origin()));
        assert_eq!(nl.self_image_count(), 0);
    }

    #[test]
    fn a_periodic_cell_includes_self_images() {
        // One atom in a cubic cell: it has no partners at all except its own images.
        let cell = Lattice::cubic(6.0).unwrap();
        let m = Molecule::new(atoms_at(&[[0.0, 0.0, 0.0]])).with_cell(cell);
        let nl = NeighborList::build(&m, 7.0);
        assert!(!nl.is_empty(), "a lone atom in a cell still has neighbours");
        assert_eq!(nl.self_image_count(), nl.len());
        // The six nearest images at distance 6.0 appear as three canonical pairs (+x, +y, +z);
        // their mirrors are the same physical interactions.
        let at_six = nl
            .pairs
            .iter()
            .filter(|p| (p.r - 6.0).abs() < 1.0e-9)
            .count();
        assert_eq!(
            at_six, 3,
            "expected 3 canonical nearest-image pairs, got {at_six}"
        );
    }

    #[test]
    fn the_cutoff_bounds_the_translation_not_the_pair_distance() {
        // The cutoff applies to |T|, so a pair can be further apart than the cutoff by up to
        // the span of the cell's contents — that is the point of the scheme, since keeping
        // whole images is what preserves charge neutrality.
        let cell = Lattice::cubic(6.0).unwrap();
        let m = Molecule::new(atoms_at(&[[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]])).with_cell(cell);
        let cutoff = 12.0;
        let nl = NeighborList::build(&m, cutoff);

        for p in &nl.pairs {
            let t = cell.translation(p.offset).norm();
            assert!(
                t <= cutoff + 1.0e-12,
                "translation {t} exceeds the cutoff {cutoff}"
            );
        }
        assert!(
            nl.max_distance() > cutoff,
            "with a 2.5 Bohr intra-cell span, some pair should reach past the {cutoff} Bohr \
             translation cutoff; the largest was {}",
            nl.max_distance()
        );
    }

    #[test]
    fn none_are_duplicated() {
        let cell = Lattice::from_vectors(
            Vec3::new(7.0, 0.0, 0.0),
            Vec3::new(2.0, 6.5, 0.0),
            Vec3::new(0.0, 1.0, 8.0),
            [true, true, true],
        )
        .unwrap();
        let m = Molecule::new(atoms_at(&[
            [0.0, 0.0, 0.0],
            [1.4, 0.3, 0.2],
            [3.0, 2.0, 1.0],
        ]))
        .with_cell(cell);
        let cutoff = 12.0;
        let nl = NeighborList::build(&m, cutoff);

        // No physical pair listed twice, counting (i,j,T) and (j,i,-T) as the same thing.
        let mut seen = std::collections::HashSet::new();
        for p in &nl.pairs {
            let key = if (p.i, p.offset.n) <= (p.j, p.offset.negated().n) {
                (p.i, p.j, p.offset.n)
            } else {
                (p.j, p.i, p.offset.negated().n)
            };
            assert!(seen.insert(key), "duplicate pair {:?}", key);
        }
    }

    #[test]
    fn the_pair_count_matches_a_brute_force_enumeration() {
        // Independent check of the canonical rule: count every (i, j, T) within the cutoff
        // including both orientations, and require exactly twice the canonical count.
        let cell = Lattice::cubic(5.0).unwrap();
        let m = Molecule::new(atoms_at(&[[0.0, 0.0, 0.0], [1.2, 0.7, 0.4]])).with_cell(cell);
        let cutoff = 9.0;
        let nl = NeighborList::build(&m, cutoff);

        // Brute force over the same rule the list uses: a translation is in if |T| <= cutoff,
        // and then *every* atom pair of that image is in.
        let mut brute = 0usize;
        let span = 6;
        for a in -span..=span {
            for b in -span..=span {
                for c in -span..=span {
                    let offset = ImageOffset { n: [a, b, c] };
                    let t = cell.translation(offset);
                    if t.norm() > cutoff {
                        continue;
                    }
                    for i in 0..2 {
                        for j in 0..2 {
                            let d = m.atoms[j].position + t - m.atoms[i].position;
                            if d.norm() > 1.0e-8 {
                                brute += 1;
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(
            2 * nl.len(),
            brute,
            "canonical list has {} pairs; brute force found {brute} directed interactions",
            nl.len()
        );
    }

    #[test]
    fn a_slab_never_translates_along_the_free_axis() {
        let cell = Lattice::from_vectors(
            Vec3::new(5.0, 0.0, 0.0),
            Vec3::new(0.0, 5.0, 0.0),
            Vec3::new(0.0, 0.0, 30.0),
            [true, true, false],
        )
        .unwrap();
        let m = Molecule::new(atoms_at(&[[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]])).with_cell(cell);
        let nl = NeighborList::build(&m, 12.0);
        assert!(nl.pairs.iter().all(|p| p.offset.n[2] == 0));
        assert!(
            nl.len() > 2,
            "the periodic directions should still generate images"
        );
    }
}

// SPDX-License-Identifier: GPL-3.0-or-later

//! Molecular-graph perception from geometry: bonds (covalent-radius overlap), per-atom
//! coordination and hybridization, ring perception with **ring sizes**, Hückel-based
//! aromaticity, delocalized-group detection, and a bond-order guess. This feeds the AM1-BCC
//! atom/bond typing in [`crate::bcc`].
//!
//! The perception is geometry plus valence heuristics, not a full cheminformatics toolkit. What
//! it does *not* recognize it reports, rather than guessing silently — see [`Topology::warnings`].

use crate::constants::{covalent_radius_angstrom, BOHR_TO_ANGSTROM};
use crate::system::{z_to_symbol, Molecule};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BondOrder {
    Single,
    Double,
    Triple,
    Aromatic,
    /// A bond in a symmetric delocalized group — a carboxylate, nitro, phosphate, sulfonate or
    /// N-oxide — where the central atom carries two or more equivalent terminal O/S atoms (or is
    /// a four-coordinate nitrogen with one).
    ///
    /// This is a distinct bond order in AM1-BCC, not a rounding of single and double, and the
    /// distinction is worth a variant: the tabulated correction for a delocalized C–O
    /// (`0.16`–`0.34` depending on the carbon type) differs from both the single-bond value
    /// (`0.03`–`0.17`) and the double-bond one, by up to 0.2 e on the bond.
    Delocalized,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Hybridization {
    Sp3,
    Sp2,
    Sp,
    None,
}

#[derive(Clone, Debug)]
pub struct Bond {
    pub i: usize,
    pub j: usize,
    pub order: BondOrder,
    /// Distance in Ångström.
    pub length: f64,
}

/// A perceived ring, in cyclic order.
#[derive(Clone, Debug)]
pub struct Ring {
    pub atoms: Vec<usize>,
    pub aromatic: bool,
}

impl Ring {
    pub fn size(&self) -> usize {
        self.atoms.len()
    }
}

#[derive(Clone, Debug)]
pub struct Topology {
    pub bonds: Vec<Bond>,
    /// Neighbour atom indices per atom.
    pub neighbors: Vec<Vec<usize>>,
    pub hybridization: Vec<Hybridization>,
    pub aromatic: Vec<bool>,
    pub in_ring: Vec<bool>,
    /// The perceived rings, smallest first.
    pub rings: Vec<Ring>,
    /// Size of the smallest ring containing each atom, if any.
    pub smallest_ring: Vec<Option<usize>>,
    /// Per bond, whether a Kekulé structure makes it a **double** bond.
    ///
    /// Parallel to [`Self::bonds`]. For a non-aromatic bond this just mirrors
    /// `order == BondOrder::Double`; for an aromatic one it is the assignment found by
    /// [`kekule_double_bonds`], because `ATOMTYPE_BCC.DEF` distinguishes an atom with two formal
    /// single bonds (`[2sb]`) from one with a single and a double (`[sb,db]`) and an aromatic ring
    /// bond is one or the other, never both.
    ///
    /// This is what separates pyridine's nitrogen (one formal single, one formal double, hence
    /// type 24) from an aliphatic two-coordinate nitrogen (two singles, type 21). The parameter
    /// file settles which is intended: it carries `17–24` at bond type 7, an aromatic bond between
    /// a type-17 carbon — whose own rule *requires* an aromatic two-connected nitrogen neighbour —
    /// and a type-24 nitrogen. That pair is pyridine and nothing else.
    pub kekule_double: Vec<bool>,
    /// Whether the Kekulé structure in [`Self::kekule_double`] is the **only** one.
    ///
    /// `false` covers both ways it can fail to determine anything: no perfect matching exists (the
    /// assignment is then the all-single fallback), or more than one exists and the one found is an
    /// artifact of the search order. Benzene is the second case — its two Kekulé structures are
    /// equivalent, so calling one particular bond "single" says nothing about the molecule.
    ///
    /// AM1-BCC has a bond type for exactly this distinction: 7 is "aromatic single", 8 is
    /// "aromatic double", and 10 is plain "aromatic" — a bond that is aromatic *without* a resolved
    /// single/double character. So this is what selects between them. See [`crate::bcc`].
    pub kekule_unique: bool,
    /// Anything the perception could not do confidently. Empty for a molecule it fully
    /// understands. [`crate::bcc`] adds to this and surfaces it on the result, because a
    /// silently mistyped atom is the failure mode worth preventing.
    pub warnings: Vec<String>,
}

impl Topology {
    pub fn perceive(molecule: &Molecule) -> Self {
        let nat = molecule.atoms.len();
        let mut neighbors = vec![Vec::new(); nat];
        let mut bonds = Vec::new();
        let mut warnings = Vec::new();

        for i in 0..nat {
            for j in (i + 1)..nat {
                let d = (molecule.atoms[j].position - molecule.atoms[i].position).norm()
                    * BOHR_TO_ANGSTROM;
                let ri = covalent_radius_angstrom(molecule.atoms[i].z);
                let rj = covalent_radius_angstrom(molecule.atoms[j].z);
                if d < 1.3 * (ri + rj) && d > 0.4 {
                    neighbors[i].push(j);
                    neighbors[j].push(i);
                    bonds.push(Bond {
                        i,
                        j,
                        order: BondOrder::Single, // refined below
                        length: d,
                    });
                }
            }
        }

        // Rings first: aromaticity needs ring sizes, and the old code could not ask for them
        // because it only tracked a boolean.
        let rings_raw = smallest_rings(nat, &bonds);
        let mut in_ring = vec![false; nat];
        let mut smallest_ring: Vec<Option<usize>> = vec![None; nat];
        for ring in &rings_raw {
            for &a in ring {
                in_ring[a] = true;
                smallest_ring[a] = Some(match smallest_ring[a] {
                    Some(s) => s.min(ring.len()),
                    None => ring.len(),
                });
            }
        }

        let mut hybridization = vec![Hybridization::None; nat];
        for i in 0..nat {
            hybridization[i] = perceive_hybridization(molecule, &neighbors, i);
        }

        // Bond orders from length, now that the pair table covers more than C/N/O.
        for b in &mut bonds {
            b.order = guess_bond_order(molecule.atoms[b.i].z, molecule.atoms[b.j].z, b.length);
        }

        // Aromaticity, per ring, by size + planarity + a Hückel 4n+2 π count.
        //
        // One class, not five. `ATOMTYPE_BCC.DEF` defines AR1..AR5 but every rule in it asks for
        // the *union* `[AR1.AR2]` and none asks for either alone, so the sub-classification would
        // be machinery with no consumer. What the file does need beyond the union is the indole
        // exclusion below, which narrows it.
        let mut rings: Vec<Ring> = Vec::with_capacity(rings_raw.len());
        for atoms in rings_raw {
            let is_aromatic = ring_is_aromatic(molecule, &neighbors, &hybridization, &atoms);
            rings.push(Ring {
                atoms,
                aromatic: is_aromatic,
            });
        }
        rings.sort_by_key(|r| r.size());

        // The indole rule, from the definition file's own closing note: "For AM1-BCC, five-
        // memberred ring connected to a six memberred aromatic ring (like indole) is not
        // considerred aromatic." Sharing an *edge* is what "connected" means here — two atoms in
        // common — so a biphenyl-like single bond between rings does not trigger it.
        let demote: Vec<usize> = (0..rings.len())
            .filter(|&k| {
                rings[k].aromatic
                    && rings[k].size() == 5
                    && rings.iter().enumerate().any(|(m, other)| {
                        m != k
                            && other.aromatic
                            && other.size() == 6
                            && rings[k]
                                .atoms
                                .iter()
                                .filter(|a| other.atoms.contains(a))
                                .count()
                                >= 2
                    })
            })
            .collect();
        for k in demote {
            rings[k].aromatic = false;
        }

        // An atom is aromatic if it survives in some aromatic ring, so the demotion above really
        // removes it rather than leaving a stale flag from the ring it was set by.
        let mut aromatic = vec![false; nat];
        for ring in rings.iter().filter(|r| r.aromatic) {
            for &a in &ring.atoms {
                aromatic[a] = true;
            }
        }

        // A bond between two aromatic atoms that share an aromatic ring is aromatic. Sharing a
        // ring matters: in biphenyl the bond joining the two rings connects two aromatic atoms
        // but is itself an ordinary single bond.
        for b in &mut bonds {
            if aromatic[b.i]
                && aromatic[b.j]
                && rings
                    .iter()
                    .any(|r| r.aromatic && r.atoms.contains(&b.i) && r.atoms.contains(&b.j))
            {
                b.order = BondOrder::Aromatic;
            }
        }

        // Delocalized groups: carboxylate, nitro, phosphate, sulfonate, N-oxide.
        mark_delocalized(molecule, &neighbors, &mut bonds);

        // A Kekulé structure for the aromatic bonds, so `sb` and `db` are defined on them.
        let (kekule_double, kekule_unique) = kekule_double_bonds(
            molecule,
            &neighbors,
            &hybridization,
            &rings,
            &bonds,
            &mut warnings,
        );

        // Report what could not be typed, rather than letting it fall through silently.
        for (i, atom) in molecule.atoms.iter().enumerate() {
            if !matches!(atom.z, 1 | 6 | 7 | 8 | 9 | 14 | 15 | 16 | 17 | 35 | 53) {
                warnings.push(format!(
                    "atom {i} ({}) has no AM1-BCC atom type; its charge will be the raw AM1 \
                     Mulliken value and every bond to it is left uncorrected",
                    z_to_symbol(atom.z).unwrap_or("?")
                ));
            }
        }

        Self {
            bonds,
            neighbors,
            hybridization,
            aromatic,
            in_ring,
            rings,
            smallest_ring,
            kekule_double,
            kekule_unique,
            warnings,
        }
    }

    pub fn coordination(&self, i: usize) -> usize {
        self.neighbors[i].len()
    }

    /// The formal bond kinds of bond `k`, as `ATOMTYPE_BCC.DEF` counts them: `(sb, db, tb)`.
    ///
    /// These are the file's **lowercase** kinds, which are the inclusive ones: `sb` is "single
    /// bond, including aromatic single, delocalized bond (9 in AM1-BCC)" and `db` is "double bond,
    /// including aromatic double". An aromatic bond is therefore one *or* the other according to
    /// [`Self::kekule_double`], never both — which is the distinction rules like `[2sb]` versus
    /// `[sb,db]` rest on.
    pub fn bond_kinds(&self, k: usize) -> (bool, bool, bool) {
        match self.bonds[k].order {
            BondOrder::Single | BondOrder::Delocalized => (true, false, false),
            BondOrder::Double => (false, true, false),
            BondOrder::Triple => (false, false, true),
            BondOrder::Aromatic => {
                if self.kekule_double[k] {
                    (false, true, false)
                } else {
                    (true, false, false)
                }
            }
        }
    }

    /// Index of the bond joining `i` and `j`, if they are bonded.
    pub fn bond_between(&self, i: usize, j: usize) -> Option<usize> {
        self.bonds
            .iter()
            .position(|b| (b.i == i && b.j == j) || (b.i == j && b.j == i))
    }

    /// How many bonds of atom `i` are of each formal kind: `(sb, db, tb)`.
    pub fn bond_kind_counts(&self, i: usize) -> (usize, usize, usize) {
        let (mut s, mut d, mut t) = (0, 0, 0);
        for (k, b) in self.bonds.iter().enumerate() {
            if b.i != i && b.j != i {
                continue;
            }
            let (sb, db, tb) = self.bond_kinds(k);
            s += usize::from(sb);
            d += usize::from(db);
            t += usize::from(tb);
        }
        (s, d, t)
    }

    /// The smallest ring containing both atoms of a bond, if any.
    pub fn ring_size_of_bond(&self, i: usize, j: usize) -> Option<usize> {
        self.rings
            .iter()
            .filter(|r| r.atoms.contains(&i) && r.atoms.contains(&j))
            .map(|r| r.size())
            .min()
    }
}

/// The smallest ring through each bond, deduplicated.
///
/// Not the union-find spanning tree the previous version used. That built one arbitrary spanning
/// forest and took the fundamental cycle of each remaining edge, which in a fused system returns
/// whichever large cycle the tree happened to produce — for naphthalene it can hand back the
/// 10-membered perimeter instead of the two 6-rings. Aromaticity is decided by ring *size*, so a
/// 10-ring where a 6-ring belongs is not a cosmetic difference.
///
/// Here, for every bond, the shortest path between its endpoints that does not use the bond
/// itself is found by breadth-first search; that path plus the bond is the smallest ring through
/// it. The union over bonds contains every ring that any bond is the smallest member of, which is
/// what ring perception is for. It is not a formal SSSR basis — it can return more rings than
/// `E − V + 1` — but it never returns a large ring in place of a small one, which is the failure
/// that mattered.
fn smallest_rings(nat: usize, bonds: &[Bond]) -> Vec<Vec<usize>> {
    let mut adjacency = vec![Vec::new(); nat];
    for b in bonds {
        adjacency[b.i].push(b.j);
        adjacency[b.j].push(b.i);
    }

    let mut found: Vec<Vec<usize>> = Vec::new();
    let mut seen: std::collections::HashSet<Vec<usize>> = std::collections::HashSet::new();

    for b in bonds {
        if let Some(path) = shortest_path_avoiding_edge(&adjacency, b.i, b.j) {
            // `path` runs from j back to i; with the bond itself it closes the ring.
            if path.len() < 3 || path.len() > 12 {
                continue;
            }
            let mut key = path.clone();
            key.sort_unstable();
            if seen.insert(key) {
                found.push(path);
            }
        }
    }
    found.sort_by_key(|r| r.len());
    found
}

/// Shortest path from `from` to `to` that does not traverse the direct `from`–`to` edge.
fn shortest_path_avoiding_edge(
    adjacency: &[Vec<usize>],
    from: usize,
    to: usize,
) -> Option<Vec<usize>> {
    use std::collections::VecDeque;
    let n = adjacency.len();
    let mut prev = vec![usize::MAX; n];
    let mut seen = vec![false; n];
    let mut queue = VecDeque::new();
    seen[from] = true;
    queue.push_back(from);

    while let Some(x) = queue.pop_front() {
        for &y in &adjacency[x] {
            // Skip the direct edge, in both directions, but only as the very first step.
            if x == from && y == to {
                continue;
            }
            if y == to {
                let mut path = vec![to, x];
                let mut c = x;
                while c != from {
                    c = prev[c];
                    path.push(c);
                }
                return Some(path);
            }
            if !seen[y] {
                seen[y] = true;
                prev[y] = x;
                queue.push_back(y);
            }
        }
    }
    None
}

/// π electrons an atom contributes to its ring, or `None` if it cannot be part of an aromatic
/// system at all.
fn pi_electrons(
    molecule: &Molecule,
    neighbors: &[Vec<usize>],
    hybridization: &[Hybridization],
    atom: usize,
    ring: &[usize],
) -> Option<usize> {
    let z = molecule.atoms[atom].z;
    let coordination = neighbors[atom].len();
    let in_ring_neighbors = neighbors[atom].iter().filter(|n| ring.contains(n)).count();
    if in_ring_neighbors != 2 {
        return None; // not actually a ring member of this ring
    }

    match z {
        6 => {
            if hybridization[atom] != Hybridization::Sp2 {
                return None;
            }
            // An exocyclic double bond to O or S drains the π system rather than joining it
            // (a cyclohexadienone carbonyl carbon is not aromatic).
            let exocyclic_carbonyl = neighbors[atom].iter().any(|&n| {
                !ring.contains(&n)
                    && matches!(molecule.atoms[n].z, 8 | 16)
                    && neighbors[n].len() == 1
            });
            if exocyclic_carbonyl {
                Some(0)
            } else {
                Some(1)
            }
        }
        7 => match coordination {
            // Pyridine-type: two ring bonds only, one π electron in the ring.
            2 => Some(1),
            // Pyrrole-type: the lone pair joins the ring.
            3 => Some(2),
            _ => None,
        },
        8 | 16 => {
            // Furan / thiophene: a divalent O or S donates its lone pair. Only in a 5-ring;
            // a divalent oxygen in a 6-ring (pyran) is not aromatic.
            if coordination == 2 && ring.len() == 5 {
                Some(2)
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Whether a ring is aromatic: the right size, planar, and 4n+2 π electrons.
///
/// All three are needed. Size alone admits cycloheptatriene and macrocyclic lactones, which the
/// previous version typed as aromatic because it never looked at ring size at all. Planarity
/// alone admits cyclobutadiene. The Hückel count is what distinguishes benzene from
/// cyclooctatetraene.
fn ring_is_aromatic(
    molecule: &Molecule,
    neighbors: &[Vec<usize>],
    hybridization: &[Hybridization],
    ring: &[usize],
) -> bool {
    if !matches!(ring.len(), 5 | 6) {
        return false;
    }
    let mut total = 0usize;
    for &a in ring {
        match pi_electrons(molecule, neighbors, hybridization, a, ring) {
            Some(n) => total += n,
            None => return false,
        }
    }
    // 4n + 2
    if total < 2 || (total - 2) % 4 != 0 {
        return false;
    }
    ring_is_planar(molecule, ring)
}

/// All ring atoms within 0.15 Å of a best-fit plane.
fn ring_is_planar(molecule: &Molecule, ring: &[usize]) -> bool {
    use crate::math::Vec3;
    let points: Vec<Vec3> = ring
        .iter()
        .map(|&a| molecule.atoms[a].position * BOHR_TO_ANGSTROM)
        .collect();
    let n = points.len() as f64;
    let centroid = points.iter().fold(Vec3::zero(), |acc, p| acc + *p) * (1.0 / n);

    // Plane normal from the two largest cross products of centroid-relative vectors: cheaper
    // than an eigen-decomposition and adequate for deciding a 0.15 Å question.
    let mut normal = Vec3::zero();
    for k in 0..points.len() {
        let a = points[k] - centroid;
        let b = points[(k + 1) % points.len()] - centroid;
        normal += a.cross(b);
    }
    if normal.norm() < 1.0e-8 {
        return false;
    }
    let normal = normal.normalized();
    points
        .iter()
        .all(|p| (*p - centroid).dot(normal).abs() < 0.15)
}

/// Mark the bonds of symmetric delocalized groups.
///
/// The rule is a chemical one, not a length threshold: a central atom carrying **two or more
/// equivalent terminal O/S atoms** has delocalized bonds to all of them. That is exactly
/// carboxylate, nitro, phosphate, sulfate, sulfonate and sulfone, and it correctly excludes a
/// carboxylic acid (one terminal `=O` and one two-coordinate `–OH`, which are a genuine double
/// and a genuine single bond) and a sulfoxide (one terminal O).
///
/// A four-coordinate nitrogen with a single terminal oxygen is the N-oxide case and is included
/// separately: there is only one such bond, but it is delocalized rather than a double bond.
fn mark_delocalized(molecule: &Molecule, neighbors: &[Vec<usize>], bonds: &mut [Bond]) {
    let nat = molecule.atoms.len();
    let mut delocalized_centre = vec![false; nat];

    for centre in 0..nat {
        let z = molecule.atoms[centre].z;
        if !matches!(z, 6 | 7 | 15 | 16) {
            continue;
        }
        let terminal_chalcogens = neighbors[centre]
            .iter()
            .filter(|&&n| matches!(molecule.atoms[n].z, 8 | 16) && neighbors[n].len() == 1)
            .count();
        let is_n_oxide = z == 7 && neighbors[centre].len() == 4 && terminal_chalcogens == 1;
        if terminal_chalcogens >= 2 || is_n_oxide {
            delocalized_centre[centre] = true;
        }
    }

    for b in bonds {
        let (centre, terminal) = if delocalized_centre[b.i] {
            (b.i, b.j)
        } else {
            (b.j, b.i)
        };
        if !delocalized_centre[centre] {
            continue;
        }
        if matches!(molecule.atoms[terminal].z, 8 | 16) && neighbors[terminal].len() == 1 {
            b.order = BondOrder::Delocalized;
        }
    }
}

/// A Kekulé structure for the aromatic bonds: which of them are formally double.
///
/// # Why a matching rather than a rule of thumb
///
/// `ATOMTYPE_BCC.DEF` asks whether a two-coordinate nitrogen has `[2sb]` (two formal single bonds)
/// or `[sb,db]` (one of each), and those two rules give different types — 21 against 24. An
/// aromatic ring bond is formally one or the other, and which one is not a local property: it is
/// fixed by the whole ring alternating. Pyridine's nitrogen must come out `[sb,db]`, and the
/// parameter file's `17–24` aromatic entry is what says so.
///
/// # The matching
///
/// Each aromatic atom that contributes **one** π electron to its ring needs exactly one double
/// bond; one that contributes two (pyrrole's N, furan's O, thiophene's S) or none (an exocyclic
/// carbonyl carbon) needs none. So this is a perfect matching on the subgraph of aromatic bonds,
/// restricted to the atoms that need pairing. Aromatic systems are small — a fused pair is a dozen
/// atoms — so plain backtracking finds it without needing Blossom.
///
/// A system with no such matching is not a Kekulé structure at all. That is reported rather than
/// papered over, and every aromatic bond in it is left formally single, which is the choice that
/// changes the fewest types.
///
/// Returns the assignment and whether it is **unique**. Non-uniqueness is not a failure — benzene's
/// two Kekulé structures are equivalent and the search picks one arbitrarily — but it means the
/// single/double label on any particular aromatic bond is an artifact, which is what AM1-BCC's
/// bond type 10 ("aromatic", no resolved order) is for.
fn kekule_double_bonds(
    molecule: &Molecule,
    neighbors: &[Vec<usize>],
    hybridization: &[Hybridization],
    rings: &[Ring],
    bonds: &[Bond],
    warnings: &mut Vec<String>,
) -> (Vec<bool>, bool) {
    let mut double = vec![false; bonds.len()];
    for (k, b) in bonds.iter().enumerate() {
        if b.order == BondOrder::Double {
            double[k] = true;
        }
    }

    // Which aromatic atoms need a partner, from the π count that made their ring aromatic.
    let nat = molecule.atoms.len();
    let mut needs = vec![false; nat];
    let mut any_aromatic = false;
    for ring in rings.iter().filter(|r| r.aromatic) {
        for &a in &ring.atoms {
            any_aromatic = true;
            if pi_electrons(molecule, neighbors, hybridization, a, &ring.atoms) == Some(1) {
                needs[a] = true;
            }
        }
    }
    if !any_aromatic {
        return (double, true);
    }

    // Aromatic bonds incident on each atom, by bond index.
    let mut incident: Vec<Vec<usize>> = vec![Vec::new(); nat];
    for (k, b) in bonds.iter().enumerate() {
        if b.order == BondOrder::Aromatic {
            incident[b.i].push(k);
            incident[b.j].push(k);
        }
    }

    let pending: Vec<usize> = (0..nat).filter(|&a| needs[a]).collect();
    let mut matched = vec![false; nat];
    let mut chosen = vec![false; bonds.len()];
    if !assign_kekule(
        &pending,
        0,
        bonds,
        &incident,
        &mut matched,
        &mut chosen,
        &[],
    ) {
        warnings.push(format!(
            "no Kekule structure was found for the aromatic system spanning {} atoms; its bonds \
             are treated as formally single, which can change the atom type of a two-coordinate \
             ring nitrogen",
            pending.len()
        ));
        return (double, false);
    }
    for (k, c) in chosen.iter().enumerate() {
        if *c {
            double[k] = true;
        }
    }

    // Is it the only one? Forbid each chosen bond in turn and look for another perfect matching.
    // If one exists, the structure found above is an artifact of the search order — benzene, where
    // both Kekulé structures are equivalent and no bond is "the" double bond.
    //
    // Bounded work: one matching attempt per aromatic bond that was chosen, and an aromatic system
    // has a handful.
    let unique = !chosen.iter().enumerate().any(|(k, &c)| {
        if !c {
            return false;
        }
        let mut m = vec![false; nat];
        let mut c2 = vec![false; bonds.len()];
        assign_kekule(&pending, 0, bonds, &incident, &mut m, &mut c2, &[k])
    });
    (double, unique)
}

/// Backtracking perfect matching over `pending`, used by [`kekule_double_bonds`].
///
/// `forbidden` lists bond indices the matching may not use, which is how uniqueness is tested:
/// forbid one bond of a known matching and ask whether another exists.
fn assign_kekule(
    pending: &[usize],
    at: usize,
    bonds: &[Bond],
    incident: &[Vec<usize>],
    matched: &mut [bool],
    chosen: &mut [bool],
    forbidden: &[usize],
) -> bool {
    let Some(&atom) = pending.get(at) else {
        return true;
    };
    if matched[atom] {
        return assign_kekule(pending, at + 1, bonds, incident, matched, chosen, forbidden);
    }
    for &k in &incident[atom] {
        if forbidden.contains(&k) {
            continue;
        }
        let other = if bonds[k].i == atom {
            bonds[k].j
        } else {
            bonds[k].i
        };
        // The partner must also be an atom that wants a double bond, and be free.
        if matched[other] || !pending.contains(&other) {
            continue;
        }
        matched[atom] = true;
        matched[other] = true;
        chosen[k] = true;
        if assign_kekule(pending, at + 1, bonds, incident, matched, chosen, forbidden) {
            return true;
        }
        matched[atom] = false;
        matched[other] = false;
        chosen[k] = false;
    }
    false
}

/// Reference bond lengths (Å) per element pair, for the length-based order guess.
///
/// Extended well beyond the six C/N/O pairs the previous version knew about: without an entry
/// the guess falls back to `Single`, which typed every C=S, P=O and S=O as a single bond and so
/// mistyped every thiocarbonyl, phosphate and sulfonyl group in the molecule.
fn reference_lengths(zi: u8, zj: u8) -> Option<(f64, f64, f64)> {
    // (single, double, triple). A triple length equal to the double one disables the triple
    // branch for pairs that do not form one.
    let key = (zi.min(zj), zi.max(zj));
    Some(match key {
        (6, 6) => (1.54, 1.34, 1.20),
        (6, 7) => (1.47, 1.28, 1.16),
        (6, 8) => (1.43, 1.21, 1.13),
        (7, 7) => (1.45, 1.25, 1.10),
        (7, 8) => (1.40, 1.21, 1.06),
        (8, 8) => (1.48, 1.21, 1.21),
        (6, 15) => (1.84, 1.66, 1.54),
        (6, 16) => (1.82, 1.60, 1.53),
        (7, 15) => (1.77, 1.58, 1.48),
        (7, 16) => (1.71, 1.54, 1.44),
        (8, 15) => (1.63, 1.48, 1.48),
        (8, 16) => (1.57, 1.44, 1.44),
        (15, 15) => (2.21, 2.02, 1.89),
        (15, 16) => (2.10, 1.92, 1.92),
        (16, 16) => (2.05, 1.88, 1.88),
        // Silicon and the halogens form single bonds in everything this is used for.
        _ => return None,
    })
}

fn guess_bond_order(zi: u8, zj: u8, d: f64) -> BondOrder {
    let Some((single, double, triple)) = reference_lengths(zi, zj) else {
        return BondOrder::Single;
    };
    let ds = (d - single).abs();
    let dd = (d - double).abs();
    let dt = (d - triple).abs();
    if triple < double && dt < dd && dt < ds {
        BondOrder::Triple
    } else if dd < ds {
        BondOrder::Double
    } else {
        BondOrder::Single
    }
}

fn perceive_hybridization(
    molecule: &Molecule,
    neighbors: &[Vec<usize>],
    i: usize,
) -> Hybridization {
    let z = molecule.atoms[i].z;
    let cn = neighbors[i].len();
    match z {
        1 => Hybridization::None, // H
        6 => match cn {
            4 => Hybridization::Sp3,
            3 => Hybridization::Sp2,
            2 => Hybridization::Sp,
            _ => Hybridization::Sp3,
        },
        7 => match cn {
            4 | 3 => {
                // sp2 if planar/aromatic-like: use average angle proxy.
                if is_planar(molecule, neighbors, i) {
                    Hybridization::Sp2
                } else {
                    Hybridization::Sp3
                }
            }
            2 => Hybridization::Sp2,
            1 => Hybridization::Sp,
            _ => Hybridization::Sp3,
        },
        8 => match cn {
            2 => Hybridization::Sp3,
            1 => Hybridization::Sp2,
            _ => Hybridization::Sp3,
        },
        // Divalent sulfur in a ring is a thiophene-type π donor, which the previous version
        // could never express: it returned Sp3 unconditionally, so the `16` in its aromaticity
        // test was unreachable and thiophene's sulfur was never aromatic.
        16 => match cn {
            1 => Hybridization::Sp2,
            2 => {
                if is_planar(molecule, neighbors, i) {
                    Hybridization::Sp2
                } else {
                    Hybridization::Sp3
                }
            }
            _ => Hybridization::Sp3,
        },
        _ => Hybridization::Sp3,
    }
}

fn is_planar(molecule: &Molecule, neighbors: &[Vec<usize>], i: usize) -> bool {
    let nb = &neighbors[i];
    if nb.len() < 3 {
        return false;
    }
    // Sum of the three largest angles ≈ 360° → planar (sp2).
    let p0 = molecule.atoms[i].position;
    let mut angles = Vec::new();
    for a in 0..nb.len() {
        for b in (a + 1)..nb.len() {
            let va = (molecule.atoms[nb[a]].position - p0).normalized();
            let vb = (molecule.atoms[nb[b]].position - p0).normalized();
            angles.push(va.dot(vb).clamp(-1.0, 1.0).acos());
        }
    }
    // `total_cmp`, not `partial_cmp(..).unwrap()`: a NaN angle would panic there, and a
    // comparator that panics is the worst failure mode available — it aborts a calculation from
    // inside a sort. `acos` of a clamped dot cannot be NaN today, which is exactly why the
    // `unwrap` survived; this makes the guarantee the code's rather than the caller's.
    angles.sort_by(|a, b| b.total_cmp(a));
    let sum: f64 = angles.iter().take(3).sum();
    sum > 350.0_f64.to_radians()
}

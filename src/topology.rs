// SPDX-License-Identifier: GPL-3.0-or-later

//! Molecular-graph perception from geometry: bonds (covalent-radius overlap), per-atom
//! coordination and hybridization, simple ring/aromaticity detection, and a heuristic
//! bond-order guess. This feeds the AM1-BCC atom/bond typing in [`crate::bcc`].
//!
//! The perception here is deliberately lightweight (geometry + valence heuristics), not a
//! full cheminformatics toolkit; it covers common organic molecules.

use crate::constants::{covalent_radius_angstrom, BOHR_TO_ANGSTROM};
use crate::system::Molecule;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BondOrder {
    Single,
    Double,
    Triple,
    Aromatic,
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

#[derive(Clone, Debug)]
pub struct Topology {
    pub bonds: Vec<Bond>,
    /// Neighbour atom indices per atom.
    pub neighbors: Vec<Vec<usize>>,
    pub hybridization: Vec<Hybridization>,
    pub aromatic: Vec<bool>,
    pub in_ring: Vec<bool>,
}

impl Topology {
    pub fn perceive(molecule: &Molecule) -> Self {
        let nat = molecule.atoms.len();
        let mut neighbors = vec![Vec::new(); nat];
        let mut bonds = Vec::new();

        for i in 0..nat {
            for j in (i + 1)..nat {
                let d = (molecule.atoms[j].position - molecule.atoms[i].position).norm()
                    * BOHR_TO_ANGSTROM;
                let ri = covalent_radius_angstrom(molecule.atoms[i].z);
                let rj = covalent_radius_angstrom(molecule.atoms[j].z);
                if d < 1.3 * (ri + rj) && d > 0.4 {
                    neighbors[i].push(j);
                    neighbors[j].push(i);
                    let order = guess_bond_order(molecule.atoms[i].z, molecule.atoms[j].z, d);
                    bonds.push(Bond {
                        i,
                        j,
                        order,
                        length: d,
                    });
                }
            }
        }

        // Ring membership by DFS back-edge detection.
        let in_ring = detect_rings(nat, &neighbors, &bonds);

        // Hybridization from coordination + geometry.
        let mut hybridization = vec![Hybridization::None; nat];
        for i in 0..nat {
            hybridization[i] = perceive_hybridization(molecule, &neighbors, i);
        }

        // Aromaticity: ring atoms that are sp2 C/N/O in a 5- or 6-membered ring.
        let mut aromatic = vec![false; nat];
        for i in 0..nat {
            if in_ring[i]
                && hybridization[i] == Hybridization::Sp2
                && matches!(molecule.atoms[i].z, 6 | 7 | 8 | 16)
            {
                aromatic[i] = true;
            }
        }
        for b in &mut bonds {
            if aromatic[b.i] && aromatic[b.j] && in_ring[b.i] && in_ring[b.j] {
                b.order = BondOrder::Aromatic;
            }
        }

        Self {
            bonds,
            neighbors,
            hybridization,
            aromatic,
            in_ring,
        }
    }

    pub fn coordination(&self, i: usize) -> usize {
        self.neighbors[i].len()
    }
}

fn guess_bond_order(zi: u8, zj: u8, d: f64) -> BondOrder {
    // Heuristic reference bond lengths (Å) for common pairs; fall back to single.
    let key = (zi.min(zj), zi.max(zj));
    let (single, double, triple) = match key {
        (6, 6) => (1.54, 1.34, 1.20),
        (6, 7) => (1.47, 1.28, 1.16),
        (6, 8) => (1.43, 1.21, 1.13),
        (7, 7) => (1.45, 1.25, 1.10),
        (7, 8) => (1.40, 1.21, 1.06),
        (8, 8) => (1.48, 1.21, 1.10),
        _ => return BondOrder::Single,
    };
    let ds = (d - single).abs();
    let dd = (d - double).abs();
    let dt = (d - triple).abs();
    if dt < dd && dt < ds {
        BondOrder::Triple
    } else if dd < ds {
        BondOrder::Double
    } else {
        BondOrder::Single
    }
}

fn perceive_hybridization(molecule: &Molecule, neighbors: &[Vec<usize>], i: usize) -> Hybridization {
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
        16 => Hybridization::Sp3,
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
    angles.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let sum: f64 = angles.iter().take(3).sum();
    sum > 350.0_f64.to_radians()
}

fn detect_rings(nat: usize, neighbors: &[Vec<usize>], bonds: &[Bond]) -> Vec<bool> {
    // An edge is a ring edge iff its removal keeps both endpoints connected.
    // Simpler: an atom is in a ring iff it lies on a cycle. Use union-find on a spanning
    // forest; any non-tree edge closes a cycle whose atoms are all "in ring".
    let mut parent: Vec<usize> = (0..nat).collect();
    fn find(parent: &mut [usize], x: usize) -> usize {
        let mut r = x;
        while parent[r] != r {
            r = parent[r];
        }
        let mut c = x;
        while parent[c] != c {
            let next = parent[c];
            parent[c] = r;
            c = next;
        }
        r
    }
    let mut in_ring = vec![false; nat];
    // Build spanning forest; collect back-edges.
    let mut back_edges = Vec::new();
    for b in bonds {
        let (ri, rj) = (find(&mut parent, b.i), find(&mut parent, b.j));
        if ri == rj {
            back_edges.push((b.i, b.j));
        } else {
            parent[ri] = rj;
        }
    }
    // For each back edge, mark the cycle via BFS in the spanning tree between endpoints.
    let tree: Vec<Vec<usize>> = {
        let mut t = vec![Vec::new(); nat];
        let mut par: Vec<usize> = (0..nat).collect();
        for b in bonds {
            let (ri, rj) = (find(&mut par, b.i), find(&mut par, b.j));
            if ri != rj {
                par[ri] = rj;
                t[b.i].push(b.j);
                t[b.j].push(b.i);
            }
        }
        t
    };
    for (u, v) in back_edges {
        if let Some(path) = bfs_path(&tree, u, v) {
            for a in path {
                in_ring[a] = true;
            }
        } else {
            in_ring[u] = true;
            in_ring[v] = true;
        }
    }
    let _ = neighbors;
    in_ring
}

fn bfs_path(tree: &[Vec<usize>], start: usize, goal: usize) -> Option<Vec<usize>> {
    use std::collections::VecDeque;
    let mut prev = vec![usize::MAX; tree.len()];
    let mut seen = vec![false; tree.len()];
    let mut q = VecDeque::new();
    q.push_back(start);
    seen[start] = true;
    while let Some(x) = q.pop_front() {
        if x == goal {
            let mut path = vec![goal];
            let mut c = goal;
            while c != start {
                c = prev[c];
                path.push(c);
            }
            return Some(path);
        }
        for &n in &tree[x] {
            if !seen[n] {
                seen[n] = true;
                prev[n] = x;
                q.push_back(n);
            }
        }
    }
    None
}

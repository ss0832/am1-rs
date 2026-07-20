// SPDX-License-Identifier: GPL-3.0-or-later

//! AM1-BCC partial charges for AMBER (Jakalian, Bush, Jack & Bayly, *J. Comput. Chem.*
//! **21**, 132 (2000); **23**, 1623 (2002)).
//!
//! Pipeline: run the AM1 SCF → AM1 Mulliken charges (exact) → perceive the molecular graph →
//! assign antechamber **BCC atom types** (the numeric 11–91 scheme of `ATOMTYPE_BCC.DEF`) and
//! bond types → apply the additive **bond charge corrections** from antechamber's
//! `BCCPARM.DAT` → AMBER-ready per-atom charges, optionally written as mol2.
//!
//! **Provenance.** The 405 bond-charge-correction parameters in `data/bccparm.dat` are the
//! exact antechamber `BCCPARM.DAT` file from AmberTools (GPL-3); the atom-type scheme follows
//! `ATOMTYPE_BCC.DEF` (retained under `third_party/antechamber/`). See `THIRD_PARTY_NOTICES.md`.
//!
//! **Parity note.** The BCC *parameters* are exact. The atom/bond typing here reimplements the
//! common `ATOMTYPE_BCC.DEF` rules from geometry-perceived topology; it is faithful for typical
//! organic molecules but is **not** guaranteed byte-identical to antechamber's full definition-
//! file matching engine and penalty-based bond-order perception for every edge case. Full
//! byte-exact parity is a documented larger effort.

use crate::error::Result;
use crate::params::Am1Parameters;
use crate::scf::{run_am1, Am1Options};
use crate::system::{z_to_symbol, Molecule};
use crate::topology::{BondOrder, Topology};
use std::collections::HashMap;

/// Embedded antechamber bond-charge-correction parameters (AmberTools `BCCPARM.DAT`, GPL-3).
const BCCPARM: &str = include_str!("data/bccparm.dat");

#[derive(Clone, Debug)]
pub struct BccResult {
    pub charges: Vec<f64>,
    /// antechamber BCC atom-type code per atom (11–91).
    pub atom_types: Vec<String>,
    /// AM1 Mulliken charges before the bond charge corrections.
    pub mulliken: Vec<f64>,
}

/// Compute AM1-BCC charges for a molecule.
pub fn am1_bcc_charges(
    molecule: &Molecule,
    params: &Am1Parameters,
    scf_opts: &Am1Options,
) -> Result<BccResult> {
    let scf = run_am1(molecule, params, scf_opts)?;
    let mulliken = scf.charges.clone();
    let topo = Topology::perceive(molecule);
    let types = assign_bcc_types(molecule, &topo);
    let table = bcc_table();

    let mut charges = mulliken.clone();
    for bond in &topo.bonds {
        let (ta, tb) = (types[bond.i], types[bond.j]);
        let bt = antechamber_bond_code(bond.order);
        // Directional lookup: parameter δ for (ta,tb,bt) shifts +δ onto atom i, −δ onto j;
        // if only the reverse (tb,ta,bt) is tabulated, the sign flips.
        if let Some(&delta) = table.get(&(ta, tb, bt)) {
            charges[bond.i] += delta;
            charges[bond.j] -= delta;
        } else if let Some(&delta) = table.get(&(tb, ta, bt)) {
            charges[bond.i] -= delta;
            charges[bond.j] += delta;
        }
    }

    Ok(BccResult {
        charges,
        atom_types: types.iter().map(|t| t.to_string()).collect(),
        mulliken,
    })
}

/// Parse the embedded `BCCPARM.DAT` into `(type_i, type_j, bond_type) -> correction`.
pub fn bcc_table() -> HashMap<(u32, u32, u32), f64> {
    let mut m = HashMap::new();
    for line in BCCPARM.lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        // Format: index  type_i  type_j  bond_type  correction
        if f.len() < 5 {
            continue;
        }
        if let (Ok(ti), Ok(tj), Ok(bt), Ok(d)) = (
            f[1].parse::<u32>(),
            f[2].parse::<u32>(),
            f[3].parse::<u32>(),
            f[4].parse::<f64>(),
        ) {
            m.insert((ti, tj, bt), d);
        }
    }
    m
}

/// Map a perceived bond order to the antechamber bond-type code used in `BCCPARM.DAT`.
fn antechamber_bond_code(order: BondOrder) -> u32 {
    match order {
        BondOrder::Single => 1,
        BondOrder::Double => 2,
        BondOrder::Triple => 3,
        BondOrder::Aromatic => 7, // primary aromatic bond type
    }
}

/// Assign the antechamber BCC atom-type code (11–91) per atom, following the common
/// `ATOMTYPE_BCC.DEF` rules.
pub fn assign_bcc_types(molecule: &Molecule, topo: &Topology) -> Vec<u32> {
    let z_of = |i: usize| molecule.atoms[i].z;
    let cn = |i: usize| topo.neighbors[i].len();
    // Whether atom `i` has a double bond to an atom of element `zt`.
    let has_double_to = |i: usize, zt: u8| {
        topo.bonds.iter().any(|b| {
            b.order == BondOrder::Double
                && ((b.i == i && z_of(b.j) == zt) || (b.j == i && z_of(b.i) == zt))
        })
    };
    // Terminal (1-connected) atom of element `zt` bonded to `i` (e.g. a carbonyl =O).
    let has_terminal = |i: usize, zt: u8| {
        topo.neighbors[i]
            .iter()
            .any(|&n| z_of(n) == zt && cn(n) == 1)
    };

    (0..molecule.atoms.len())
        .map(|i| {
            let z = z_of(i);
            let c = cn(i);
            let arom = topo.aromatic[i];
            match z {
                1 => 91, // H
                6 => {
                    // Carbon
                    if arom {
                        // 17 if an aromatic O/N neighbour in the ring, else 16.
                        if topo.neighbors[i]
                            .iter()
                            .any(|&n| topo.aromatic[n] && matches!(z_of(n), 7 | 8))
                        {
                            17
                        } else {
                            16
                        }
                    } else if c >= 4 {
                        11
                    } else if c == 3 {
                        // 14: carbonyl-like (terminal O/S); 13: sp2 with =N/=P; else 12.
                        if has_terminal(i, 8) || has_terminal(i, 16) {
                            14
                        } else if has_double_to(i, 7) || has_double_to(i, 15) {
                            13
                        } else {
                            12
                        }
                    } else {
                        15 // sp
                    }
                }
                7 => {
                    // Nitrogen
                    if arom {
                        23
                    } else if c >= 4 {
                        21
                    } else if c == 3 {
                        // 22: amide N (neighbour C bearing a terminal O/S).
                        let amide = topo.neighbors[i].iter().any(|&n| {
                            z_of(n) == 6 && (has_terminal(n, 8) || has_terminal(n, 16))
                        });
                        if amide {
                            22
                        } else {
                            21
                        }
                    } else if c == 2 {
                        24
                    } else {
                        25
                    }
                }
                8 => {
                    // Oxygen
                    if c == 1 {
                        // Look at the attached carbon's environment.
                        if let Some(&nbc) = topo.neighbors[i].first() {
                            if z_of(nbc) == 6 {
                                let n_o = topo.neighbors[nbc]
                                    .iter()
                                    .filter(|&&x| z_of(x) == 8)
                                    .count();
                                let has_n3 = topo.neighbors[nbc]
                                    .iter()
                                    .any(|&x| z_of(x) == 7 && cn(x) >= 3);
                                if n_o >= 2 || has_n3 {
                                    33
                                } else {
                                    32
                                }
                            } else {
                                31
                            }
                        } else {
                            31
                        }
                    } else {
                        31
                    }
                }
                16 => match c {
                    3 => 52,
                    4 => 53,
                    _ => 51,
                },
                15 => {
                    if c >= 4 {
                        42
                    } else {
                        41
                    }
                }
                14 => 61, // Si
                9 => 71,
                17 => 72,
                35 => 73,
                53 => 74,
                _ => 0,
            }
        })
        .collect()
}

/// Write a minimal Tripos MOL2 file with the AM1-BCC charges.
pub fn write_mol2(path: &str, molecule: &Molecule, bcc: &BccResult) -> Result<()> {
    use crate::constants::BOHR_TO_ANGSTROM;
    let topo = Topology::perceive(molecule);
    let mut s = String::new();
    s.push_str("@<TRIPOS>MOLECULE\nam1_bcc\n");
    s.push_str(&format!(" {} {} 0 0 0\n", molecule.atoms.len(), topo.bonds.len()));
    s.push_str("SMALL\nuser_charges\n\n@<TRIPOS>ATOM\n");
    for (i, atom) in molecule.atoms.iter().enumerate() {
        let p = atom.position * BOHR_TO_ANGSTROM;
        let sym = z_to_symbol(atom.z).unwrap_or("X");
        s.push_str(&format!(
            "{:>7} {:<4} {:>10.4} {:>10.4} {:>10.4} {:<6} 1 UNL {:>10.5}\n",
            i + 1,
            format!("{sym}{}", i + 1),
            p.x,
            p.y,
            p.z,
            sym,
            bcc.charges[i],
        ));
    }
    s.push_str("@<TRIPOS>BOND\n");
    for (k, b) in topo.bonds.iter().enumerate() {
        let code = match b.order {
            BondOrder::Single => "1",
            BondOrder::Double => "2",
            BondOrder::Triple => "3",
            BondOrder::Aromatic => "ar",
        };
        s.push_str(&format!("{:>6} {:>5} {:>5} {}\n", k + 1, b.i + 1, b.j + 1, code));
    }
    std::fs::write(path, s)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bccparm_parses() {
        let t = bcc_table();
        // The embedded antechamber BCCPARM.DAT has 405 entries.
        assert!(t.len() > 350, "only {} BCC params parsed", t.len());
        // Spot-check a known entry: sp3-C to sp2-C single bond = +0.0042.
        assert!((t.get(&(11, 12, 1)).copied().unwrap_or(0.0) - 0.0042).abs() < 1e-9);
    }

    #[test]
    fn water_bcc_conserves_charge_and_types() {
        let xyz = "3\nwater\nO 0.0 0.0 0.0\nH 0.9584 0.0 0.0\nH -0.24 0.9278 0.0\n";
        let mol = Molecule::from_xyz_str(xyz, 0.0).unwrap();
        let params = Am1Parameters::standard().unwrap();
        let bcc = am1_bcc_charges(&mol, &params, &Am1Options::default()).unwrap();
        let sum: f64 = bcc.charges.iter().sum();
        assert!(sum.abs() < 1e-9, "charge not conserved: {sum}");
        assert_eq!(bcc.atom_types[0], "31"); // O, 2-connected -> hydroxyl/water O
        assert_eq!(bcc.atom_types[1], "91"); // H
    }

    #[test]
    fn methanol_types() {
        // CH3-OH: C is 11 (sp3), O is 31, methyl H are 91, hydroxyl H is 91.
        let xyz = "6\nmethanol\nC -0.36 0.0 0.0\nO 1.06 0.0 0.0\nH -0.74 1.02 0.0\nH -0.74 -0.51 0.88\nH -0.74 -0.51 -0.88\nH 1.36 0.90 0.0\n";
        let mol = Molecule::from_xyz_str(xyz, 0.0).unwrap();
        let params = Am1Parameters::standard().unwrap();
        let bcc = am1_bcc_charges(&mol, &params, &Am1Options::default()).unwrap();
        assert_eq!(bcc.atom_types[0], "11");
        assert_eq!(bcc.atom_types[1], "31");
        let sum: f64 = bcc.charges.iter().sum();
        assert!(sum.abs() < 1e-9);
    }
}

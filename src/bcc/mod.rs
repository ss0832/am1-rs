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
//! # Parameter coverage
//!
//! `BCCPARM.DAT` holds 405 entries across nine bond types, and **all nine are emitted** since
//! 0.2.2:
//!
//! | type | entries | what selects it |
//! |---|---|---|
//! | 1 single | 247 | a single bond |
//! | 2 double | 37 | a double bond |
//! | 3 triple | 3 | a triple bond |
//! | 6 conjugated | 6 | a delocalized bond whose centre is nitrogen — nitro, N-oxide |
//! | 7 aromatic single | 25 | an aromatic bond that is formally single in the Kekulé structure |
//! | 8 aromatic double | 15 | …and one that is formally double |
//! | 9 delocalized | 21 | carboxylate, phosphate, sulfonate |
//! | 10 aromatic | 25 | an aromatic bond with **no resolved** single/double character |
//! | 11 same type | 26 | both ends the same atom type, where no other code is tabulated |
//!
//! Types 7, 8 and 10 are separated by the Kekulé assignment ([`crate::topology::Topology`]): a
//! six-membered aromatic has two equivalent Kekulé structures, so none of its bonds is "the"
//! double bond and they take 10; a five-membered heteroaromatic has one, so 7 and 8 apply.
//!
//! **Reaching 8 and 10 changes no charge** — they are byte-identical to type 7 on every pair they
//! share with it — and every type-11 value is exactly 0.0. Both facts are asserted against the
//! parameter file in `tests/bcc_bond_types.rs` rather than stated here and trusted.
//!
//! Type 11 is not cosmetic, though. Ten of its twenty-six atom types — including **hydrogen and
//! every halogen** — have no single-bond entry at all, so through 0.2.1 an H–H or Cl–Cl bond found
//! no parameter under any emitted code and came back as a *warning* saying the bond was left at raw
//! Mulliken charges. The charges were right (the correction is zero); the warning was not, and
//! [`BccResult::warnings`] is the thing callers are told to check.
//!
//! **Parity note.** The BCC *parameter values* are exact, and since 0.2.2 the atom typing is not a
//! transcription of `ATOMTYPE_BCC.DEF` either — [`atomtype`] **interprets the file**, so the rules
//! and their order are the file's. What remains a reimplementation is the layer underneath: the
//! *perception* of rings, aromaticity and bond orders from geometry, which antechamber derives from
//! its own penalty-based bond-order assignment. Anything the perception could not do confidently is
//! reported in [`BccResult::warnings`] rather than silently guessed.

pub mod atomtype;

pub use atomtype::assign_bcc_types;

use crate::error::Result;
use crate::params::Am1Parameters;
use crate::scf::{run_am1, Am1Options};
use crate::system::{z_to_symbol, Molecule};
use crate::topology::{BondOrder, Topology};
use std::collections::HashMap;

/// Embedded antechamber bond-charge-correction parameters (AmberTools `BCCPARM.DAT`, GPL-3).
const BCCPARM: &str = include_str!("../data/bccparm.dat");

#[derive(Clone, Debug)]
pub struct BccResult {
    pub charges: Vec<f64>,
    /// antechamber BCC atom-type code per atom (11–91). `"0"` means no type was assignable.
    pub atom_types: Vec<String>,
    /// AM1 Mulliken charges before the bond charge corrections.
    pub mulliken: Vec<f64>,
    /// Everything the perception or the parameter lookup could not do confidently.
    ///
    /// Empty for a molecule fully covered by the typing rules and the parameter file. A
    /// non-empty list means some bond was left uncorrected, which shows up as charges that are
    /// partly raw Mulliken — a difference of tenths of an electron, not a rounding. Previously
    /// both cases were skipped in silence.
    pub warnings: Vec<String>,
    /// The perceived bonds as `(i, j, order)`, in the order the corrections were applied.
    ///
    /// Carried on the result because the perception is the expensive, opinionated part and a
    /// caller writing a mol2 needs exactly this. [`write_mol2`] used to re-derive it, which meant
    /// running the whole `O(N²)` pair scan and ring search a second time — and, worse, meant the
    /// file could in principle disagree with the charges it was written next to.
    pub bonds: Vec<(usize, usize, BondOrder)>,
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
    let table = bcc_table_shared();
    let mut warnings = topo.warnings.clone();

    let mut charges = mulliken.clone();
    for (k, bond) in topo.bonds.iter().enumerate() {
        let (ta, tb) = (types[bond.i], types[bond.j]);
        // The bond type, and the fallbacks to try when the more specific one is not tabulated.
        // See `antechamber_bond_codes`.
        let codes = antechamber_bond_codes(molecule, &topo, bond, k, ta, tb);
        let bt = codes[0];
        let found = codes.iter().find_map(|&code| {
            // Directional lookup: parameter δ for (ta,tb,code) shifts +δ onto atom i, −δ onto j;
            // if only the reverse (tb,ta,code) is tabulated, the sign flips.
            table
                .get(&(ta, tb, code))
                .copied()
                .or_else(|| table.get(&(tb, ta, code)).map(|&d| -d))
        });
        if let Some(delta) = found {
            charges[bond.i] += delta;
            charges[bond.j] -= delta;
        } else if ta != 0 && tb != 0 {
            // A typed bond with no parameter. Worth saying: the charges returned are then not
            // AM1-BCC charges for that bond, they are the raw AM1 ones.
            // ASCII only: this reaches the user through both CLIs, whose output has to encode
            // under a cp932 or C locale. See the note in `src/bin/am1_rs.rs`.
            warnings.push(format!(
                "no bond charge correction for atoms {}-{} (types {ta}-{tb}, bond type {bt}); \
                 that bond is left at its raw AM1 Mulliken charges",
                bond.i, bond.j
            ));
        }
    }

    Ok(BccResult {
        charges,
        atom_types: types.iter().map(|t| t.to_string()).collect(),
        mulliken,
        warnings,
        bonds: topo.bonds.iter().map(|b| (b.i, b.j, b.order)).collect(),
    })
}

/// The parsed `BCCPARM.DAT`, built once per process.
///
/// [`bcc_table`] hands back a clone for a caller that wants to own one; nothing inside the crate
/// needs to, and rebuilding a 405-entry hash map on every charge calculation is a fixed per-call
/// cost of the kind that only shows up on small molecules.
fn bcc_table_shared() -> &'static HashMap<(u32, u32, u32), f64> {
    static TABLE: std::sync::OnceLock<HashMap<(u32, u32, u32), f64>> = std::sync::OnceLock::new();
    TABLE.get_or_init(parse_bcc_table)
}

/// Parse the embedded `BCCPARM.DAT` into `(type_i, type_j, bond_type) -> correction`.
pub fn bcc_table() -> HashMap<(u32, u32, u32), f64> {
    bcc_table_shared().clone()
}

fn parse_bcc_table() -> HashMap<(u32, u32, u32), f64> {
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

/// Map a perceived bond to the antechamber bond-type codes used in `BCCPARM.DAT`, most specific
/// first.
///
/// # The codes
///
/// `BCCPARM.DAT` uses nine: **1** single, **2** double, **3** triple, **6** conjugated, **7**
/// aromatic single, **8** aromatic double, **9** delocalized, **10** aromatic, **11** the same type
/// on both ends. The AM1-BCC papers name six of them (single, double, triple, aromatic single,
/// aromatic double, delocalized); 6, 10 and 11 are antechamber's.
///
/// # Why a list rather than one code
///
/// The specific types are not tabulated for every pair. Type 8 has 15 entries against type 7's 25 —
/// the missing ten are heteroatom–heteroatom pairs the fit had no data for — so an aromatic double
/// bond between, say, two type-23 nitrogens has no type-8 parameter and must fall back to type 7 or
/// be left uncorrected. Returning the fallback chain, rather than one code, is what lets the more
/// specific type be emitted wherever it exists without making the less specific one unreachable.
///
/// **This changes no charge**, and that is measured rather than hoped: types 8 and 10 are
/// byte-identical to type 7 on every pair they share with it, and every type-11 entry is exactly
/// 0.0 — `tests/bcc_atom_types.rs` asserts both against the parameter file, and
/// `tests/bcc_bond_types.rs` asserts that emitting them leaves the charges unchanged. They are
/// emitted because the file defines them and the code should reach what it ships, not because
/// reaching them buys a number.
///
/// # The rules
///
/// * **Same type on both ends → 11.** Every one of the file's 26 type-11 entries is an `X–X` pair,
///   and every value is zero — which antisymmetry forces, since the correction puts `+δ` on one
///   end and `−δ` on the other and there is nothing to choose a direction by.
/// * **Aromatic → 7 or 8** by the Kekulé structure ([`Topology::kekule_double`]), which is exactly
///   the single/double distinction those two names make. Where no Kekulé structure exists — a
///   system for which the matching failed, which `Topology` reports — the bond is "aromatic"
///   without a resolved order, which is **10**. That reading of 10 is an interpretation of the
///   name rather than something a source states, and it is recorded here as such; it cannot affect
///   a charge for the reason above.
/// * **Delocalized → 6 or 9**, following how the file is organized: type 6 is tabulated only for
///   nitrogen–chalcogen pairs (all six entries are `2x–31` or `25–51`, i.e. nitro and N-oxide), and
///   type 9 for everything else bonded to a divalent-oxygen or sulfur type (carboxylate, phosphate,
///   sulfonate). So the centre atom being nitrogen is what selects 6.
fn antechamber_bond_codes(
    molecule: &Molecule,
    topo: &Topology,
    bond: &crate::topology::Bond,
    bond_index: usize,
    ta: u32,
    tb: u32,
) -> Vec<u32> {
    let primary = bond_code_for_order(molecule, topo, bond, bond_index);
    let mut chain = match primary {
        // Aromatic double and generic aromatic both fall back to aromatic single, which is the one
        // tabulated for every aromatic pair.
        8 | 10 => vec![primary, 7],
        _ => vec![primary],
    };
    // Same type on both ends: type 11 last, as the file's catch-all for that case.
    //
    // This is not only tidiness. Ten of the twenty-six atom types with a type-11 entry have **no
    // single-bond entry at all** — 25, 32, 33, 53, 61, the four halogens, and 91 — so an H–H bond,
    // or F–F, or Cl–Cl, found no parameter under any code this crate emitted and came back as a
    // warning with the bond left at its raw Mulliken charges. The correction is zero, so the
    // charges were right; the warning was not, and it is the warning a caller is told to check.
    if ta == tb && ta != 0 {
        chain.push(11);
    }
    chain
}

fn bond_code_for_order(
    molecule: &Molecule,
    topo: &Topology,
    bond: &crate::topology::Bond,
    bond_index: usize,
) -> u32 {
    match bond.order {
        BondOrder::Single => 1,
        BondOrder::Double => 2,
        BondOrder::Triple => 3,
        BondOrder::Aromatic => {
            if !topo.kekule_unique {
                10 // aromatic, with no resolved single/double character
            } else if topo.kekule_double[bond_index] {
                8 // aromatic double
            } else {
                7 // aromatic single
            }
        }
        BondOrder::Delocalized => {
            // The chalcogen is the terminal atom; the other end is the group's centre.
            let terminal_is_j =
                matches!(molecule.atoms[bond.j].z, 8 | 16) && topo.neighbors[bond.j].len() == 1;
            let centre = if terminal_is_j { bond.i } else { bond.j };
            if molecule.atoms[centre].z == 7 {
                6
            } else {
                9
            }
        }
    }
}

/// Write a minimal Tripos MOL2 file with the AM1-BCC charges.
pub fn write_mol2(path: &str, molecule: &Molecule, bcc: &BccResult) -> Result<()> {
    use crate::constants::BOHR_TO_ANGSTROM;
    // The bonds the charges were computed against, not a second perception of the same geometry.
    let bonds = &bcc.bonds;
    let mut s = String::new();
    s.push_str("@<TRIPOS>MOLECULE\nam1_bcc\n");
    s.push_str(&format!(
        " {} {} 0 0 0\n",
        molecule.atoms.len(),
        bonds.len()
    ));
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
    for (k, (bi, bj, order)) in bonds.iter().enumerate() {
        let code = match order {
            BondOrder::Single => "1",
            BondOrder::Double => "2",
            BondOrder::Triple => "3",
            BondOrder::Aromatic => "ar",
            // MOL2 has no delocalized order of its own; `am` (amide) is the Tripos code for a
            // partial bond of this kind and is what a carboxylate or nitro group is written as.
            BondOrder::Delocalized => "am",
        };
        s.push_str(&format!(
            "{:>6} {:>5} {:>5} {}\n",
            k + 1,
            bi + 1,
            bj + 1,
            code
        ));
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

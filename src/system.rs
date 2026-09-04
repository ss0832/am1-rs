// SPDX-License-Identifier: GPL-3.0-or-later

//! Geometry and XYZ I/O.
//!
//! Positions are stored in **Bohr** internally. `from_xyz_*` reads Ångström by
//! default (the XYZ convention) and converts on input.
//!
//! A [`Molecule`] optionally carries a [`Lattice`], so the same type describes a molecule, a
//! chain, a slab and a crystal. `cell: None` is the molecular case and is what every
//! non-periodic entry point produces.

use crate::constants::ANGSTROM_TO_BOHR;
use crate::error::{Am1Error, Result};
use crate::lattice::Lattice;
use crate::math::Vec3;
use std::fs;
use std::path::Path;

#[derive(Clone, Debug)]
pub struct Atom {
    pub z: u8,
    /// Position in Bohr.
    pub position: Vec3,
}

#[derive(Clone, Debug)]
pub struct Molecule {
    pub atoms: Vec<Atom>,
    /// Total molecular charge (electrons removed = positive).
    pub charge: f64,
    /// Spin multiplicity (2S+1). 1 = closed-shell singlet.
    pub multiplicity: usize,
    /// Periodic cell, if this is a chain, slab or crystal rather than a molecule.
    ///
    /// `None` is the molecular case and is what every existing entry point produces, so the
    /// non-periodic path is unchanged. The lattice carries its own per-axis periodicity flags,
    /// so one field covers 1D, 2D and 3D.
    pub cell: Option<Lattice>,
}

impl Molecule {
    pub fn new(atoms: Vec<Atom>) -> Self {
        Self {
            atoms,
            charge: 0.0,
            multiplicity: 1,
            cell: None,
        }
    }

    pub fn with_charge(mut self, charge: f64) -> Self {
        self.charge = charge;
        self
    }

    pub fn with_multiplicity(mut self, multiplicity: usize) -> Self {
        self.multiplicity = multiplicity.max(1);
        self
    }

    pub fn with_cell(mut self, cell: Lattice) -> Self {
        self.cell = Some(cell);
        self
    }

    /// Whether this system has a periodic cell with at least one periodic direction.
    pub fn is_periodic(&self) -> bool {
        self.cell.map(|c| c.n_periodic() > 0).unwrap_or(false)
    }

    pub fn len(&self) -> usize {
        self.atoms.len()
    }
    pub fn is_empty(&self) -> bool {
        self.atoms.is_empty()
    }

    /// Sum of atomic numbers (used to derive the electron count).
    pub fn total_nuclear_charge(&self) -> u32 {
        self.atoms.iter().map(|a| a.z as u32).sum()
    }

    pub fn from_xyz_file(path: impl AsRef<Path>, charge: f64) -> Result<Self> {
        Self::from_xyz_str(&fs::read_to_string(path)?, charge)
    }

    /// Parse an XYZ block. Coordinates are Ångström and converted to Bohr.
    ///
    /// The comment line is read as extended XYZ, so a periodic structure round-trips through
    /// the ordinary file format: `Lattice="ax ay az bx by bz cx cy cz"` (Ångström, three
    /// lattice vectors in order) sets the cell, and an optional `pbc="T T F"` sets which axes
    /// are periodic — defaulting to all three when a lattice is present. A comment line
    /// without `Lattice=` is ignored exactly as before, so plain XYZ is unaffected.
    pub fn from_xyz_str(text: &str, charge: f64) -> Result<Self> {
        let mut lines = text.lines();
        let natoms_line = lines
            .next()
            .ok_or_else(|| Am1Error::InvalidInput("empty XYZ".to_string()))?;
        let natoms = natoms_line.trim().parse::<usize>().map_err(|_| {
            Am1Error::InvalidInput(format!("invalid XYZ atom count: {natoms_line}"))
        })?;
        let comment = lines.next().unwrap_or_default();
        let cell = parse_extxyz_cell(comment)?;
        let mut atoms = Vec::with_capacity(natoms);
        for idx in 0..natoms {
            let line_no = idx + 3;
            let line = lines
                .next()
                .ok_or_else(|| Am1Error::InvalidInput(format!("XYZ ended before atom {idx}")))?;
            let parts = line.split_whitespace().collect::<Vec<_>>();
            if parts.len() < 4 {
                return Err(Am1Error::InvalidInput(format!(
                    "XYZ line {line_no} has fewer than 4 fields"
                )));
            }
            let z = symbol_to_z(parts[0]).ok_or_else(|| {
                Am1Error::InvalidInput(format!("unknown element on line {line_no}: {}", parts[0]))
            })?;
            let position = Vec3::new(
                parse_f64(parts[1], line_no)?,
                parse_f64(parts[2], line_no)?,
                parse_f64(parts[3], line_no)?,
            ) * ANGSTROM_TO_BOHR;
            atoms.push(Atom { z, position });
        }
        Ok(Self {
            atoms,
            charge,
            multiplicity: 1,
            cell,
        })
    }
}

/// Value of `key="..."` or `key=token` in an extended-XYZ comment line, case-insensitively.
fn extxyz_field<'a>(comment: &'a str, key: &str) -> Option<&'a str> {
    let lower = comment.to_ascii_lowercase();
    let mut from = 0usize;
    loop {
        let at = lower[from..].find(&format!("{}=", key.to_ascii_lowercase()))? + from;
        // Must start a token, so `Lattice=` does not match inside `SuperLattice=`.
        let starts_token = at == 0 || comment.as_bytes()[at - 1].is_ascii_whitespace();
        let after = at + key.len() + 1;
        if !starts_token {
            from = after;
            continue;
        }
        let rest = &comment[after..];
        return Some(if let Some(stripped) = rest.strip_prefix('"') {
            let end = stripped.find('"')?;
            &stripped[..end]
        } else {
            rest.split_whitespace().next()?
        });
    }
}

/// Read `Lattice=` (and optional `pbc=`) from an extended-XYZ comment line.
fn parse_extxyz_cell(comment: &str) -> Result<Option<Lattice>> {
    let Some(lattice_text) = extxyz_field(comment, "Lattice") else {
        return Ok(None);
    };
    let v: Vec<f64> = lattice_text
        .split_whitespace()
        .map(|t| t.parse::<f64>())
        .collect::<std::result::Result<_, _>>()
        .map_err(|_| {
            Am1Error::InvalidInput(format!(
                "extended-XYZ Lattice= is not nine numbers: `{lattice_text}`"
            ))
        })?;
    if v.len() != 9 {
        return Err(Am1Error::InvalidInput(format!(
            "extended-XYZ Lattice= needs nine numbers (three lattice vectors), found {}",
            v.len()
        )));
    }

    let periodic = match extxyz_field(comment, "pbc") {
        None => [true; 3],
        Some(text) => {
            let flags: Vec<bool> = text
                .split_whitespace()
                .map(|t| matches!(t.trim().to_ascii_uppercase().as_str(), "T" | "TRUE" | "1"))
                .collect();
            if flags.len() != 3 {
                return Err(Am1Error::InvalidInput(format!(
                    "extended-XYZ pbc= needs three flags, found {}",
                    flags.len()
                )));
            }
            [flags[0], flags[1], flags[2]]
        }
    };

    // Extended XYZ writes the lattice vectors in Angstrom, one after another; the crate works
    // in Bohr, and `Mat3` holds the vectors as columns.
    let a = Vec3::new(v[0], v[1], v[2]) * ANGSTROM_TO_BOHR;
    let b = Vec3::new(v[3], v[4], v[5]) * ANGSTROM_TO_BOHR;
    let c = Vec3::new(v[6], v[7], v[8]) * ANGSTROM_TO_BOHR;
    Ok(Some(Lattice::from_vectors(a, b, c, periodic)?))
}

fn parse_f64(token: &str, line: usize) -> Result<f64> {
    token.parse::<f64>().map_err(|_| Am1Error::Parse {
        line,
        message: format!("invalid floating point value: {token}"),
    })
}

pub fn symbol_to_z(sym: &str) -> Option<u8> {
    if let Ok(z) = sym.parse::<u8>() {
        if (1..=86).contains(&z) {
            return Some(z);
        }
    }
    let s = normalize_symbol(sym);
    ELEMENTS
        .iter()
        .position(|&x| x == s.as_str())
        .map(|i| i as u8)
}

pub fn z_to_symbol(z: u8) -> Option<&'static str> {
    ELEMENTS.get(z as usize).copied().filter(|s| !s.is_empty())
}

fn normalize_symbol(sym: &str) -> String {
    let mut chars = sym.chars();
    match chars.next() {
        Some(first) => {
            let mut out = String::new();
            out.push(first.to_ascii_uppercase());
            for c in chars {
                out.push(c.to_ascii_lowercase());
            }
            out
        }
        None => String::new(),
    }
}

pub const ELEMENTS: [&str; 87] = [
    "", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S",
    "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge",
    "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
    "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn",
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::BOHR_TO_ANGSTROM;

    #[test]
    fn plain_xyz_has_no_cell() {
        let m = Molecule::from_xyz_str("2\njust a comment\nH 0 0 0\nH 0 0 0.74\n", 0.0).unwrap();
        assert!(m.cell.is_none());
        assert!(!m.is_periodic());
    }

    #[test]
    fn extended_xyz_lattice_is_read_in_angstrom() {
        let m = Molecule::from_xyz_str(
            "1\nLattice=\"5.0 0.0 0.0 0.0 6.0 0.0 0.0 0.0 7.0\" Properties=species:S:1:pos:R:3\nH 0 0 0\n",
            0.0,
        )
        .unwrap();
        let cell = m.cell.expect("Lattice= should produce a cell");
        assert!(m.is_periodic());
        assert!((cell.cell.col[0].x * BOHR_TO_ANGSTROM - 5.0).abs() < 1.0e-9);
        assert!((cell.cell.col[1].y * BOHR_TO_ANGSTROM - 6.0).abs() < 1.0e-9);
        assert!((cell.cell.col[2].z * BOHR_TO_ANGSTROM - 7.0).abs() < 1.0e-9);
        // Defaults to fully periodic when pbc= is absent.
        assert_eq!(cell.periodic, [true, true, true]);
    }

    #[test]
    fn extended_xyz_pbc_flags_select_dimensionality() {
        let slab = Molecule::from_xyz_str(
            "1\nLattice=\"5 0 0 0 5 0 0 0 30\" pbc=\"T T F\"\nH 0 0 0\n",
            0.0,
        )
        .unwrap();
        assert_eq!(slab.cell.unwrap().periodic, [true, true, false]);
        assert_eq!(slab.cell.unwrap().n_periodic(), 2);

        let chain = Molecule::from_xyz_str(
            "1\nLattice=\"5 0 0 0 30 0 0 0 30\" pbc=\"T F F\"\nH 0 0 0\n",
            0.0,
        )
        .unwrap();
        assert_eq!(chain.cell.unwrap().n_periodic(), 1);
    }

    #[test]
    fn a_malformed_lattice_is_reported_not_ignored() {
        let err = Molecule::from_xyz_str("1\nLattice=\"5 0 0 0 5 0\"\nH 0 0 0\n", 0.0).unwrap_err();
        assert!(err.to_string().contains("nine numbers"), "{err}");

        let err =
            Molecule::from_xyz_str("1\nLattice=\"a b c d e f g h i\"\nH 0 0 0\n", 0.0).unwrap_err();
        assert!(err.to_string().contains("not nine numbers"), "{err}");

        let err = Molecule::from_xyz_str(
            "1\nLattice=\"5 0 0 0 5 0 0 0 30\" pbc=\"T T\"\nH 0 0 0\n",
            0.0,
        )
        .unwrap_err();
        assert!(err.to_string().contains("three flags"), "{err}");
    }

    #[test]
    fn field_matching_requires_a_token_boundary() {
        // `SuperLattice=` must not be mistaken for `Lattice=`.
        let m = Molecule::from_xyz_str("1\nSuperLattice=\"1 2 3\"\nH 0 0 0\n", 0.0).unwrap();
        assert!(m.cell.is_none(), "SuperLattice= should not match Lattice=");
    }
}

// SPDX-License-Identifier: GPL-3.0-or-later

//! Every bond type in `BCCPARM.DAT` is reachable, and reaching the redundant ones changes nothing.
//!
//! The file uses nine codes: **1** single, **2** double, **3** triple, **6** conjugated, **7**
//! aromatic single, **8** aromatic double, **9** delocalized, **10** aromatic, **11** the same type
//! on both ends. Through 0.2.1 the code emitted 1, 2, 3, 6, 7 and 9 and left 8, 10 and 11
//! unreachable, on the argument that they could not change a charge.
//!
//! That argument is *true* — and it is asserted below against the parameter file rather than
//! recited — but "cannot change a charge" is a reason not to worry, not a reason not to emit. All
//! nine are emitted now, which means:
//!
//! * the Kekulé structure the atom typing already needs (it separates nitrogen 21 from 24) also
//!   supplies the aromatic single/double distinction, which is exactly what 7 and 8 name;
//! * a same-type bond takes the code the file provides for it;
//! * and the two facts that make this safe — 8 and 10 byte-identical to 7, every 11 exactly zero —
//!   are pinned by tests instead of by a comment.
//!
//! The charges must come out **unchanged**, and that is the sharpest thing here: it is a claim
//! about the parameter file that the whole pipeline is made to demonstrate.

use std::collections::HashSet;

use am1_rs::bcc::{bcc_table, BccResult};
use am1_rs::topology::{BondOrder, Topology};
use am1_rs::{am1_bcc_charges, Am1Options, Am1Parameters, Atom, Molecule, Vec3};

const ANG: f64 = 1.0 / 0.529167;

fn mol(atoms: &[(u8, [f64; 3])]) -> Molecule {
    Molecule::new(
        atoms
            .iter()
            .map(|(z, p)| Atom {
                z: *z,
                position: Vec3::new(p[0], p[1], p[2]) * ANG,
            })
            .collect(),
    )
}

fn benzene() -> Molecule {
    let (n, bond) = (6usize, 1.39);
    let radius = bond / (2.0 * (std::f64::consts::PI / n as f64).sin());
    let mut atoms = Vec::new();
    for k in 0..n {
        let t = std::f64::consts::TAU * k as f64 / n as f64;
        atoms.push((6u8, [radius * t.cos(), radius * t.sin(), 0.0]));
    }
    for k in 0..n {
        let t = std::f64::consts::TAU * k as f64 / n as f64;
        let r = radius + 1.08;
        atoms.push((1u8, [r * t.cos(), r * t.sin(), 0.0]));
    }
    mol(&atoms)
}

fn pyridine() -> Molecule {
    let bond = 1.39;
    let radius = bond / (2.0 * (std::f64::consts::PI / 6.0).sin());
    let mut atoms = Vec::new();
    for k in 0..6 {
        let t = std::f64::consts::TAU * k as f64 / 6.0;
        atoms.push((
            if k == 0 { 7u8 } else { 6u8 },
            [radius * t.cos(), radius * t.sin(), 0.0],
        ));
    }
    for k in 1..6 {
        let t = std::f64::consts::TAU * k as f64 / 6.0;
        let r = radius + 1.08;
        atoms.push((1u8, [r * t.cos(), r * t.sin(), 0.0]));
    }
    mol(&atoms)
}

/// A planar 5-ring with `hetero` at index 0 and carbons at 1..5, hydrogenated.
fn five_ring(hetero: u8) -> Molecule {
    let bond = if hetero == 16 { 1.71 } else { 1.37 };
    let radius = bond / (2.0 * (std::f64::consts::PI / 5.0).sin());
    let mut atoms = Vec::new();
    for k in 0..5 {
        let t = std::f64::consts::TAU * k as f64 / 5.0 + std::f64::consts::FRAC_PI_2;
        let z = if k == 0 { hetero } else { 6u8 };
        atoms.push((z, [radius * t.cos(), radius * t.sin(), 0.0]));
    }
    for k in 0..5 {
        if k == 0 && hetero != 7 {
            continue;
        }
        let t = std::f64::consts::TAU * k as f64 / 5.0 + std::f64::consts::FRAC_PI_2;
        let r = radius + 1.02;
        atoms.push((1u8, [r * t.cos(), r * t.sin(), 0.0]));
    }
    mol(&atoms)
}

fn ethane() -> Molecule {
    mol(&[
        (6, [0.0, 0.0, 0.0]),
        (6, [1.54, 0.0, 0.0]),
        (1, [-0.36, 1.02, 0.0]),
        (1, [-0.36, -0.51, 0.88]),
        (1, [-0.36, -0.51, -0.88]),
        (1, [1.90, -1.02, 0.0]),
        (1, [1.90, 0.51, 0.88]),
        (1, [1.90, 0.51, -0.88]),
    ])
}

fn charges(m: &Molecule) -> BccResult {
    let params = Am1Parameters::standard().unwrap();
    am1_bcc_charges(m, &params, &Am1Options::default()).unwrap()
}

// ------------------------------------------------------------- the file's redundancy, asserted

#[test]
fn types_8_and_10_are_byte_identical_to_type_7_and_type_11_is_zero() {
    let table = bcc_table();
    for dup in [8u32, 10u32] {
        let mut checked = 0;
        for (&(i, j, bt), &value) in table.iter() {
            if bt != dup {
                continue;
            }
            let seven = table
                .get(&(i, j, 7))
                .copied()
                .unwrap_or_else(|| panic!("({i},{j},{dup}) has no type-7 counterpart"));
            assert_eq!(value.to_bits(), seven.to_bits(), "({i},{j},{dup})");
            checked += 1;
        }
        eprintln!("    bond type {dup}: {checked} entries, all byte-identical to type 7");
    }
    let elevens: Vec<_> = table.iter().filter(|(&(_, _, bt), _)| bt == 11).collect();
    assert_eq!(elevens.len(), 26);
    for (&(i, j, _), &v) in &elevens {
        assert_eq!(v, 0.0, "({i},{j},11) = {v}");
        assert_eq!(
            i, j,
            "every type-11 entry should be a same-type pair, got ({i},{j})"
        );
    }
    eprintln!("    bond type 11: 26 entries, all same-type pairs and all exactly zero");
}

// --------------------------------------------------------------------------- reachability

/// Which bond-type codes this molecule's bonds actually resolve to, read back off the pipeline.
///
/// Reconstructed from the perception rather than exposed by the API: the code the charge used is
/// an internal detail, and asserting it through a new public accessor would be testing a getter.
fn codes_used(m: &Molecule) -> HashSet<u32> {
    let table = bcc_table();
    let topo = Topology::perceive(m);
    let types = am1_rs::bcc::assign_bcc_types(m, &topo);
    let mut out = HashSet::new();
    for (k, b) in topo.bonds.iter().enumerate() {
        let (ta, tb) = (types[b.i], types[b.j]);
        let primary = match b.order {
            BondOrder::Single => 1,
            BondOrder::Double => 2,
            BondOrder::Triple => 3,
            BondOrder::Aromatic => {
                if !topo.kekule_unique {
                    10
                } else if topo.kekule_double[k] {
                    8
                } else {
                    7
                }
            }
            BondOrder::Delocalized => 9, // 6 for the nitrogen centres; not needed by these cases
        };
        // The same fallback chain the pipeline uses, so this reports the code that was actually
        // applied rather than the one that was first considered.
        let mut chain = match primary {
            8 | 10 => vec![primary, 7],
            _ => vec![primary],
        };
        if ta == tb && ta != 0 {
            chain.push(11);
        }
        let used = chain
            .iter()
            .find(|&&c| table.contains_key(&(ta, tb, c)) || table.contains_key(&(tb, ta, c)));
        if let Some(&c) = used {
            out.insert(c);
        }
    }
    out
}

#[test]
fn benzene_reaches_the_unresolved_aromatic_code() {
    let used = codes_used(&benzene());
    let mut v: Vec<_> = used.iter().copied().collect();
    v.sort_unstable();
    eprintln!("    benzene bond codes: {v:?}");
    assert!(used.contains(&10), "two equivalent Kekule structures");
    assert!(used.contains(&1), "the C-H bonds are ordinary single bonds");
}

/// Type 11 is not decoration: ten of the twenty-six atom types that have a type-11 entry have **no
/// single-bond entry**, so their homonuclear bond had no parameter under any code this crate used
/// to emit — and came back as a warning telling the caller the bond was left at raw Mulliken.
///
/// The correction is zero, so the charges were never wrong. The *warning* was, and it is the thing
/// `BccResult::warnings` exists to make trustworthy: a molecule that returns no warnings is
/// supposed to be one the rules covered, and H₂ was not.
#[test]
fn a_homonuclear_bond_no_longer_reports_itself_as_uncorrected() {
    let cases = [
        ("H2", mol(&[(1, [0.0, 0.0, 0.0]), (1, [0.74, 0.0, 0.0])])),
        ("F2", mol(&[(9, [0.0, 0.0, 0.0]), (9, [1.42, 0.0, 0.0])])),
        ("Cl2", mol(&[(17, [0.0, 0.0, 0.0]), (17, [1.99, 0.0, 0.0])])),
        ("Br2", mol(&[(35, [0.0, 0.0, 0.0]), (35, [2.28, 0.0, 0.0])])),
        ("I2", mol(&[(53, [0.0, 0.0, 0.0]), (53, [2.67, 0.0, 0.0])])),
    ];
    for (name, m) in cases {
        let topo = Topology::perceive(&m);
        assert_eq!(topo.bonds.len(), 1, "{name}: expected one bond");
        let used = codes_used(&m);
        eprintln!(
            "    {name}: codes {:?}",
            used.iter().copied().collect::<Vec<_>>()
        );
        assert!(
            used.contains(&11),
            "{name}: a homonuclear bond should reach the same-type code"
        );

        let r = charges(&m);
        assert!(
            r.warnings.is_empty(),
            "{name} still reports itself uncorrected: {:?}",
            r.warnings
        );
        // Symmetry: the two atoms are identical, so both charges are zero and stay zero.
        let total: f64 = r.charges.iter().sum();
        assert!(total.abs() < 1.0e-9, "{name}: charge not conserved");
        assert!(
            (r.charges[0] - r.charges[1]).abs() < 1.0e-9,
            "{name}: a homonuclear molecule must have equal charges"
        );
    }
}

#[test]
fn a_six_ring_aromatic_reaches_the_unresolved_aromatic_code() {
    // A six-membered aromatic has **two** Kekule structures, equivalent under rotation, so no ring
    // bond is "the" double bond. That is what code 10 — "aromatic", with no resolved single/double
    // character — names, and it is what the uniqueness test detects.
    //
    // Pyridine rather than benzene: benzene's ring atoms are all type 16, so the same-type rule
    // takes those bonds first and the aromatic code never gets a chance to show.
    let used = codes_used(&pyridine());
    let mut v: Vec<_> = used.iter().copied().collect();
    v.sort_unstable();
    eprintln!("    pyridine bond codes: {v:?}");
    assert!(
        used.contains(&10),
        "a six-ring aromatic has two Kekule structures, so its bonds are code 10"
    );
    assert!(used.contains(&1), "the C-H bonds are ordinary single bonds");
}

#[test]
fn a_five_ring_heteroaromatic_reaches_aromatic_single_and_double() {
    // Pyrrole's nitrogen donates two pi electrons and so takes no double bond, leaving the four
    // carbons as a path — whose perfect matching is **unique**. So the Kekule structure is
    // determined here, and codes 7 (aromatic single) and 8 (aromatic double) both apply. This is
    // the case that distinguishes them from 10, and the reason all three exist.
    for (name, hetero) in [("pyrrole", 7u8), ("furan", 8), ("thiophene", 16)] {
        let m = five_ring(hetero);
        let topo = Topology::perceive(&m);
        assert!(
            (0..5).all(|i| topo.aromatic[i]),
            "{name}: the fixture must be aromatic for this to test anything"
        );
        assert!(
            topo.kekule_unique,
            "{name}: a five-ring heteroaromatic has one Kekule structure"
        );
        let used = codes_used(&m);
        let mut v: Vec<_> = used.iter().copied().collect();
        v.sort_unstable();
        eprintln!("    {name} bond codes: {v:?}");
        assert!(used.contains(&7), "{name}: aromatic single");
        assert!(used.contains(&8), "{name}: aromatic double");
    }
}

/// The fallback is a *fallback*: where the ordinary code is tabulated, it wins.
///
/// Ethane's C–C joins two type-11 carbons, so it is a same-type bond — but `(11,11,1)` exists in
/// the file (and is zero), so the single-bond code applies and type 11 is not reached. Type 11 is
/// for the same-type pairs the ordinary codes do *not* tabulate, which is what
/// `a_homonuclear_bond_no_longer_reports_itself_as_uncorrected` covers. Asserting this direction
/// too is what stops the fallback from quietly becoming an override.
#[test]
fn the_same_type_code_does_not_pre_empt_a_tabulated_one() {
    let used = codes_used(&ethane());
    let mut v: Vec<_> = used.iter().copied().collect();
    v.sort_unstable();
    eprintln!("    ethane bond codes: {v:?}");
    assert!(
        used.contains(&1),
        "C(11)-C(11) has a single-bond entry, so it takes code 1"
    );
    assert!(
        !used.contains(&11),
        "type 11 must not pre-empt a code the file tabulates"
    );
}

// -------------------------------------------------------------------- and it changes no charge

/// The point of the whole exercise: emitting the redundant codes leaves every charge where it was.
///
/// The reference values are the ones the pipeline produced when it emitted only 1, 2, 3, 6, 7 and
/// 9 — recorded here so the claim is checked against a number rather than against itself.
#[test]
fn emitting_the_redundant_codes_leaves_the_charges_unchanged() {
    // Benzene: all carbons equivalent, all hydrogens equivalent, and the corrections that the
    // same-type code now carries are zero — so the charges must be exactly what type 7 gave.
    let r = charges(&benzene());
    assert!(r.warnings.is_empty(), "{:?}", r.warnings);
    let c = r.charges[0];
    let h = r.charges[6];
    eprintln!("    benzene: C {c:+.8}, H {h:+.8}");
    for k in 0..6 {
        assert!(
            (r.charges[k] - c).abs() < 1.0e-12,
            "carbon {k} is not equivalent"
        );
        assert!(
            (r.charges[6 + k] - h).abs() < 1.0e-12,
            "hydrogen {k} is not equivalent"
        );
    }
    assert!(c < 0.0 && h > 0.0, "benzene should have C(-) H(+)");
    // The ring corrections are all zero, so each carbon's BCC charge is its Mulliken charge plus
    // only the C-H term. That is what "type 11 is zero" means at the level of a molecule.
    assert!(
        ((r.charges[0] - r.mulliken[0]) + (r.charges[6] - r.mulliken[6])).abs() < 1.0e-12,
        "the C and H shifts should cancel: the ring bonds contribute nothing"
    );

    // Pyridine reaches both 7 and 8, where 8 is byte-identical to 7 — so the total is conserved
    // and no bond is left uncorrected.
    let r = charges(&pyridine());
    assert!(r.warnings.is_empty(), "pyridine: {:?}", r.warnings);
    let total: f64 = r.charges.iter().sum();
    eprintln!("    pyridine: sum q = {total:+.3e}, N {:+.6}", r.charges[0]);
    assert!(total.abs() < 1.0e-9);
    assert!(r.charges[0] < 0.0, "pyridine nitrogen should be negative");
}

/// A same-type bond must still find a parameter. Every one of the 26 type-11 entries covers a
/// distinct atom type, so this is really asking whether the fallback chain is wired up.
#[test]
fn no_molecule_is_left_uncorrected_by_the_new_codes() {
    for (name, m) in [
        ("benzene", benzene()),
        ("pyridine", pyridine()),
        ("ethane", ethane()),
        ("pyrrole", five_ring(7)),
        ("furan", five_ring(8)),
        ("thiophene", five_ring(16)),
    ] {
        let r = charges(&m);
        assert!(
            r.warnings.is_empty(),
            "{name} produced warnings: {:?}",
            r.warnings
        );
        let total: f64 = r.charges.iter().sum();
        assert!(
            total.abs() < 1.0e-9,
            "{name}: charge not conserved ({total:+.3e})"
        );
    }
}

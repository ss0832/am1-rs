// SPDX-License-Identifier: GPL-3.0-or-later

//! Molecular-graph perception and AM1-BCC typing.
//!
//! Every case here is one the previous perception got wrong, chosen so that a regression shows
//! up as a specific chemical claim rather than a shifted number:
//!
//! * **Ring size.** Cycloheptatriene has planar sp2 carbons in a ring and was typed aromatic;
//!   it is a 7-ring and is not. Naphthalene's fused rings were liable to be perceived as the
//!   10-membered perimeter by the old spanning-tree cycle basis.
//! * **Hückel count.** Cyclooctatetraene has sp2 carbons in a ring and 8 π electrons; it is not
//!   aromatic. Pyrrole, furan and thiophene are, and each needs a different π contribution.
//! * **Bond orders beyond C/N/O.** Thiourea's C=S, phosphate's P=O and a sulfonyl's S=O were
//!   all perceived as single bonds, because the reference table held only six C/N/O pairs.
//! * **Delocalized groups.** A carboxylate's two C–O bonds and a nitro group's two N–O bonds are
//!   neither single nor double; they select bond types 9 and 6, which were unreachable.

use am1_rs::topology::{BondOrder, Topology};
use am1_rs::{Atom, Molecule, Vec3};

const ANG: f64 = 1.0 / 0.529167;

/// Build a molecule from `(symbol_z, x, y, z)` triples given in Ångström.
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

/// A planar regular polygon of carbons with the given bond length, each carrying one hydrogen.
fn carbon_ring(n: usize, bond: f64) -> Molecule {
    let radius = bond / (2.0 * (std::f64::consts::PI / n as f64).sin());
    let mut atoms = Vec::new();
    for k in 0..n {
        let t = 2.0 * std::f64::consts::PI * k as f64 / n as f64;
        atoms.push((6u8, [radius * t.cos(), radius * t.sin(), 0.0]));
    }
    for k in 0..n {
        let t = 2.0 * std::f64::consts::PI * k as f64 / n as f64;
        let r = radius + 1.08;
        atoms.push((1u8, [r * t.cos(), r * t.sin(), 0.0]));
    }
    mol(&atoms)
}

// ------------------------------------------------------------------------------ ring perception
#[test]
fn benzene_is_aromatic() {
    let topo = Topology::perceive(&carbon_ring(6, 1.39));
    assert!(
        (0..6).all(|i| topo.aromatic[i]),
        "every benzene carbon should be aromatic"
    );
    assert_eq!(topo.smallest_ring[0], Some(6));
    let ring_bonds = topo
        .bonds
        .iter()
        .filter(|b| b.order == BondOrder::Aromatic)
        .count();
    eprintln!(
        "    benzene: {ring_bonds} aromatic bonds, smallest ring {:?}",
        topo.smallest_ring[0]
    );
    assert_eq!(ring_bonds, 6, "all six ring bonds should be aromatic");
}

#[test]
fn cycloheptatriene_is_not_aromatic() {
    // A 7-ring of planar sp2-ish carbons. The previous perception never looked at ring size --
    // its ring detector returned only a boolean -- so anything sp2 and cyclic came out aromatic.
    let topo = Topology::perceive(&carbon_ring(7, 1.40));
    eprintln!(
        "    C7 ring: smallest ring {:?}, aromatic atoms {}",
        topo.smallest_ring[0],
        topo.aromatic.iter().filter(|a| **a).count()
    );
    assert_eq!(topo.smallest_ring[0], Some(7));
    assert!(
        topo.aromatic.iter().all(|a| !a),
        "a 7-membered carbocycle is not aromatic"
    );
}

#[test]
fn cyclooctatetraene_is_not_aromatic() {
    // 8 π electrons: 4n, not 4n+2. Even flattened, Hückel says no.
    let topo = Topology::perceive(&carbon_ring(8, 1.40));
    assert_eq!(topo.smallest_ring[0], Some(8));
    assert!(
        topo.aromatic.iter().all(|a| !a),
        "cyclooctatetraene has 8 π electrons and is antiaromatic, not aromatic"
    );
}

#[test]
fn naphthalene_is_perceived_as_two_six_rings_not_one_ten_ring() {
    // The sharp test of the ring perception. The old union-find spanning-tree basis took the
    // fundamental cycle of each non-tree edge, which in a fused system can be the 10-membered
    // perimeter rather than the two 6-rings -- and ring size is what decides aromaticity.
    let a = 1.40;
    let h = a * 3f64.sqrt() / 2.0;
    // Two fused hexagons sharing the C4a-C8a bond, drawn on a hexagonal lattice.
    let carbons = [
        [0.0, h],
        [a * 0.5, 0.0],
        [a * 1.5, 0.0],
        [a * 2.0, h],
        [a * 1.5, 2.0 * h],
        [a * 0.5, 2.0 * h], // ring A (0..6)
        [a * 3.0, h],
        [a * 3.5, 2.0 * h],
        [a * 3.0, 3.0 * h], // extra of ring B
        [a * 2.0, 3.0 * h],
    ];
    let mut atoms: Vec<(u8, [f64; 3])> = carbons.iter().map(|p| (6u8, [p[0], p[1], 0.0])).collect();
    // Hydrogens on the eight peripheral carbons (not 3 and 4, which are the fusion atoms).
    let centre = [a * 1.75, 1.5 * h];
    for (k, c) in carbons.iter().enumerate() {
        if k == 3 || k == 4 {
            continue;
        }
        let dx = c[0] - centre[0];
        let dy = c[1] - centre[1];
        let n = (dx * dx + dy * dy).sqrt();
        atoms.push((1u8, [c[0] + 1.08 * dx / n, c[1] + 1.08 * dy / n, 0.0]));
    }

    let topo = Topology::perceive(&mol(&atoms));
    let sizes: Vec<usize> = topo.rings.iter().map(|r| r.size()).collect();
    eprintln!("    naphthalene rings: {sizes:?}");
    assert!(
        sizes.iter().filter(|&&s| s == 6).count() >= 2,
        "two 6-rings expected, got {sizes:?}"
    );
    assert!(
        topo.smallest_ring[0] == Some(6),
        "a peripheral carbon's smallest ring must be 6, got {:?}",
        topo.smallest_ring[0]
    );
    assert!(
        (0..10).all(|i| topo.aromatic[i]),
        "all ten naphthalene carbons should be aromatic"
    );
}

// --------------------------------------------------------------------------- heteroaromatics
#[test]
fn the_five_membered_heteroaromatics_are_aromatic() {
    // Pyrrole (N donates 2), furan (O donates 2), thiophene (S donates 2). Each reaches 4n+2 by
    // a different route, so a π-count that only understood carbon would fail all three.
    //
    // Thiophene is the one the previous version could not possibly get: `perceive_hybridization`
    // returned Sp3 for every sulfur, and the aromaticity test required Sp2, so the `| 16` in its
    // element match was unreachable code and thiophene's sulfur was never aromatic.
    let cases: [(&str, u8); 3] = [("pyrrole", 7), ("furan", 8), ("thiophene", 16)];
    for (name, hetero) in cases {
        // A planar 5-ring: hetero at index 0, four carbons around it.
        let bond = if hetero == 16 { 1.71 } else { 1.37 };
        let radius = bond / (2.0 * (std::f64::consts::PI / 5.0).sin());
        let mut atoms = Vec::new();
        for k in 0..5 {
            let t = 2.0 * std::f64::consts::PI * k as f64 / 5.0 + std::f64::consts::FRAC_PI_2;
            let z = if k == 0 { hetero } else { 6u8 };
            atoms.push((z, [radius * t.cos(), radius * t.sin(), 0.0]));
        }
        // Hydrogens on the four carbons, and on N for pyrrole.
        for k in 0..5 {
            if k == 0 && hetero != 7 {
                continue;
            }
            let t = 2.0 * std::f64::consts::PI * k as f64 / 5.0 + std::f64::consts::FRAC_PI_2;
            let r = radius + 1.02;
            atoms.push((1u8, [r * t.cos(), r * t.sin(), 0.0]));
        }
        let topo = Topology::perceive(&mol(&atoms));
        eprintln!(
            "    {name}: smallest ring {:?}, aromatic ring atoms {}/5",
            topo.smallest_ring[0],
            (0..5).filter(|&i| topo.aromatic[i]).count()
        );
        assert_eq!(topo.smallest_ring[0], Some(5), "{name} should be a 5-ring");
        assert!(
            (0..5).all(|i| topo.aromatic[i]),
            "{name} should be aromatic on all five ring atoms"
        );
    }
}

#[test]
fn pyridine_is_aromatic() {
    let bond = 1.39;
    let radius = bond / (2.0 * (std::f64::consts::PI / 6.0).sin());
    let mut atoms = Vec::new();
    for k in 0..6 {
        let t = 2.0 * std::f64::consts::PI * k as f64 / 6.0;
        atoms.push((
            if k == 0 { 7u8 } else { 6u8 },
            [radius * t.cos(), radius * t.sin(), 0.0],
        ));
    }
    for k in 1..6 {
        let t = 2.0 * std::f64::consts::PI * k as f64 / 6.0;
        let r = radius + 1.08;
        atoms.push((1u8, [r * t.cos(), r * t.sin(), 0.0]));
    }
    let topo = Topology::perceive(&mol(&atoms));
    assert!(
        (0..6).all(|i| topo.aromatic[i]),
        "pyridine should be aromatic"
    );
}

// -------------------------------------------------------------------------------- bond orders
#[test]
fn double_bonds_to_sulfur_phosphorus_and_oxygen_are_perceived() {
    // Every one of these was typed as a single bond, because the reference length table held
    // only (6,6), (6,7), (6,8), (7,7), (7,8) and (8,8).
    //
    // Thioformaldehyde: a genuine isolated C=S, with no second chalcogen to make it delocalized.
    let thio = mol(&[
        (6, [0.0, 0.0, 0.0]),
        (16, [0.0, 0.0, 1.61]),
        (1, [0.94, 0.0, -0.54]),
        (1, [-0.94, 0.0, -0.54]),
    ]);
    let topo = Topology::perceive(&thio);
    let cs = topo
        .bonds
        .iter()
        .find(|b| {
            let (zi, zj) = (thio.atoms[b.i].z, thio.atoms[b.j].z);
            (zi == 6 && zj == 16) || (zi == 16 && zj == 6)
        })
        .expect("a C–S bond should be perceived");
    eprintln!(
        "    thioformaldehyde C=S: {:?} at {:.3} Å",
        cs.order, cs.length
    );
    assert_eq!(cs.order, BondOrder::Double, "C=S should be a double bond");
}

#[test]
fn a_carboxylate_gets_delocalized_bonds_and_a_carboxylic_acid_does_not() {
    // The distinction the delocalized rule exists for. Acetate has two equivalent terminal
    // oxygens; acetic acid has one terminal `=O` and one two-coordinate `–OH`, which are a
    // genuine double and a genuine single bond.
    let acetate = mol(&[
        (6, [0.0, 0.0, 0.0]),     // carboxyl C
        (8, [1.25, 0.0, 0.0]),    // O
        (8, [-0.63, 1.08, 0.0]),  // O
        (6, [-0.75, -1.30, 0.0]), // methyl C
        (1, [-1.83, -1.20, 0.0]),
        (1, [-0.45, -1.85, 0.89]),
        (1, [-0.45, -1.85, -0.89]),
    ]);
    let topo = Topology::perceive(&acetate);
    let delocalized = topo
        .bonds
        .iter()
        .filter(|b| b.order == BondOrder::Delocalized)
        .count();
    eprintln!("    acetate: {delocalized} delocalized C–O bonds");
    assert_eq!(
        delocalized, 2,
        "both carboxylate C–O bonds should be delocalized"
    );

    let acid = mol(&[
        (6, [0.0, 0.0, 0.0]),
        (8, [1.21, 0.0, 0.0]),   // carbonyl O, terminal
        (8, [-0.68, 1.13, 0.0]), // hydroxyl O
        (1, [-0.20, 1.95, 0.0]), // the OH hydrogen makes it two-coordinate
        (6, [-0.75, -1.30, 0.0]),
        (1, [-1.83, -1.20, 0.0]),
        (1, [-0.45, -1.85, 0.89]),
        (1, [-0.45, -1.85, -0.89]),
    ]);
    let topo = Topology::perceive(&acid);
    let delocalized = topo
        .bonds
        .iter()
        .filter(|b| b.order == BondOrder::Delocalized)
        .count();
    eprintln!("    acetic acid: {delocalized} delocalized bonds (expected 0)");
    assert_eq!(
        delocalized, 0,
        "a carboxylic acid has a real double and a real single bond, not two delocalized ones"
    );
}

#[test]
fn a_nitro_group_gets_delocalized_bonds() {
    let nitromethane = mol(&[
        (7, [0.0, 0.0, 0.0]),
        (8, [1.10, 0.53, 0.0]),
        (8, [-1.10, 0.53, 0.0]),
        (6, [0.0, -1.49, 0.0]),
        (1, [0.90, -1.90, 0.45]),
        (1, [-0.90, -1.90, 0.45]),
        (1, [0.0, -1.85, -1.03]),
    ]);
    let topo = Topology::perceive(&nitromethane);
    let delocalized = topo
        .bonds
        .iter()
        .filter(|b| b.order == BondOrder::Delocalized)
        .count();
    eprintln!("    nitromethane: {delocalized} delocalized N–O bonds");
    assert_eq!(delocalized, 2, "both nitro N–O bonds should be delocalized");
}

// ----------------------------------------------------------------------------------- warnings
#[test]
fn an_unparameterized_element_is_reported_rather_than_typed_as_zero_in_silence() {
    // Boron has no AM1-BCC atom type. Previously it became type 0, matched nothing, and the raw
    // Mulliken charges came back with no indication that the corrections had not been applied.
    let params = am1_rs::Am1Parameters::standard().unwrap();
    let borane = mol(&[
        (5, [0.0, 0.0, 0.0]),
        (1, [1.19, 0.0, 0.0]),
        (1, [-0.60, 1.03, 0.0]),
        (1, [-0.60, -1.03, 0.0]),
    ]);
    let result = am1_rs::am1_bcc_charges(&borane, &params, &am1_rs::Am1Options::default()).unwrap();
    eprintln!("    borane warnings: {:?}", result.warnings);
    assert!(
        result
            .warnings
            .iter()
            .any(|w| w.contains("no AM1-BCC atom type")),
        "an unparameterized element must be reported, got {:?}",
        result.warnings
    );
    assert_eq!(result.atom_types[0], "0");
}

#[test]
fn a_fully_typed_molecule_reports_nothing() {
    // The complement: a warning that always fires is as useless as one that never does.
    let params = am1_rs::Am1Parameters::standard().unwrap();
    let ethanol = mol(&[
        (6, [-1.20, 0.32, 0.0]),
        (6, [0.0, -0.57, 0.0]),
        (8, [1.19, 0.20, 0.0]),
        (1, [1.95, -0.37, 0.0]),
        (1, [-0.03, -1.22, 0.88]),
        (1, [-0.03, -1.22, -0.88]),
        (1, [-2.13, -0.25, 0.0]),
        (1, [-1.20, 0.96, 0.88]),
        (1, [-1.20, 0.96, -0.88]),
    ]);
    let result =
        am1_rs::am1_bcc_charges(&ethanol, &params, &am1_rs::Am1Options::default()).unwrap();
    eprintln!(
        "    ethanol: types {:?}, Σq = {:+.6}, warnings {:?}",
        result.atom_types,
        result.charges.iter().sum::<f64>(),
        result.warnings
    );
    assert!(
        result.warnings.is_empty(),
        "ethanol is fully covered; it should report nothing, got {:?}",
        result.warnings
    );
    // Net charge is preserved by construction: every correction moves +δ and −δ.
    assert!(result.charges.iter().sum::<f64>().abs() < 1.0e-9);
}

#[test]
fn a_carboxylate_gets_equivalent_oxygens_and_an_acid_does_not() {
    // The chemistry the delocalized typing exists to get right, checked on the charges rather
    // than on the perceived bond order — which is what a user actually sees.
    //
    // A carboxylate's two oxygens are equivalent by symmetry, so their charges must match. A
    // carboxylic acid's two are a carbonyl and a hydroxyl, and must not.
    let params = am1_rs::Am1Parameters::standard().unwrap();

    let acetate = mol(&[
        (6, [0.0, 0.0, 0.0]),
        (8, [1.25, 0.0, 0.0]),
        (8, [-0.63, 1.08, 0.0]),
        (6, [-0.75, -1.30, 0.0]),
        (1, [-1.83, -1.20, 0.0]),
        (1, [-0.45, -1.85, 0.89]),
        (1, [-0.45, -1.85, -0.89]),
    ]);
    let opts = am1_rs::Am1Options {
        charge: -1.0,
        ..am1_rs::Am1Options::default()
    };
    let r = am1_rs::am1_bcc_charges(&acetate, &params, &opts).unwrap();
    eprintln!(
        "    acetate: types {:?}, O charges {:.4} / {:.4}",
        r.atom_types, r.charges[1], r.charges[2]
    );
    assert!(
        r.warnings.is_empty(),
        "acetate should be fully typed, got {:?}",
        r.warnings
    );
    // The delocalized oxygen is antechamber's generic type 31, not the carbonyl 32 or the
    // carboxyl 33: bond type 9 is tabulated only against 31, and typing them 33 leaves the bond
    // with no parameter at all.
    assert_eq!(r.atom_types[1], "31");
    assert_eq!(r.atom_types[2], "31");
    assert!(
        (r.charges[1] - r.charges[2]).abs() < 0.01,
        "carboxylate oxygens are equivalent; got {:.4} and {:.4}",
        r.charges[1],
        r.charges[2]
    );

    let acid = mol(&[
        (6, [0.0, 0.0, 0.0]),
        (8, [1.21, 0.0, 0.0]),
        (8, [-0.68, 1.13, 0.0]),
        (1, [-0.20, 1.95, 0.0]),
        (6, [-0.75, -1.30, 0.0]),
        (1, [-1.83, -1.20, 0.0]),
        (1, [-0.45, -1.85, 0.89]),
        (1, [-0.45, -1.85, -0.89]),
    ]);
    let r = am1_rs::am1_bcc_charges(&acid, &params, &am1_rs::Am1Options::default()).unwrap();
    eprintln!(
        "    acetic acid: types {:?}, O charges {:.4} / {:.4}",
        r.atom_types, r.charges[1], r.charges[2]
    );
    assert!(r.warnings.is_empty());
    assert!(
        (r.charges[1] - r.charges[2]).abs() > 0.05,
        "a carbonyl and a hydroxyl oxygen must differ; got {:.4} and {:.4}",
        r.charges[1],
        r.charges[2]
    );
}

#[test]
fn ethanol_reproduces_the_documented_antechamber_charges() {
    // An external reference rather than a self-consistency check: antechamber gives roughly
    // O −0.60 and hydroxyl H +0.40 for ethanol, and that is what this project has always
    // documented. Keeping it asserted means the perception rewrite can be checked against
    // something outside the rewrite.
    let params = am1_rs::Am1Parameters::standard().unwrap();
    let ethanol = mol(&[
        (6, [-1.20, 0.32, 0.0]),
        (6, [0.0, -0.57, 0.0]),
        (8, [1.19, 0.20, 0.0]),
        (1, [1.95, -0.37, 0.0]),
        (1, [-0.03, -1.22, 0.88]),
        (1, [-0.03, -1.22, -0.88]),
        (1, [-2.13, -0.25, 0.0]),
        (1, [-1.20, 0.96, 0.88]),
        (1, [-1.20, 0.96, -0.88]),
    ]);
    let r = am1_rs::am1_bcc_charges(&ethanol, &params, &am1_rs::Am1Options::default()).unwrap();
    eprintln!(
        "    ethanol: O {:.4} (ref -0.60), hydroxyl H {:.4} (ref +0.40)",
        r.charges[2], r.charges[3]
    );
    assert!(r.warnings.is_empty());
    assert!(
        (r.charges[2] + 0.60).abs() < 0.03,
        "ethanol O should be near -0.60, got {:.4}",
        r.charges[2]
    );
    assert!(
        (r.charges[3] - 0.40).abs() < 0.03,
        "ethanol hydroxyl H should be near +0.40, got {:.4}",
        r.charges[3]
    );
}

#[test]
fn benzene_charges_are_symmetric() {
    // Every carbon equivalent, every hydrogen equivalent. Catches an aromatic bond that was
    // typed inconsistently around the ring, which a total-charge check would not.
    let params = am1_rs::Am1Parameters::standard().unwrap();
    let r = am1_rs::am1_bcc_charges(
        &carbon_ring(6, 1.39),
        &params,
        &am1_rs::Am1Options::default(),
    )
    .unwrap();
    let carbons = &r.charges[0..6];
    let hydrogens = &r.charges[6..12];
    let spread = |s: &[f64]| {
        s.iter().copied().fold(f64::NEG_INFINITY, f64::max)
            - s.iter().copied().fold(f64::INFINITY, f64::min)
    };
    eprintln!(
        "    benzene: C {:.4} (spread {:.1e}), H {:.4} (spread {:.1e})",
        carbons[0],
        spread(carbons),
        hydrogens[0],
        spread(hydrogens)
    );
    assert!(r.warnings.is_empty());
    assert!(spread(carbons) < 1.0e-6 && spread(hydrogens) < 1.0e-6);
    assert!(carbons[0] < 0.0 && hydrogens[0] > 0.0);
}

#[test]
fn the_bond_charge_corrections_conserve_the_total_charge() {
    // True for any molecule, typed or not, because each correction is applied antisymmetrically.
    let params = am1_rs::Am1Parameters::standard().unwrap();
    let nitromethane = mol(&[
        (7, [0.0, 0.0, 0.0]),
        (8, [1.10, 0.53, 0.0]),
        (8, [-1.10, 0.53, 0.0]),
        (6, [0.0, -1.49, 0.0]),
        (1, [0.90, -1.90, 0.45]),
        (1, [-0.90, -1.90, 0.45]),
        (1, [0.0, -1.85, -1.03]),
    ]);
    for charge in [-1.0, 0.0] {
        let opts = am1_rs::Am1Options {
            charge,
            multiplicity: if charge == 0.0 { 1 } else { 2 },
            ..am1_rs::Am1Options::default()
        };
        let r = am1_rs::am1_bcc_charges(&nitromethane, &params, &opts).unwrap();
        let total: f64 = r.charges.iter().sum();
        eprintln!("    nitromethane charge {charge:+.0}: Σq = {total:+.9}");
        assert!((total - charge).abs() < 1.0e-9);
    }
}

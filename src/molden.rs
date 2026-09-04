// SPDX-License-Identifier: GPL-3.0-or-later

//! Wavefunction output in **Molden** format.
//!
//! # What is written
//!
//! `[Atoms]`, `[STO]` and `[MO]`. The AM1 valence basis really is Slater-type — one `s` shell for
//! H/He and an `s` + `p` set for everything heavier, with the exponents `ζ_s`, `ζ_p` and the
//! principal quantum number `n` coming straight from the parameter table — so `[STO]` represents
//! it exactly and no Gaussian expansion has to be invented.
//!
//! Molden's `[STO]` line is `atom kx ky kz kr alfa norm`, describing the primitive
//!
//! ```text
//! norm · x^kx y^ky z^kz r^kr e^{−alfa·r}
//! ```
//!
//! which maps onto a Slater orbital with no residue:
//!
//! ```text
//! n s   →  kx=ky=kz=0, kr=n−1        (r^{n−1} e^{−ζr})
//! n p_i →  k_i=1, others 0, kr=n−2   (x r^{n−2} e^{−ζr} = r^{n−1} e^{−ζr} · x/r)
//! ```
//!
//! # The caveat that matters, stated plainly
//!
//! **NDDO *assumes* an orthonormal AO basis.** Its working equations are `F C = C ε` with no
//! overlap matrix, so the coefficients in `[MO]` are expressed in an implicitly orthogonalized
//! (Löwdin-like) basis, while `[STO]` describes the *un*-orthogonalized Slater functions a viewer
//! will actually draw. The two differ by `S^{−1/2}`, and `S` is not the identity for real Slater
//! functions at bonding distances.
//!
//! So the rendered orbitals are a faithful picture of an approximation, not of an exact
//! wavefunction: shapes, nodal structure and symmetry are right, and detailed amplitudes in the
//! bonding region are not. This is the same compromise MOPAC's own Molden output makes, and it is
//! inherent to writing an NDDO wavefunction in a format that presumes a real basis. It is written
//! into the file as a comment as well as here, so it travels with the data.
//!
//! # Units
//!
//! Ångström throughout — positions on `[Atoms] (Angs)` and `alfa`/`norm` in Å⁻¹ and Å^{−3/2} —
//! because Molden documents the `[STO]` section as being in Ångström and a file that mixed the
//! two would be silently wrong rather than rejected. Orbital energies are in Hartree, converted
//! from the crate's eV with **its own** `27.21`, not CODATA's; see [`crate::constants`].

use std::fmt::Write as _;

use crate::basis::Basis;
use crate::constants::{BOHR_TO_ANGSTROM, EV_TO_HARTREE};
use crate::error::Result;
use crate::linalg::Matrix;
use crate::params::Am1Parameters;
use crate::scf::Am1Result;
use crate::system::{z_to_symbol, Molecule};

/// `(2ζ)^{n+1/2} · sqrt(c / (4π (2n)!))` — the factor that normalizes a Slater primitive written
/// in Molden's `r^{kr} e^{−ζr}` form, with `c = 1` for an `s` function and `3` for a `p`.
///
/// Derived rather than tabulated, so it stays correct for any `n` the parameter set uses, and
/// checked against a numerical radial integral in the tests below rather than trusted.
fn slater_norm(n: u8, zeta: f64, angular: u32) -> f64 {
    let n = n as u32;
    let factorial = |m: u32| -> f64 { (1..=m).map(|k| k as f64).product::<f64>().max(1.0) };
    let c = if angular == 0 { 1.0 } else { 3.0 };
    (2.0 * zeta).powf(n as f64 + 0.5) * (c / (4.0 * std::f64::consts::PI * factorial(2 * n))).sqrt()
}

/// Render a converged SCF result as a Molden-format string.
///
/// Both spin channels are written for an unrestricted result; a restricted one gets a single
/// block with occupation 2.
pub fn to_molden(molecule: &Molecule, params: &Am1Parameters, scf: &Am1Result) -> Result<String> {
    let basis = Basis::build(molecule, params)?;
    let mut out = String::with_capacity(1024 + 24 * basis.nao * basis.nao);

    out.push_str("[Molden Format]\n");
    out.push_str("[Title]\n");
    writeln!(
        out,
        " am1-rs {} wavefunction ({} parameterization)",
        env!("CARGO_PKG_VERSION"),
        params.method.display_name()
    )
    .ok();
    // The caveat travels with the file, not only with the documentation. Written without square
    // brackets on purpose: a bracketed keyword inside the title block is exactly what a parser
    // scanning for section headers would trip over.
    out.push_str(
        " NOTE: NDDO assumes an orthonormal AO basis, so these MO coefficients are in an\n\
         \x20implicitly orthogonalized basis while the Slater functions listed below are the raw,\n\
         \x20non-orthogonal ones. Orbital shapes, nodes and symmetry are faithful; amplitudes in\n\
         \x20the bonding region are approximate. See the am1-rs `molden` module documentation.\n",
    );

    // ---- geometry ----
    out.push_str("[Atoms] Angs\n");
    for (i, atom) in molecule.atoms.iter().enumerate() {
        let p = atom.position * BOHR_TO_ANGSTROM;
        writeln!(
            out,
            " {:<2} {:5} {:5} {:18.10} {:18.10} {:18.10}",
            z_to_symbol(atom.z).unwrap_or("X"),
            i + 1,
            atom.z,
            p.x,
            p.y,
            p.z
        )
        .ok();
    }

    // ---- basis ----
    out.push_str("[STO]\n");
    for ao in &basis.aos {
        let elem = params.element(ao.z)?;
        // ζ is Bohr⁻¹ inside the crate; the section is documented as Ångström, so it and the
        // normalization both move into Å before being written.
        let zeta_per_angstrom = if ao.orb == 0 {
            elem.zeta_s
        } else {
            elem.zeta_p
        } / BOHR_TO_ANGSTROM;
        let (kx, ky, kz, kr) = match ao.orb {
            0 => (0, 0, 0, elem.n as i32 - 1),
            1 => (1, 0, 0, elem.n as i32 - 2),
            2 => (0, 1, 0, elem.n as i32 - 2),
            _ => (0, 0, 1, elem.n as i32 - 2),
        };
        let angular = u32::from(ao.orb != 0);
        writeln!(
            out,
            " {:5} {:3} {:3} {:3} {:3} {:18.10} {:18.10}",
            ao.atom + 1,
            kx,
            ky,
            kz,
            kr,
            zeta_per_angstrom,
            slater_norm(elem.n, zeta_per_angstrom, angular)
        )
        .ok();
    }

    // ---- orbitals ----
    out.push_str("[MO]\n");
    match &scf.beta {
        None => write_channel(
            &mut out,
            &scf.mo_coeff,
            &scf.mo_energies,
            scf.n_occ,
            "Alpha",
            2.0,
        ),
        Some(b) => {
            write_channel(
                &mut out,
                &scf.mo_coeff,
                &scf.mo_energies,
                scf.n_occ,
                "Alpha",
                1.0,
            );
            write_channel(&mut out, &b.coeff, &b.energies, b.n_occ, "Beta", 1.0);
        }
    }
    Ok(out)
}

/// Write one spin channel's orbitals. `occupation` is the count for a *filled* orbital.
fn write_channel(
    out: &mut String,
    coeff: &Matrix,
    energies: &[f64],
    n_occ: usize,
    spin: &str,
    occupation: f64,
) {
    let nao = coeff.rows;
    for k in 0..coeff.cols {
        // No symmetry perception here, so every orbital is labelled `a`. A label is required by
        // the format; inventing an irreducible representation would be worse than declining to.
        writeln!(out, " Sym= {}a", k + 1).ok();
        writeln!(out, " Ene= {:18.10}", energies[k] * EV_TO_HARTREE).ok();
        writeln!(out, " Spin= {spin}").ok();
        writeln!(
            out,
            " Occup= {:12.6}",
            if k < n_occ { occupation } else { 0.0 }
        )
        .ok();
        for mu in 0..nao {
            writeln!(out, " {:5} {:18.10}", mu + 1, coeff[(mu, k)]).ok();
        }
    }
}

/// Run the SCF and write a Molden file to `path`.
pub fn write_molden(
    path: impl AsRef<std::path::Path>,
    molecule: &Molecule,
    params: &Am1Parameters,
    scf: &Am1Result,
) -> Result<()> {
    std::fs::write(path, to_molden(molecule, params, scf)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scf::{run_am1, Am1Options, ScfReference};

    fn water() -> Molecule {
        Molecule::from_xyz_str(
            "3\nwater\nO 0.0000 0.0000 0.0000\nH 0.9584 0.0000 0.0000\nH -0.2400 0.9278 0.0000\n",
            0.0,
        )
        .unwrap()
    }

    /// The `norm` field must actually normalize the primitive it is attached to. Checked by
    /// integrating `|χ|²` on a radial grid rather than by re-deriving the same closed form, so
    /// an algebra slip in `slater_norm` cannot agree with itself.
    #[test]
    fn the_stated_normalization_integrates_to_one() {
        // ∫|N r^{kr} e^{−ζr} · (angular)|² dV, with the angular part integrating to 1 over the
        // sphere by construction, reduces to 4π∫ r^{2kr+2}|N'|²e^{−2ζr}dr for s, and to the same
        // with the 3cos²θ weight for p — both handled by folding the angular factor into `c`.
        for (n, zeta, angular) in [(1u8, 1.3, 0u32), (2, 2.7, 0), (2, 2.0, 1), (3, 1.8, 1)] {
            let norm = slater_norm(n, zeta, angular);
            let kr = if angular == 0 {
                n as i32 - 1
            } else {
                n as i32 - 2
            };
            // Radial integral of |R(r)|² r² dr with the angular part already normalized:
            // for p, χ = norm·x·r^{kr}e^{−ζr}, and ⟨x²⟩ over the sphere is r²/3.
            let steps = 400_000;
            let rmax = 60.0 / zeta;
            let h = rmax / steps as f64;
            let mut acc = 0.0;
            for i in 0..=steps {
                let r = i as f64 * h;
                let radial = norm * r.powi(kr) * (-zeta * r).exp();
                // 4π r² dr, with an extra r²/3 for p (from x²) — i.e. the same `c = 3` folded back.
                let weight = if angular == 0 {
                    4.0 * std::f64::consts::PI * r * r
                } else {
                    4.0 * std::f64::consts::PI * r * r * r * r / 3.0
                };
                let f = radial * radial * weight;
                let w = if i == 0 || i == steps { 0.5 } else { 1.0 };
                acc += w * f * h;
            }
            eprintln!("    n={n} zeta={zeta} l={angular}: ∫|χ|² = {acc:.10}");
            assert!(
                (acc - 1.0).abs() < 1.0e-6,
                "n={n} zeta={zeta} l={angular} integrates to {acc}"
            );
        }
    }

    #[test]
    fn a_restricted_file_has_the_expected_sections_and_counts() {
        let mol = water();
        let params = Am1Parameters::standard().unwrap();
        let scf = run_am1(&mol, &params, &Am1Options::default()).unwrap();
        let text = to_molden(&mol, &params, &scf).unwrap();

        assert!(text.starts_with("[Molden Format]"));
        for section in ["[Atoms] Angs", "[STO]", "[MO]"] {
            assert!(text.contains(section), "missing {section}");
        }
        // Water: O contributes 4 AOs and each H one, so 6 STO lines and 6 orbitals of 6
        // coefficients.
        let basis = Basis::build(&mol, &params).unwrap();
        assert_eq!(basis.nao, 6);
        // Anchored on whole lines, not on a substring: a section keyword mentioned in prose would
        // otherwise be indistinguishable from the header, which is a mistake a real parser can
        // make too.
        let lines: Vec<&str> = text.lines().collect();
        let sto_start = lines.iter().position(|l| l.trim() == "[STO]").unwrap();
        let mo_start = lines.iter().position(|l| l.trim() == "[MO]").unwrap();
        assert_eq!(mo_start - sto_start - 1, basis.nao, "one [STO] line per AO");
        assert!(
            !lines[..sto_start].iter().any(|l| l.trim() == "[MO]"),
            "no stray section header before the basis"
        );
        assert_eq!(text.matches("Sym=").count(), basis.nao);
        assert_eq!(text.matches("Spin= Alpha").count(), basis.nao);
        // Four doubly-occupied orbitals in water's minimal valence basis.
        assert_eq!(text.matches("Occup=     2.000000").count(), scf.n_occ);
    }

    /// The energies and occupations in the file have to be the ones the SCF reports, and the
    /// coefficient block has to be the orbital matrix — a transposition here would still produce
    /// a syntactically valid file.
    #[test]
    fn the_file_carries_the_scf_orbitals_unchanged() {
        let mol = water();
        let params = Am1Parameters::standard().unwrap();
        let scf = run_am1(&mol, &params, &Am1Options::default()).unwrap();
        let text = to_molden(&mol, &params, &scf).unwrap();

        let energies: Vec<f64> = text
            .lines()
            .filter_map(|l| l.trim().strip_prefix("Ene= "))
            .map(|v| v.trim().parse::<f64>().unwrap())
            .collect();
        assert_eq!(energies.len(), scf.mo_energies.len());
        for (written, &native) in energies.iter().zip(&scf.mo_energies) {
            assert!(
                (written - native * EV_TO_HARTREE).abs() < 1.0e-9,
                "orbital energy {written} != {}",
                native * EV_TO_HARTREE
            );
        }

        // The first orbital's coefficients, in order, must be column 0 of `mo_coeff`.
        let first: Vec<f64> = text
            .lines()
            .skip_while(|l| !l.trim().starts_with("Occup="))
            .skip(1)
            .take(scf.mo_coeff.rows)
            .map(|l| l.split_whitespace().nth(1).unwrap().parse::<f64>().unwrap())
            .collect();
        for (mu, v) in first.iter().enumerate() {
            assert!((v - scf.mo_coeff[(mu, 0)]).abs() < 1.0e-9);
        }
    }

    /// An unrestricted result must write both channels, at one electron each.
    #[test]
    fn an_unrestricted_file_writes_both_spin_channels() {
        let mol = Molecule::from_xyz_str(
            "4\nmethyl\nC 0.0 0.0 0.0\nH 1.079 0.0 0.0\nH -0.5395 0.9344 0.0\nH -0.5395 -0.9344 0.0\n",
            0.0,
        )
        .unwrap();
        let params = Am1Parameters::standard().unwrap();
        let opts = Am1Options {
            multiplicity: 2,
            reference: ScfReference::Unrestricted,
            ..Am1Options::default()
        };
        let scf = run_am1(&mol, &params, &opts).unwrap();
        let text = to_molden(&mol, &params, &scf).unwrap();
        let basis = Basis::build(&mol, &params).unwrap();
        assert_eq!(text.matches("Spin= Alpha").count(), basis.nao);
        assert_eq!(
            text.matches("Spin= Beta").count(),
            basis.nao,
            "the beta channel is missing; before 0.2.1 it was discarded by the SCF"
        );
        assert!(text.contains("Occup=     1.000000"));
    }
}

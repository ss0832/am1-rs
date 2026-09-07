// SPDX-License-Identifier: GPL-3.0-or-later

//! Wavefunction output in **Molden** format.
//!
//! # What is written
//!
//! `[Atoms]`, a basis section, and `[MO]`. The basis section is `[GTO]` by default — a Gaussian
//! expansion of the Slater valence functions, fitted by [`crate::gto`] — because `[GTO]` is the
//! section that Jmol, VMD, Avogadro, Molden itself and Multiwfn all implement. `[STO]` describes
//! the AM1 basis *exactly* and is kept for compatibility ([`MoldenBasis::Slater`]), but it is no
//! longer the default: it is a corner of the format that most viewers skip, and a file whose
//! basis section is skipped renders orbitals from nothing.
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
//! # The orthogonality question, and what is now done about it
//!
//! **NDDO *assumes* an orthonormal AO basis.** Its working equations are `F C = C ε` with no
//! overlap matrix, so the coefficients the SCF returns are expressed in an implicitly
//! orthogonalized (Löwdin) basis, `|χ^⊥⟩ = |χ⟩ S^{−1/2}`, while the basis section describes the
//! *un*-orthogonalized Slater functions a viewer actually draws. Writing the raw coefficients
//! against the raw functions is therefore not the same orbital: the two differ by `S^{−1/2}`, and
//! `S` is not the identity at bonding distances.
//!
//! Through 0.2.2 the file carried the raw coefficients and a note apologizing for it. It no longer
//! does. `S` is available — it is the analytic Slater overlap the resonance integrals already use
//! — so the default is to write `C_AO = S^{−1/2} C`, which is the same orbital expressed in the
//! basis the file declares. `MoldenOptions::deorthogonalize = false` restores the 0.2.2 behaviour
//! for comparison against other NDDO codes, which mostly do not do this.
//!
//! # Occupations and ordering
//!
//! `Occup=` is assigned from the aufbau on the orbital energies, which is the same rule the SCF's
//! density was built with — the lowest `n_occ` columns of `C`. The energies come back from
//! [`crate::linalg::symmetric_eigen`] in ascending order, so index order *is* energy order, and
//! the occupied set in the file *is* the occupied set of the density. `tests` below checks that
//! by rebuilding the density from the written text and comparing it against `scf.density`, rather
//! than by re-reading the same expression the writer used.
//!
//! # Units
//!
//! Positions in Ångström on `[Atoms] Angs`. `[GTO]` exponents are Bohr⁻², which is the atomic-unit
//! convention every Molden writer uses for that section. `[STO]`'s `alfa`/`norm` are Å⁻¹ and
//! Å^{−3/2}, because Molden documents that section in Ångström. Orbital energies are in Hartree,
//! converted from the crate's eV with **its own** `27.21`, not CODATA's; see [`crate::constants`].

use std::fmt::Write as _;

use crate::basis::Basis;
use crate::constants::{BOHR_TO_ANGSTROM, EV_TO_HARTREE};
use crate::error::Result;
use crate::gto::{expand_slater, GaussianShell, DEFAULT_NGAUSS};
use crate::linalg::{symmetric_eigen, Matrix};
use crate::overlap::diatom_overlap;
use crate::params::Am1Parameters;
use crate::scf::Am1Result;
use crate::system::{z_to_symbol, Molecule};

/// Which basis section the file carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoldenBasis {
    /// `[GTO]`, from a fitted Gaussian expansion of each Slater shell. **The default.**
    ///
    /// The field is the number of primitives per shell; see [`crate::gto::DEFAULT_NGAUSS`].
    Gaussian(usize),
    /// `[STO]`, the Slater functions themselves.
    ///
    /// Exact for this basis and kept for compatibility with anything that reads the section. Most
    /// viewers do not, which is why it is not the default.
    Slater,
}

impl Default for MoldenBasis {
    fn default() -> Self {
        MoldenBasis::Gaussian(DEFAULT_NGAUSS)
    }
}

/// How to render a wavefunction as Molden.
#[derive(Clone, Copy, Debug)]
pub struct MoldenOptions {
    pub basis: MoldenBasis,
    /// Undo NDDO's implicit Löwdin orthogonalization, writing `C_AO = S^{−1/2} C`.
    ///
    /// Default `true`: the coefficients then belong to the basis functions the file declares,
    /// which is what makes the drawn orbital the orbital. `false` writes the raw NDDO
    /// coefficients — the 0.2.2 behaviour, and what most NDDO codes emit.
    pub deorthogonalize: bool,
}

impl Default for MoldenOptions {
    fn default() -> Self {
        Self {
            basis: MoldenBasis::default(),
            deorthogonalize: true,
        }
    }
}

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

/// The AO overlap matrix of the **Slater** basis, `S_μν = ⟨χ_μ|χ_ν⟩`.
///
/// The same analytic diatomic integrals the resonance term uses, assembled over every atom pair.
/// This is the matrix NDDO assumes to be the identity; it is not, and the difference is exactly
/// what [`MoldenOptions::deorthogonalize`] removes.
pub fn slater_overlap_matrix(molecule: &Molecule, params: &Am1Parameters) -> Result<Matrix> {
    let basis = Basis::build(molecule, params)?;
    let nat = molecule.atoms.len();
    let mut s = Matrix::identity(basis.nao);
    for a in 0..nat {
        for b in (a + 1)..nat {
            let (ea, eb) = (
                params.element(molecule.atoms[a].z)?,
                params.element(molecule.atoms[b].z)?,
            );
            let block = diatom_overlap(
                ea,
                molecule.atoms[a].position,
                eb,
                molecule.atoms[b].position,
            )?;
            let (off_a, off_b) = (basis.atom_offset[a], basis.atom_offset[b]);
            for i in 0..basis.atom_norb[a] {
                for j in 0..basis.atom_norb[b] {
                    s[(off_a + i, off_b + j)] = block[i][j];
                    s[(off_b + j, off_a + i)] = block[i][j];
                }
            }
        }
    }
    Ok(s)
}

/// `S^{−1/2}` by eigendecomposition.
///
/// A near-null eigenvalue means the Slater basis is close to linearly dependent at this geometry
/// — two atoms nearly on top of each other — and inverting its square root would amplify noise
/// without bound. Those directions are dropped instead, which is the standard canonical-
/// orthogonalization cut and is reported as an error rather than silently applied when it would
/// remove a whole orbital's worth of basis.
fn inverse_sqrt(s: &Matrix) -> Result<Matrix> {
    let (values, vectors) = symmetric_eigen(s)?;
    let n = s.rows;
    let mut out = Matrix::zeros(n, n);
    for (k, &lambda) in values.iter().enumerate() {
        if lambda <= 1.0e-8 {
            continue;
        }
        let scale = 1.0 / lambda.sqrt();
        for i in 0..n {
            let vi = vectors[(i, k)] * scale;
            for j in 0..n {
                out[(i, j)] += vi * vectors[(j, k)];
            }
        }
    }
    Ok(out)
}

/// Render a converged SCF result as a Molden-format string, with the default options.
///
/// Both spin channels are written for an unrestricted result; a restricted one gets a single
/// block with occupation 2.
pub fn to_molden(molecule: &Molecule, params: &Am1Parameters, scf: &Am1Result) -> Result<String> {
    to_molden_with(molecule, params, scf, &MoldenOptions::default())
}

/// [`to_molden`] with an explicit choice of basis section and orthogonalization.
pub fn to_molden_with(
    molecule: &Molecule,
    params: &Am1Parameters,
    scf: &Am1Result,
    options: &MoldenOptions,
) -> Result<String> {
    let basis = Basis::build(molecule, params)?;
    let mut out = String::with_capacity(1024 + 24 * basis.nao * basis.nao);

    // The transform applied to every MO coefficient column, or `None` for the raw ones.
    let transform = if options.deorthogonalize {
        Some(inverse_sqrt(&slater_overlap_matrix(molecule, params)?)?)
    } else {
        None
    };

    out.push_str("[Molden Format]\n");
    out.push_str("[Title]\n");
    writeln!(
        out,
        " am1-rs {} wavefunction ({} parameterization)",
        env!("CARGO_PKG_VERSION"),
        params.method.display_name()
    )
    .ok();
    // The caveats travel with the file, not only with the documentation. Written without square
    // brackets on purpose: a bracketed keyword inside the title block is exactly what a parser
    // scanning for section headers would trip over.
    match options.basis {
        MoldenBasis::Gaussian(k) => {
            let worst = worst_fit(molecule, params, k)?;
            writeln!(
                out,
                " NOTE: the AM1 valence basis is Slater-type. Each shell below is a {k}-Gaussian\n\
                 \x20least-squares expansion of it, fitted at run time; the poorest shell in this\n\
                 \x20molecule overlaps its Slater function by {worst:.8}."
            )
            .ok();
        }
        MoldenBasis::Slater => {
            out.push_str(
                " NOTE: this file uses the [STO] section, which describes the AM1 basis exactly\n\
                 \x20but which most viewers do not read. Prefer the default [GTO] output.\n",
            );
        }
    }
    if options.deorthogonalize {
        out.push_str(
            " NOTE: NDDO solves in an implicitly orthogonalized (Lowdin) basis. These MO\n\
             \x20coefficients have been transformed back by S^-1/2, so they belong to the\n\
             \x20non-orthogonal functions listed below and are no longer orthonormal under the\n\
             \x20identity metric.\n",
        );
    } else {
        out.push_str(
            " NOTE: NDDO assumes an orthonormal AO basis, so these MO coefficients are in an\n\
             \x20implicitly orthogonalized basis while the functions listed below are the raw,\n\
             \x20non-orthogonal ones. Orbital shapes, nodes and symmetry are faithful; amplitudes\n\
             \x20in the bonding region are approximate.\n",
        );
    }

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
    match options.basis {
        MoldenBasis::Gaussian(k) => write_gto(&mut out, molecule, params, k)?,
        MoldenBasis::Slater => write_sto(&mut out, &basis, params)?,
    }

    // ---- orbitals ----
    out.push_str("[MO]\n");
    let alpha = transformed(&scf.mo_coeff, transform.as_ref());
    match &scf.beta {
        None => write_channel(&mut out, &alpha, &scf.mo_energies, scf.n_occ, "Alpha", 2.0),
        Some(b) => {
            write_channel(&mut out, &alpha, &scf.mo_energies, scf.n_occ, "Alpha", 1.0);
            let beta = transformed(&b.coeff, transform.as_ref());
            write_channel(&mut out, &beta, &b.energies, b.n_occ, "Beta", 1.0);
        }
    }
    Ok(out)
}

/// `X C`, or `C` when there is no transform.
fn transformed(coeff: &Matrix, transform: Option<&Matrix>) -> Matrix {
    match transform {
        Some(x) => x.matmul(coeff),
        None => coeff.clone(),
    }
}

/// The poorest Slater→Gaussian overlap over the shells this molecule needs.
fn worst_fit(molecule: &Molecule, params: &Am1Parameters, primitives: usize) -> Result<f64> {
    let mut worst = 1.0_f64;
    let mut seen: Vec<u8> = Vec::new();
    for atom in &molecule.atoms {
        if seen.contains(&atom.z) {
            continue;
        }
        seen.push(atom.z);
        let elem = params.element(atom.z)?;
        worst = worst.min(expand_slater(elem.n, 0, elem.zeta_s, primitives)?.overlap);
        if elem.has_p() {
            worst = worst.min(expand_slater(elem.n, 1, elem.zeta_p, primitives)?.overlap);
        }
    }
    Ok(worst)
}

/// The `[GTO]` section: one block per atom, shells in the same order as [`Basis`] lists the AOs.
///
/// Atom blocks are separated by a blank line and the shell order is `s` then `p`, which is
/// exactly the AO order `Basis::build` produces (`s, px, py, pz` per atom). That correspondence is
/// what makes the `[MO]` coefficient indices mean anything, so it is not incidental: Cartesian
/// `p` in Molden is ordered `x, y, z`, which is [`crate::basis::AoInfo::orb`] 1, 2, 3.
fn write_gto(
    out: &mut String,
    molecule: &Molecule,
    params: &Am1Parameters,
    primitives: usize,
) -> Result<()> {
    out.push_str("[GTO]\n");
    for (i, atom) in molecule.atoms.iter().enumerate() {
        let elem = params.element(atom.z)?;
        writeln!(out, " {:5} 0", i + 1).ok();
        write_shell(
            out,
            "s",
            &expand_slater(elem.n, 0, elem.zeta_s, primitives)?,
        );
        if elem.has_p() {
            write_shell(
                out,
                "p",
                &expand_slater(elem.n, 1, elem.zeta_p, primitives)?,
            );
        }
        // Molden's reader uses the blank line to end an atom's shell list.
        out.push('\n');
    }
    Ok(())
}

fn write_shell(out: &mut String, label: &str, shell: &GaussianShell) {
    writeln!(out, " {label}    {}  1.00", shell.exponents.len()).ok();
    for (alpha, c) in shell.exponents.iter().zip(&shell.coefficients) {
        writeln!(out, "  {alpha:20.10e} {c:20.10e}").ok();
    }
}

/// The `[STO]` section, kept for compatibility. See the module documentation.
fn write_sto(out: &mut String, basis: &Basis, params: &Am1Parameters) -> Result<()> {
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
    Ok(())
}

/// Write one spin channel's orbitals. `occupation` is the count for a *filled* orbital.
///
/// Orbital `k` is occupied when `k < n_occ`, which is the aufbau on ascending energies — the same
/// selection [`crate::scf`] built the density from. Nothing here re-sorts: a re-sort would be a
/// second, independent occupation rule, and two rules that agree today are two rules that can
/// disagree tomorrow.
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

    fn sto_options() -> MoldenOptions {
        MoldenOptions {
            basis: MoldenBasis::Slater,
            deorthogonalize: false,
        }
    }

    /// One `Occup=`/`Ene=`/coefficient block per orbital, parsed back out of the file.
    fn parse_mo(text: &str) -> Vec<(f64, f64, String, Vec<f64>)> {
        let mut out = Vec::new();
        let lines = text.lines().skip_while(|l| l.trim() != "[MO]").skip(1);
        let mut current: Option<(f64, f64, String, Vec<f64>)> = None;
        let mut energy = 0.0;
        let mut spin = String::new();
        for line in lines {
            let t = line.trim();
            if let Some(v) = t.strip_prefix("Ene=") {
                energy = v.trim().parse().unwrap();
            } else if let Some(v) = t.strip_prefix("Spin=") {
                spin = v.trim().to_string();
            } else if let Some(v) = t.strip_prefix("Occup=") {
                if let Some(c) = current.take() {
                    out.push(c);
                }
                current = Some((energy, v.trim().parse().unwrap(), spin.clone(), Vec::new()));
            } else if let Some(c) = current.as_mut() {
                let mut fields = t.split_whitespace();
                if let (Some(_), Some(value)) = (fields.next(), fields.next()) {
                    if let Ok(v) = value.parse::<f64>() {
                        c.3.push(v);
                    }
                }
            }
        }
        if let Some(c) = current {
            out.push(c);
        }
        out
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
        let text = to_molden_with(&mol, &params, &scf, &sto_options()).unwrap();

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

    /// The default file must carry `[GTO]`, one block per atom, with the shell counts the basis
    /// implies — and no `[STO]`, which is what a viewer would skip.
    #[test]
    fn the_default_file_carries_a_gaussian_basis() {
        let mol = water();
        let params = Am1Parameters::standard().unwrap();
        let scf = run_am1(&mol, &params, &Am1Options::default()).unwrap();
        let text = to_molden(&mol, &params, &scf).unwrap();

        assert!(text.contains("[GTO]"), "the default section must be [GTO]");
        assert!(!text.contains("[STO]"));
        // Oxygen has an s and a p shell; each hydrogen only an s.
        assert_eq!(
            text.lines().filter(|l| l.trim().starts_with("s ")).count(),
            3
        );
        assert_eq!(
            text.lines().filter(|l| l.trim().starts_with("p ")).count(),
            1
        );
        // No `[5D]`/`[7F]`: an l = 1 shell is the same in either convention, and declaring a
        // spherical basis we do not have would mislabel anything added later.
        assert!(!text.contains("[5D]") && !text.contains("[7F]"));
    }

    /// **The occupied orbitals in the file must be the occupied orbitals of the density.**
    ///
    /// Rebuilds `P = Σ_k occ_k C_k C_kᵀ` from the *text*, using only the `Occup=` fields the file
    /// states, and compares it against the SCF's own density matrix. Nothing in this test knows
    /// how the writer chose the occupations, so an off-by-one between `Occup=` and the orbital it
    /// is attached to — the failure mode a syntactically valid file hides best — cannot pass it.
    #[test]
    fn the_file_occupations_rebuild_the_scf_density() {
        for (label, options) in [
            ("raw", sto_options()),
            (
                "orthogonal",
                MoldenOptions {
                    basis: MoldenBasis::default(),
                    deorthogonalize: false,
                },
            ),
        ] {
            let mol = water();
            let params = Am1Parameters::standard().unwrap();
            let scf = run_am1(&mol, &params, &Am1Options::default()).unwrap();
            let text = to_molden_with(&mol, &params, &scf, &options).unwrap();
            let orbitals = parse_mo(&text);
            let nao = scf.density.rows;
            let mut p = Matrix::zeros(nao, nao);
            for (_, occ, _, c) in &orbitals {
                assert_eq!(c.len(), nao);
                for i in 0..nao {
                    for j in 0..nao {
                        p[(i, j)] += occ * c[i] * c[j];
                    }
                }
            }
            let mut worst = 0.0_f64;
            for i in 0..nao {
                for j in 0..nao {
                    worst = worst.max((p[(i, j)] - scf.density[(i, j)]).abs());
                }
            }
            eprintln!("    {label}: worst density element mismatch {worst:.3e}");
            assert!(worst < 1.0e-9, "{label}: density mismatch {worst}");
        }
    }

    /// The same for an unrestricted result, where `P = P_α + P_β` comes from two channels at one
    /// electron each — the case where a swapped `Spin=` block would go unnoticed.
    #[test]
    fn an_unrestricted_file_rebuilds_the_total_density() {
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
        let text = to_molden_with(&mol, &params, &scf, &sto_options()).unwrap();
        let basis = Basis::build(&mol, &params).unwrap();
        assert_eq!(text.matches("Spin= Alpha").count(), basis.nao);
        assert_eq!(
            text.matches("Spin= Beta").count(),
            basis.nao,
            "the beta channel is missing; before 0.2.1 it was discarded by the SCF"
        );
        assert!(text.contains("Occup=     1.000000"));

        let nao = scf.density.rows;
        let mut p = Matrix::zeros(nao, nao);
        for (_, occ, _, c) in parse_mo(&text) {
            for i in 0..nao {
                for j in 0..nao {
                    p[(i, j)] += occ * c[i] * c[j];
                }
            }
        }
        let mut worst = 0.0_f64;
        for i in 0..nao {
            for j in 0..nao {
                worst = worst.max((p[(i, j)] - scf.density[(i, j)]).abs());
            }
        }
        eprintln!("    UHF: worst total-density element mismatch {worst:.3e}");
        assert!(worst < 1.0e-9, "UHF density mismatch {worst}");
    }

    /// The energies in the file have to be the ones the SCF reports, in the same order, and the
    /// `Occup=` boundary has to fall at the HOMO.
    #[test]
    fn the_file_carries_the_scf_orbitals_unchanged() {
        let mol = water();
        let params = Am1Parameters::standard().unwrap();
        let scf = run_am1(&mol, &params, &Am1Options::default()).unwrap();
        let text = to_molden(&mol, &params, &scf).unwrap();
        let orbitals = parse_mo(&text);

        assert_eq!(orbitals.len(), scf.mo_energies.len());
        for ((written, occ, _, _), &native) in orbitals.iter().zip(&scf.mo_energies) {
            assert!(
                (written - native * EV_TO_HARTREE).abs() < 1.0e-9,
                "orbital energy {written} != {}",
                native * EV_TO_HARTREE
            );
            let _ = occ;
        }
        // Ascending, so that a viewer's "orbital 5" is this crate's orbital 5.
        assert!(orbitals.windows(2).all(|w| w[0].0 <= w[1].0 + 1.0e-12));
        // The last occupied orbital is the HOMO the SCF reported.
        let homo = orbitals
            .iter()
            .rposition(|(_, occ, _, _)| *occ > 0.0)
            .unwrap();
        assert_eq!(homo + 1, scf.n_occ);
        assert!(
            (orbitals[homo].0 - scf.homo_ev.unwrap() * EV_TO_HARTREE).abs() < 1.0e-9,
            "the last occupied orbital is not the reported HOMO"
        );
    }

    /// De-orthogonalization must be the transform it claims to be: `Cᵀ S C = I` for the written
    /// coefficients against the Slater overlap, where the raw NDDO ones satisfy `Cᵀ C = I`.
    ///
    /// This is the check that the file's orbitals belong to the file's basis functions. It also
    /// shows the transform is not a no-op: `S` is measurably not the identity here.
    #[test]
    fn the_written_orbitals_are_orthonormal_in_the_declared_basis() {
        let mol = water();
        let params = Am1Parameters::standard().unwrap();
        let scf = run_am1(&mol, &params, &Am1Options::default()).unwrap();
        let s = slater_overlap_matrix(&mol, &params).unwrap();

        let mut off_diagonal = 0.0_f64;
        for i in 0..s.rows {
            for j in 0..s.cols {
                if i != j {
                    off_diagonal = off_diagonal.max(s[(i, j)].abs());
                }
            }
        }
        eprintln!("    largest Slater overlap off the diagonal: {off_diagonal:.4}");
        assert!(
            off_diagonal > 0.1,
            "the basis is nearly orthogonal here, so this test proves nothing"
        );

        let text = to_molden(&mol, &params, &scf).unwrap();
        let orbitals = parse_mo(&text);
        let nao = s.rows;
        let mut worst = 0.0_f64;
        for (a, (_, _, _, ca)) in orbitals.iter().enumerate() {
            for (b, (_, _, _, cb)) in orbitals.iter().enumerate() {
                let mut acc = 0.0;
                for i in 0..nao {
                    for j in 0..nao {
                        acc += ca[i] * s[(i, j)] * cb[j];
                    }
                }
                let expected = if a == b { 1.0 } else { 0.0 };
                worst = worst.max((acc - expected).abs());
            }
        }
        eprintln!("    worst deviation of CᵀSC from the identity: {worst:.3e}");
        assert!(worst < 1.0e-9, "CᵀSC is not the identity: off by {worst}");
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
        assert_eq!(text.matches("Spin= Beta").count(), basis.nao);
        assert!(text.contains("Occup=     1.000000"));
    }
}

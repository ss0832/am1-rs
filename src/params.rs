// SPDX-License-Identifier: GPL-3.0-or-later

//! AM1 element parameters and the derived NDDO multipole quantities.
//!
//! Parses the embedded MOPAC AM1 parameter table and, per element, precomputes the
//! dipole/quadrupole charge separations `dd`/`qq` and the Klopman–Ohno additive terms
//! `rho0/rho1/rho2` used by the two-center two-electron integrals. The closed forms and
//! the `rho1`/`rho2` secant solves follow MOPAC `calpar.f`/`ddpo.f` (verified against the
//! PySEQM reference implementation).

use crate::constants::AM1_EV;
use crate::data_tables::{self, AM1_PARAM_CSV, RM1_PARAM_CSV};
use crate::error::{Am1Error, Result};
use crate::method::NddoMethod;
use std::collections::HashMap;

/// Per-element AM1 parameters plus derived NDDO quantities.
#[derive(Clone, Debug)]
pub struct Am1Element {
    pub z: u8,
    pub n: u8, // valence principal quantum number
    pub u_ss: f64,
    pub u_pp: f64,
    pub zeta_s: f64,
    pub zeta_p: f64,
    pub beta_s: f64,
    pub beta_p: f64,
    pub g_ss: f64,
    pub g_sp: f64,
    pub g_pp: f64,
    pub g_p2: f64,
    pub h_sp: f64,
    pub alpha: f64,
    /// AM1 core-core Gaussian corrections `(K, L, M)` (only the nonzero triples).
    pub gauss: Vec<(f64, f64, f64)>,
    /// Number of valence AOs: 1 for H/He (s only), 4 for s,p elements.
    pub n_orb: usize,
    /// Core charge (valence-electron count).
    pub core_charge: f64,
    /// Experimental atomic heat of formation (eV).
    pub eheat_ev: f64,
    /// Isolated-atom electronic energy (eV).
    pub e_isol: f64,
    // Derived NDDO multipole terms (Bohr).
    pub dd: f64,
    pub qq: f64,
    pub rho0: f64,
    pub rho1: f64,
    pub rho2: f64,
}

impl Am1Element {
    pub fn has_p(&self) -> bool {
        self.n_orb >= 4
    }
}

#[derive(Clone, Debug, Default)]
pub struct Am1Parameters {
    pub elements: HashMap<u8, Am1Element>,
    /// Which parameterization these values belong to. AM1 and RM1 share a functional form, so
    /// this is carried for reporting and for the "which elements does this method cover"
    /// error message rather than to switch code paths.
    pub method: NddoMethod,
}

impl Am1Parameters {
    /// Load the standard embedded AM1 parameter set.
    pub fn standard() -> Result<Self> {
        Self::for_method(NddoMethod::Am1)
    }

    /// Load the embedded parameter set for `method`, from a per-method cache.
    ///
    /// # Why this is cached
    ///
    /// Building a set is not free: it parses a hundred-odd CSV rows and then, per element, runs
    /// the **secant solves** for `rho1` and `rho2` ([`derive_parameters`]). That is invisible
    /// against a large molecule and dominant against a small one — and every function on the
    /// Python surface calls this at its top, so a molecular-dynamics loop on a water molecule was
    /// paying for it on every step. It is a fixed per-call cost, which is exactly the kind a
    /// large-system profile cannot see.
    ///
    /// The returned set is a clone of the cached one. [`Self::shared`] avoids even that for a
    /// caller that only needs to read.
    pub fn for_method(method: NddoMethod) -> Result<Self> {
        Self::shared(method).cloned()
    }

    /// The cached parameter set for `method`, borrowed rather than cloned.
    ///
    /// The sets are immutable once built and live for the process, so a caller that only reads
    /// them — which is every caller inside the crate — can borrow. Returns the stored `Result` by
    /// reference so a malformed embedded table still surfaces as an error rather than a panic.
    pub fn shared(method: NddoMethod) -> Result<&'static Self> {
        use std::sync::OnceLock;
        static AM1: OnceLock<std::result::Result<Am1Parameters, String>> = OnceLock::new();
        static RM1: OnceLock<std::result::Result<Am1Parameters, String>> = OnceLock::new();

        let build = |text: &str| -> std::result::Result<Am1Parameters, String> {
            let mut params = Self::from_csv(text).map_err(|e| e.to_string())?;
            params.method = method;
            Ok(params)
        };
        let slot = match method {
            NddoMethod::Am1 => AM1.get_or_init(|| build(AM1_PARAM_CSV)),
            NddoMethod::Rm1 => RM1.get_or_init(|| build(RM1_PARAM_CSV)),
        };
        slot.as_ref().map_err(|e| Am1Error::InvalidInput(e.clone()))
    }

    pub fn element(&self, z: u8) -> Result<&Am1Element> {
        self.elements.get(&z).ok_or_else(|| {
            // Naming the method matters here: RM1 covers ten elements where AM1 covers
            // twenty-one, so "missing parameter block for Z=14" is a confusing thing to read
            // when silicon is perfectly well parameterized -- just not by RM1.
            Am1Error::ElementNotParameterized {
                method: self.method.display_name(),
                z,
                supported: self.supported_symbols().join(", "),
            }
        })
    }

    /// Element symbols this parameter set covers, in ascending atomic number.
    pub fn supported_symbols(&self) -> Vec<&'static str> {
        let mut zs: Vec<u8> = self.elements.keys().copied().collect();
        zs.sort_unstable();
        zs.iter()
            .filter_map(|&z| crate::system::z_to_symbol(z))
            .collect()
    }

    pub fn from_csv(text: &str) -> Result<Self> {
        // Skip leading provenance/comment lines (`#`); the first data line is the header.
        let mut lines = text.lines().filter(|l| !l.trim_start().starts_with('#'));
        let header = lines
            .next()
            .ok_or_else(|| Am1Error::InvalidInput("empty parameter CSV".to_string()))?;
        let cols: Vec<String> = header
            .split(',')
            .map(|s| s.trim().replace(' ', ""))
            .collect();
        let col = |name: &str| -> Result<usize> {
            cols.iter()
                .position(|c| c == name)
                .ok_or_else(|| Am1Error::MissingParameter(name.to_string()))
        };
        let ci_n = col("N")?;
        let ci = |n: &str| col(n);
        let (c_uss, c_upp) = (ci("U_ss")?, ci("U_pp")?);
        let (c_zs, c_zp) = (ci("zeta_s")?, ci("zeta_p")?);
        let (c_bs, c_bp) = (ci("beta_s")?, ci("beta_p")?);
        let (c_gss, c_gsp) = (ci("g_ss")?, ci("g_sp")?);
        let (c_gpp, c_gp2, c_hsp) = (ci("g_pp")?, ci("g_p2")?, ci("h_sp")?);
        let c_alpha = ci("alpha")?;
        let gcols = [
            (ci("Gaussian1_K")?, ci("Gaussian1_L")?, ci("Gaussian1_M")?),
            (ci("Gaussian2_K")?, ci("Gaussian2_L")?, ci("Gaussian2_M")?),
            (ci("Gaussian3_K")?, ci("Gaussian3_L")?, ci("Gaussian3_M")?),
            (ci("Gaussian4_K")?, ci("Gaussian4_L")?, ci("Gaussian4_M")?),
        ];

        let mut elements = HashMap::new();
        for (idx, line) in lines.enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let f: Vec<f64> = line
                .split(',')
                .map(|s| s.trim().parse::<f64>().unwrap_or(0.0))
                .collect();
            let z = f[ci_n] as u8;
            // Only keep elements that are actually parametrized (nonzero U_ss or beta_s).
            if f[c_uss] == 0.0 && f[c_bs] == 0.0 {
                continue;
            }
            let zeta_p = f[c_zp];
            let n_orb = if zeta_p > 0.0 { 4 } else { 1 };
            let mut gauss = Vec::new();
            for (kk, ll, mm) in gcols {
                let (k, l, m) = (f[kk], f[ll], f[mm]);
                if k != 0.0 || l != 0.0 {
                    gauss.push((k, l, m));
                }
            }

            let n = data_tables::QN[z as usize];
            let zeta_s = f[c_zs];
            let g_ss = f[c_gss];
            let g_pp = f[c_gpp];
            let g_p2 = f[c_gp2];
            let h_sp = f[c_hsp];

            let (dd, qq, rho1, rho2) = if n_orb == 4 {
                let dd = dd_charge_sep(n, zeta_s, zeta_p);
                let qq = qq_charge_sep(n, zeta_p);
                let hpp = (0.5 * (g_pp - g_p2)).max(0.1);
                let rho1 = additive_rho1(h_sp, dd);
                let rho2 = additive_rho2(hpp, qq);
                (dd, qq, rho1, rho2)
            } else {
                (0.0, 0.0, 0.0, 0.0)
            };
            let rho0 = 0.5 * AM1_EV / g_ss;

            let zi = z as usize;
            let e_isol = f[c_uss] * data_tables::N_S[zi]
                + f[c_upp] * data_tables::N_P[zi]
                + g_ss * data_tables::gssc(z)
                + g_pp * data_tables::gppc(z)
                + f[c_gsp] * data_tables::gspc(z)
                + g_p2 * data_tables::gp2c(z)
                + h_sp * data_tables::hspc(z);

            elements.insert(
                z,
                Am1Element {
                    z,
                    n,
                    u_ss: f[c_uss],
                    u_pp: f[c_upp],
                    zeta_s,
                    zeta_p,
                    beta_s: f[c_bs],
                    beta_p: f[c_bp],
                    g_ss,
                    g_sp: f[c_gsp],
                    g_pp,
                    g_p2,
                    h_sp,
                    alpha: f[c_alpha],
                    gauss,
                    n_orb,
                    core_charge: data_tables::core_charge(z),
                    eheat_ev: data_tables::EHEAT_KCAL[zi] / crate::constants::EV_TO_KCAL,
                    e_isol,
                    dd,
                    qq,
                    rho0,
                    rho1,
                    rho2,
                },
            );
            let _ = idx;
        }
        if elements.is_empty() {
            return Err(Am1Error::InvalidInput(
                "no AM1 elements parsed from parameter table".to_string(),
            ));
        }
        Ok(Self {
            elements,
            method: NddoMethod::default(),
        })
    }
}

/// Dipole charge separation `dd` (Bohr). MOPAC `ddpo.f`.
pub fn dd_charge_sep(n: u8, zs: f64, zp: f64) -> f64 {
    let nf = n as f64;
    (2.0 * nf + 1.0) * (4.0 * zs * zp).powf(nf + 0.5)
        / (zs + zp).powf(2.0 * nf + 2.0)
        / 3.0_f64.sqrt()
}

/// Quadrupole charge separation `qq` (Bohr). MOPAC `ddpo.f`.
pub fn qq_charge_sep(n: u8, zp: f64) -> f64 {
    let nf = n as f64;
    ((4.0 * nf * nf + 6.0 * nf + 2.0) / 20.0).sqrt() / zp
}

/// Additive term `rho1` reproducing the one-center dipole integral `H_sp` (Bohr).
///
/// Solves `H_sp(au) = ½ d − ½ / √(4 D1² + 1/d²)` for `d`, returning `rho1 = 0.5/d`
/// (secant iteration, MOPAC `calpar.f`).
pub fn additive_rho1(hsp_ev: f64, d1: f64) -> f64 {
    let hsp = hsp_ev / AM1_EV;
    let g = |d: f64| 0.5 * d - 0.5 / (4.0 * d1 * d1 + 1.0 / (d * d)).sqrt();
    let mut a = (hsp.abs() / (d1 * d1)).powf(1.0 / 3.0);
    if hsp < 0.0 {
        a = -a;
    }
    let mut b = a + 0.04;
    for _ in 0..30 {
        let (ga, gb) = (g(a), g(b));
        let c = if (gb - ga).abs() > 1.0e-16 {
            a + (b - a) * (hsp - ga) / (gb - ga)
        } else {
            b
        };
        a = b;
        b = c;
    }
    0.5 / b
}

/// Additive term `rho2` reproducing the one-center quadrupole integral `H_pp` (Bohr).
///
/// Solves `H_pp(au) = ¼ q − ½/√(4 D2² + 1/q²) + ¼/√(8 D2² + 1/q²)` for `q`, returning
/// `rho2 = 0.5/q` (secant iteration, MOPAC `calpar.f`).
pub fn additive_rho2(hpp_ev: f64, d2: f64) -> f64 {
    let hpp = hpp_ev / AM1_EV;
    let g = |q: f64| {
        0.25 * q - 0.5 / (4.0 * d2 * d2 + 1.0 / (q * q)).sqrt()
            + 0.25 / (8.0 * d2 * d2 + 1.0 / (q * q)).sqrt()
    };
    let mut a = (hpp.abs() / (3.0 * d2.powi(4))).powf(0.2);
    if hpp < 0.0 {
        a = -a;
    }
    let mut b = a + 0.04;
    for _ in 0..30 {
        let (ga, gb) = (g(a), g(b));
        let c = if (gb - ga).abs() > 1.0e-16 {
            a + (b - a) * (hpp - ga) / (gb - ga)
        } else {
            b
        };
        a = b;
        b = c;
    }
    0.5 / b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_core_elements() {
        let p = Am1Parameters::standard().unwrap();
        for z in [1u8, 6, 7, 8] {
            let e = p.element(z).unwrap();
            assert!(e.rho0 > 0.0, "rho0 must be positive for Z={z}");
        }
        // Carbon reference values (MOPAC AM1).
        let c = p.element(6).unwrap();
        assert!((c.u_ss - (-52.028658)).abs() < 1e-6);
        assert!((c.g_ss - 12.23).abs() < 1e-9);
        assert_eq!(c.gauss.len(), 4);
        assert!(c.dd > 0.0 && c.qq > 0.0 && c.rho1 > 0.0 && c.rho2 > 0.0);
        // Hydrogen has only an s shell.
        let h = p.element(1).unwrap();
        assert_eq!(h.n_orb, 1);
        assert_eq!(h.dd, 0.0);
    }
}

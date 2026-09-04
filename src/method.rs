// SPDX-License-Identifier: GPL-3.0-or-later

//! Which NDDO parameterization a calculation uses.
//!
//! AM1 and RM1 share their functional form exactly — the same NDDO core, the same
//! Dewar–Sabelli–Klopman two-centre integrals, the same core–core Gaussian corrections — and
//! differ only in the numbers, so selecting between them is purely a matter of which
//! parameter table is loaded. SAM1 is not like that: it replaces the parametric two-centre
//! two-electron integrals with scaled *ab initio* ones over an STO-3G Gaussian basis, so it
//! needs its own integral path as well as its own parameters.

use crate::error::{Am1Error, Result};

/// An NDDO parameterization.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum NddoMethod {
    /// Austin Model 1 — Dewar, Zoebisch, Healy & Stewart, *JACS* **107**, 3902 (1985).
    #[default]
    Am1,
    /// Recife Model 1 — Rocha, Freire, Simas & Stewart, *J. Comput. Chem.* **27**, 1101
    /// (2006). A reparameterization of AM1; identical functional form.
    Rm1,
}

impl NddoMethod {
    /// Lower-case identifier, as accepted by [`NddoMethod::parse`] and printed by the CLI.
    pub fn name(self) -> &'static str {
        match self {
            Self::Am1 => "am1",
            Self::Rm1 => "rm1",
        }
    }

    /// Human-readable name for messages.
    pub fn display_name(self) -> &'static str {
        match self {
            Self::Am1 => "AM1",
            Self::Rm1 => "RM1",
        }
    }

    /// Every method this build supports.
    pub fn all() -> &'static [NddoMethod] {
        &[Self::Am1, Self::Rm1]
    }

    /// Parse a method name, case-insensitively.
    pub fn parse(text: &str) -> Result<Self> {
        match text.trim().to_ascii_lowercase().as_str() {
            "am1" => Ok(Self::Am1),
            "rm1" => Ok(Self::Rm1),
            other => Err(Am1Error::InvalidInput(format!(
                "unknown method `{other}`; expected one of: {}",
                Self::all()
                    .iter()
                    .map(|m| m.name())
                    .collect::<Vec<_>>()
                    .join(", ")
            ))),
        }
    }
}

impl std::fmt::Display for NddoMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.display_name())
    }
}

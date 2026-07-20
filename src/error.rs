// SPDX-License-Identifier: GPL-3.0-or-later

use std::fmt;

pub type Result<T> = std::result::Result<T, Am1Error>;

/// Errors raised across the AM1 pipeline.
///
/// Parallels `gfn1-rs`'s `Gfn1Error`, renamed and trimmed for the molecular NDDO
/// method: there is no global-parameter table and no periodic cell, and SCF
/// non-convergence is reported on the density residual rather than a shell-charge rms.
#[derive(Debug)]
pub enum Am1Error {
    Io(std::io::Error),
    Parse { line: usize, message: String },
    InvalidInput(String),
    /// No AM1 parameter block exists for this atomic number.
    MissingElement(u8),
    /// A named per-element or derived parameter is absent.
    MissingParameter(String),
    LinearAlgebra(String),
    /// The SCF loop hit `max_scf` without reaching the density/energy tolerance.
    ScfNotConverged { iterations: usize, error: f64 },
}

impl fmt::Display for Am1Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(err) => write!(f, "{err}"),
            Self::Parse { line, message } => write!(f, "parse error at line {line}: {message}"),
            Self::InvalidInput(msg) => write!(f, "{msg}"),
            Self::MissingElement(z) => write!(f, "missing AM1 parameter block for Z={z}"),
            Self::MissingParameter(key) => write!(f, "missing AM1 parameter `{key}`"),
            Self::LinearAlgebra(msg) => write!(f, "linear algebra error: {msg}"),
            Self::ScfNotConverged { iterations, error } => write!(
                f,
                "AM1 SCF did not converge after {iterations} iterations (error={error:.3e})"
            ),
        }
    }
}

impl std::error::Error for Am1Error {}

impl From<std::io::Error> for Am1Error {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

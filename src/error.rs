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
    Parse {
        line: usize,
        message: String,
    },
    InvalidInput(String),
    /// No AM1 parameter block exists for this atomic number.
    MissingElement(u8),
    /// A named per-element or derived parameter is absent.
    MissingParameter(String),
    /// The selected method has no parameters for this element. Distinct from
    /// [`Am1Error::MissingElement`] because the element may well be parameterized by another
    /// method in this crate — silicon is fine under AM1 but is not in RM1's published set —
    /// so the message has to name the method and its coverage.
    ElementNotParameterized {
        method: &'static str,
        z: u8,
        supported: String,
    },
    LinearAlgebra(String),
    /// The SCF loop hit `max_scf` without reaching the density/energy tolerance.
    ScfNotConverged {
        iterations: usize,
        error: f64,
    },
    /// A non-finite value reached the SCF's energy or commutator: the iteration **diverged**
    /// rather than failing to converge.
    ///
    /// A separate variant because it is a separate diagnosis. Once `NaN` is in the density every
    /// comparison against a tolerance is false, so the loop cannot converge and cannot detect
    /// that it cannot — the old behaviour was to run to the iteration limit and report
    /// `error=NaN`, which reads as "needs more iterations" and is the opposite of the truth.
    ScfDiverged {
        iterations: usize,
    },
    /// The coupled-perturbed (CPHF/UCPHF) solve for one or more nuclear perturbations hit its
    /// iteration limit. The orbital-relaxation part of the Hessian would be wrong, so this is
    /// reported rather than folded silently into the result.
    CphfNotConverged {
        perturbations: usize,
        iterations: usize,
        residual: f64,
        /// The orbital Hessian had non-positive curvature: the SCF converged to a saddle point in
        /// orbital space rather than a minimum. A different diagnosis from a slow solve, and one
        /// that no iteration limit or tolerance can repair.
        unstable: bool,
    },
    /// A periodic **response** was asked for on a ground state with fractionally occupied levels.
    ///
    /// Every response path here — the `q = 0` Hessian, `Z*`, `ε_∞`, the polarizability, DFPT —
    /// derives from a fixed integer occupation. The coupled-perturbed equations carry no
    /// `∂f/∂ε` term, so the Fermi surface cannot respond to the perturbation, and there is no
    /// tolerance or iteration count that supplies one.
    ///
    /// The two paths fail differently and both fail *quietly*, which is why this is an error and
    /// not a warning. The CPHF path classifies each level as occupied or virtual and a partially
    /// filled one is **neither**, so it is dropped from the response entirely. The DFPT path keeps
    /// every band pair but weights them by a frozen `f_n(k) − f_m(k+q)`, so it answers a question
    /// about a system whose occupations cannot change.
    ///
    /// This is *not* a refusal to use smearing. Smearing is what converges the ground state of a
    /// small-gap or coarsely sampled solid, and a gapped system smeared at a `kT` well below its
    /// gap has occupations exponentially close to integer and passes. What is refused is the case
    /// where the converged occupations are genuinely fractional — which is the case where the
    /// answer would have been wrong.
    FractionalOccupation {
        /// Index into the k-point list the response was solved on.
        k_index: usize,
        /// Band index within that k point.
        level: usize,
        /// Electrons in that orbital.
        occupation: f64,
        /// What a full orbital holds here: 2 restricted, 1 per unrestricted channel.
        full: f64,
        /// The electronic temperature (eV) the ground state was converged at.
        smearing_ev: f64,
        /// Which response path found it, for the message.
        path: &'static str,
    },
}

impl fmt::Display for Am1Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(err) => write!(f, "{err}"),
            Self::Parse { line, message } => write!(f, "parse error at line {line}: {message}"),
            Self::InvalidInput(msg) => write!(f, "{msg}"),
            Self::MissingElement(z) => write!(f, "missing AM1 parameter block for Z={z}"),
            Self::MissingParameter(key) => write!(f, "missing AM1 parameter `{key}`"),
            Self::ElementNotParameterized {
                method,
                z,
                supported,
            } => write!(
                f,
                "{method} has no parameters for element Z={z}; {method} covers {supported}"
            ),
            Self::LinearAlgebra(msg) => write!(f, "linear algebra error: {msg}"),
            Self::ScfNotConverged { iterations, error } => write!(
                f,
                "AM1 SCF did not converge after {iterations} iterations (error={error:.3e})"
            ),
            Self::ScfDiverged { iterations } => write!(
                f,
                "the AM1 SCF diverged at iteration {iterations}: a non-finite energy or density \
                 appeared, so no number of further iterations can help. This is usually a system \
                 the model cannot describe -- a dense ionic or metallic solid, or a geometry with \
                 overlapping atoms. Under a cell, check `max_image_overlap`: NDDO assumes an atom \
                 has no overlap with its own periodic images, and a dense crystal violates that \
                 badly. Set AM1_SCF_DEBUG=1 to trace the iteration"
            ),
            Self::CphfNotConverged {
                perturbations,
                iterations,
                residual,
                unstable,
            } => {
                if *unstable {
                    write!(
                        f,
                        "the CPHF response did not converge for {perturbations} nuclear \
                         perturbation(s) (worst residual={residual:.3e}), because the orbital \
                         Hessian is not positive definite: the SCF converged to a saddle point \
                         in orbital space, not a minimum. The response has no unique variational \
                         solution there, so no iteration limit or tolerance will fix it. For an \
                         open-shell system this is usually a spin (UHF) instability -- try a \
                         different starting guess or reference, or check that the geometry is a \
                         minimum"
                    )
                } else {
                    write!(
                        f,
                        "the CPHF response did not converge for {perturbations} nuclear \
                         perturbation(s) after {iterations} iterations (worst \
                         residual={residual:.3e}); the analytic Hessian's orbital-relaxation \
                         term would be unreliable. This usually means a small HOMO-LUMO gap"
                    )
                }
            }
            Self::FractionalOccupation {
                k_index,
                level,
                occupation,
                full,
                smearing_ev,
                path,
            } => write!(
                f,
                "the {path} response needs integer occupations, and this ground state does not \
                 have them: at k-point {k_index}, band {level} holds {occupation:.6} of {full} \
                 electrons. The coupled-perturbed equations carry no occupation-response term, so \
                 the Fermi surface cannot respond to the perturbation -- the CPHF path drops a \
                 partially filled level from both the occupied and the virtual set, and the DFPT \
                 path freezes `f_n(k) - f_m(k+q)`. Either way the answer would be wrong and \
                 nothing would say so, which is why this is an error.\n\
                 \n\
                 This is not a refusal to use smearing (here {smearing_ev} eV): a gapped system \
                 smeared well below its gap has occupations exponentially close to integer and is \
                 accepted. Reaching a converged state whose occupations are *also* integer is \
                 usually a finer k-mesh rather than more smearing -- widening `kT` moves \
                 occupations away from integer, which is the opposite of what this needs. If the \
                 system is genuinely metallic, no setting here makes its response right"
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

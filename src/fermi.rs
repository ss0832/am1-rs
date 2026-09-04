// SPDX-License-Identifier: GPL-3.0-or-later

//! Orbital filling: aufbau and Fermi–Dirac occupations with a bisected chemical potential,
//! the electronic entropy, and the `T → 0` energy extrapolation.
//!
//! The molecular SCF fills orbitals by strict aufbau, which is fine when there is one
//! spectrum and a gap. Two things coming up break both assumptions:
//!
//! * **k-point sampling.** Occupations are decided across *all* k-points at once against a
//!   single Fermi level, not per k-point. A band that is occupied at one k and empty at
//!   another is the normal case, not an error, and sharp filling makes the energy a
//!   discontinuous function of geometry there.
//! * **Divide-and-conquer.** Subsystems share electrons, so their spectra are only on a
//!   common footing through a common chemical potential. Each subsystem orbital enters with
//!   the fraction of it that belongs to that subsystem, which is not an integer — so the
//!   occupied set never sums to a whole number and sharp filling has to hand the remainder to
//!   whichever level happens to sort first.
//!
//! Both reduce to the same problem — fill a weighted list of levels to a given electron count
//! — so there is one implementation, and the weight carries whatever the caller means by it
//! (k-point weight, spin channel capacity, subsystem projection, or a product of those).

use crate::error::{Am1Error, Result};

/// Boltzmann constant in eV/K, matching the crate's internal energy unit.
pub const KB_EV_PER_K: f64 = 8.617_333_262e-5;

/// One fillable level.
#[derive(Clone, Copy, Debug)]
pub struct Level {
    /// Orbital energy, eV.
    pub energy: f64,
    /// How many electrons this level can hold: 2 for a spin-restricted molecular orbital, 1
    /// for one spin channel, times the k-point weight, times any subsystem projection.
    pub weight: f64,
}

/// How to fill.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum Filling {
    /// Fill from the bottom, splitting the remainder over exactly degenerate levels.
    #[default]
    Aufbau,
    /// Fermi–Dirac at electronic temperature `kt` (eV).
    Fermi { kt: f64 },
}

/// The result of a filling.
#[derive(Clone, Debug)]
pub struct Occupations {
    /// Occupation *fraction* per level, in the caller's original order. Multiply by the
    /// level's weight to get electrons.
    pub fractions: Vec<f64>,
    /// Chemical potential, eV. For aufbau this is the midpoint of the frontier gap.
    pub fermi_energy: f64,
    /// `T·S`, eV — the entropic term, zero for aufbau.
    pub ts: f64,
    /// `Σ wᵢ fᵢ εᵢ`, eV.
    pub band_energy: f64,
    /// Electrons actually placed, for the caller to check against what it asked for.
    pub electrons: f64,
}

impl Occupations {
    /// Electrons on level `i`.
    pub fn electrons_on(&self, levels: &[Level], i: usize) -> f64 {
        self.fractions[i] * levels[i].weight
    }

    /// `E − TS`, the quantity that is variational at finite electronic temperature.
    pub fn free_energy(&self) -> f64 {
        self.band_energy - self.ts
    }

    /// Energy extrapolated back to `T → 0`.
    ///
    /// For Fermi–Dirac the leading finite-temperature error in `E` and in `E − TS` are equal
    /// and opposite, so their mean cancels it: `E₀ ≈ E − TS/2`.
    pub fn extrapolated_energy(&self) -> f64 {
        self.band_energy - 0.5 * self.ts
    }
}

/// Fermi–Dirac occupation of a level `x = (ε − μ)/kT`, written to avoid overflow at large |x|.
#[inline]
fn fermi_dirac(x: f64) -> f64 {
    if x > 40.0 {
        0.0
    } else if x < -40.0 {
        1.0
    } else {
        1.0 / (1.0 + x.exp())
    }
}

/// Fill `levels` with `n_electrons`.
///
/// Fails if the electron count is negative or exceeds the total capacity, rather than
/// silently returning a filling that does not hold the electrons it was given.
pub fn fill(levels: &[Level], n_electrons: f64, filling: Filling) -> Result<Occupations> {
    let capacity: f64 = levels.iter().map(|l| l.weight).sum();
    if n_electrons < -1.0e-9 {
        return Err(Am1Error::InvalidInput(format!(
            "cannot fill {n_electrons} electrons"
        )));
    }
    if n_electrons > capacity + 1.0e-9 {
        return Err(Am1Error::InvalidInput(format!(
            "{n_electrons} electrons exceed the basis capacity of {capacity}"
        )));
    }
    if levels.is_empty() {
        return Ok(Occupations {
            fractions: Vec::new(),
            fermi_energy: 0.0,
            ts: 0.0,
            band_energy: 0.0,
            electrons: 0.0,
        });
    }

    match filling {
        Filling::Aufbau => Ok(fill_aufbau(levels, n_electrons)),
        Filling::Fermi { kt } if kt <= 0.0 => Ok(fill_aufbau(levels, n_electrons)),
        Filling::Fermi { kt } => Ok(fill_fermi(levels, n_electrons, kt)),
    }
}

fn fill_aufbau(levels: &[Level], n_electrons: f64) -> Occupations {
    let mut order: Vec<usize> = (0..levels.len()).collect();
    order.sort_by(|&a, &b| {
        levels[a]
            .energy
            .partial_cmp(&levels[b].energy)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut fractions = vec![0.0; levels.len()];
    let mut left = n_electrons;
    // The highest level, or zero if there are none. An empty level set is degenerate rather than
    // erroneous — a system with no orbitals holds no electrons — and it reached here as an
    // `order.last().unwrap()`, which would have panicked on it.
    let mut fermi = order.last().map(|&i| levels[i].energy).unwrap_or(0.0);
    let mut i = 0usize;
    while i < order.len() {
        if left <= 1.0e-14 {
            // The chemical potential sits between the last filled and first empty level.
            fermi = 0.5 * (fermi + levels[order[i]].energy);
            break;
        }
        // Share the remainder across an exactly degenerate group, so a symmetric system does
        // not get an arbitrary tie-break.
        let e = levels[order[i]].energy;
        let mut j = i;
        let mut group_capacity = 0.0;
        while j < order.len() && (levels[order[j]].energy - e).abs() < 1.0e-10 {
            group_capacity += levels[order[j]].weight;
            j += 1;
        }
        let take = left.min(group_capacity);
        let frac = if group_capacity > 0.0 {
            take / group_capacity
        } else {
            0.0
        };
        for &k in &order[i..j] {
            fractions[k] = frac;
        }
        left -= take;
        fermi = e;
        i = j;
    }

    let (band_energy, electrons) = totals(levels, &fractions);
    Occupations {
        fractions,
        fermi_energy: fermi,
        ts: 0.0,
        band_energy,
        electrons,
    }
}

fn fill_fermi(levels: &[Level], n_electrons: f64, kt: f64) -> Occupations {
    // Bracket the chemical potential, then bisect. The electron count is monotone in mu, so
    // bisection cannot miss and needs no derivative.
    let mut lo = levels
        .iter()
        .map(|l| l.energy)
        .fold(f64::INFINITY, f64::min)
        - 50.0 * kt
        - 1.0;
    let mut hi = levels
        .iter()
        .map(|l| l.energy)
        .fold(f64::NEG_INFINITY, f64::max)
        + 50.0 * kt
        + 1.0;

    let count = |mu: f64| -> f64 {
        levels
            .iter()
            .map(|l| l.weight * fermi_dirac((l.energy - mu) / kt))
            .sum()
    };

    // 200 bisections take the bracket below any representable width; the loop exits on the
    // electron-count residual well before that.
    let mut mu = 0.5 * (lo + hi);
    for _ in 0..200 {
        mu = 0.5 * (lo + hi);
        let n = count(mu);
        if (n - n_electrons).abs() < 1.0e-13 {
            break;
        }
        if n < n_electrons {
            lo = mu;
        } else {
            hi = mu;
        }
    }

    let fractions: Vec<f64> = levels
        .iter()
        .map(|l| fermi_dirac((l.energy - mu) / kt))
        .collect();

    // S = -k Σ w [f ln f + (1-f) ln(1-f)]; returned as T·S so it is already an energy.
    let mut s_over_k = 0.0;
    for (l, f) in levels.iter().zip(&fractions) {
        let f = *f;
        if f > 1.0e-14 && f < 1.0 - 1.0e-14 {
            s_over_k -= l.weight * (f * f.ln() + (1.0 - f) * (1.0 - f).ln());
        }
    }

    let (band_energy, electrons) = totals(levels, &fractions);
    Occupations {
        fractions,
        fermi_energy: mu,
        ts: kt * s_over_k,
        band_energy,
        electrons,
    }
}

fn totals(levels: &[Level], fractions: &[f64]) -> (f64, f64) {
    let mut band = 0.0;
    let mut electrons = 0.0;
    for (l, f) in levels.iter().zip(fractions) {
        band += l.weight * f * l.energy;
        electrons += l.weight * f;
    }
    (band, electrons)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn restricted(energies: &[f64]) -> Vec<Level> {
        energies
            .iter()
            .map(|&energy| Level {
                energy,
                weight: 2.0,
            })
            .collect()
    }

    #[test]
    fn aufbau_on_a_gapped_spectrum_is_integral() {
        let levels = restricted(&[-10.0, -8.0, -6.0, 2.0, 4.0]);
        let occ = fill(&levels, 6.0, Filling::Aufbau).unwrap();
        assert_eq!(occ.fractions, vec![1.0, 1.0, 1.0, 0.0, 0.0]);
        assert!((occ.electrons - 6.0).abs() < 1.0e-14);
        assert!((occ.band_energy - 2.0 * (-10.0 - 8.0 - 6.0)).abs() < 1.0e-12);
        // The chemical potential lands in the gap.
        assert!(occ.fermi_energy > -6.0 && occ.fermi_energy < 2.0);
        assert_eq!(occ.ts, 0.0);
    }

    #[test]
    fn aufbau_shares_a_partial_shell_over_degenerate_levels() {
        // Two electrons into a three-fold degenerate set: every partner must get the same
        // occupation, or a symmetric molecule acquires an arbitrary symmetry-broken density.
        let levels = restricted(&[-5.0, -5.0, -5.0]);
        let occ = fill(&levels, 2.0, Filling::Aufbau).unwrap();
        for f in &occ.fractions {
            assert!((f - 1.0 / 3.0).abs() < 1.0e-12, "got {:?}", occ.fractions);
        }
        assert!((occ.electrons - 2.0).abs() < 1.0e-12);
    }

    #[test]
    fn fermi_conserves_the_electron_count_exactly() {
        let levels = restricted(&[-10.0, -8.0, -7.9, -0.1, 0.0, 3.0]);
        for kt in [0.001, 0.01, 0.1, 0.5] {
            let occ = fill(&levels, 7.0, Filling::Fermi { kt }).unwrap();
            assert!(
                (occ.electrons - 7.0).abs() < 1.0e-10,
                "kt={kt} placed {} electrons",
                occ.electrons
            );
        }
    }

    #[test]
    fn fermi_approaches_aufbau_as_the_temperature_falls() {
        let levels = restricted(&[-10.0, -8.0, -6.0, 2.0, 4.0]);
        let sharp = fill(&levels, 6.0, Filling::Aufbau).unwrap();
        let mut previous_diff = f64::INFINITY;
        let mut previous_ts = f64::INFINITY;
        for kt in [0.5, 0.1, 0.01, 0.001] {
            let occ = fill(&levels, 6.0, Filling::Fermi { kt }).unwrap();
            let diff: f64 = occ
                .fractions
                .iter()
                .zip(&sharp.fractions)
                .map(|(a, b)| (a - b).abs())
                .sum();
            // Non-increasing rather than strictly decreasing, for the same reason as the
            // entropy below: with an 8 eV gap the occupations reach exactly 0 and 1 well
            // before the lowest temperature tested, and stay there.
            assert!(
                diff <= previous_diff,
                "occupations moved away from aufbau going to kt={kt}: {previous_diff:.3e} then {diff:.3e}"
            );
            // Entropy is non-negative and non-increasing as the temperature falls. Not
            // strictly decreasing: this spectrum has an 8 eV gap, so by kt = 0.1 the frontier
            // levels are 80 kT away, every occupation is 0 or 1 to machine precision, and the
            // entropy is exactly zero from there down.
            assert!(
                occ.ts >= 0.0 && occ.ts <= previous_ts,
                "T*S rose going to kt={kt}: {} then {}",
                previous_ts,
                occ.ts
            );
            previous_diff = diff;
            previous_ts = occ.ts;
        }
        assert!(
            previous_diff < 1.0e-9,
            "residual occupation difference {previous_diff:.3e}"
        );
        assert!(previous_ts < 1.0e-12, "residual T*S {previous_ts:.3e}");
    }

    #[test]
    fn the_entropy_is_positive_and_the_extrapolation_sits_between() {
        // A metallic case: levels straddling the Fermi energy.
        let levels = restricted(&[-1.0, -0.1, 0.0, 0.1, 1.0]);
        let occ = fill(&levels, 5.0, Filling::Fermi { kt: 0.2 }).unwrap();
        assert!(occ.ts > 0.0, "entropy should be positive, got {}", occ.ts);
        let (e, f, x) = (
            occ.band_energy,
            occ.free_energy(),
            occ.extrapolated_energy(),
        );
        assert!(f < x && x < e, "expected F < E0 < E, got {f} {x} {e}");
        assert!(
            ((e + f) / 2.0 - x).abs() < 1.0e-12,
            "the extrapolation is the mean of E and F"
        );
    }

    #[test]
    fn weights_carry_k_point_and_subsystem_factors() {
        // Two k-points of weight 1/2, spin-restricted, with different spectra: the filling is
        // decided against one chemical potential across both, not per k-point.
        let levels = vec![
            Level {
                energy: -5.0,
                weight: 1.0,
            },
            Level {
                energy: 1.0,
                weight: 1.0,
            },
            Level {
                energy: -4.0,
                weight: 1.0,
            },
            Level {
                energy: -0.5,
                weight: 1.0,
            },
        ];
        let occ = fill(&levels, 3.0, Filling::Fermi { kt: 0.01 }).unwrap();
        assert!((occ.electrons - 3.0).abs() < 1.0e-10);
        // The three lowest (-5, -4, -0.5) are full and the +1 level is empty.
        assert!(occ.fractions[0] > 0.99 && occ.fractions[2] > 0.99 && occ.fractions[3] > 0.99);
        assert!(occ.fractions[1] < 0.01);
        assert!(occ.fermi_energy > -0.5 && occ.fermi_energy < 1.0);
    }

    #[test]
    fn asking_for_more_electrons_than_the_basis_holds_is_an_error() {
        let levels = restricted(&[-1.0, 0.0]);
        let err = fill(&levels, 5.0, Filling::Aufbau).unwrap_err();
        assert!(err.to_string().contains("exceed"), "{err}");
        let err = fill(&levels, -1.0, Filling::Aufbau).unwrap_err();
        assert!(err.to_string().contains("cannot fill"), "{err}");
    }

    #[test]
    fn an_empty_or_full_system_fills_without_incident() {
        let levels = restricted(&[-1.0, 0.0]);
        let empty = fill(&levels, 0.0, Filling::Fermi { kt: 0.1 }).unwrap();
        assert!(empty.electrons.abs() < 1.0e-10);
        let full = fill(&levels, 4.0, Filling::Fermi { kt: 0.1 }).unwrap();
        assert!((full.electrons - 4.0).abs() < 1.0e-10);
        assert!(full.ts.abs() < 1.0e-9, "a full band has no entropy");
    }
}

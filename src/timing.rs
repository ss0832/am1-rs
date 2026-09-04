// SPDX-License-Identifier: GPL-3.0-or-later

//! Opt-in phase timing, so performance work is aimed at measurements rather than at guesses.
//!
//! Set `AM1_TIMING=1` and a breakdown is written to stderr when the calculation finishes:
//!
//! ```text
//! am1-rs timing (399 atoms, 798 AOs)
//!   basis+core          1.832 s   12.7 %
//!   scf:fock            8.104 s   56.2 %
//!   scf:eigen           3.221 s   22.3 %
//!   ...
//! ```
//!
//! Everything here is inert when the variable is unset: [`enabled`] is resolved once and the
//! [`Timer`] guard becomes a pair of branches with no clock reads.
//!
//! # Who calls [`report`]
//!
//! **Only the top-level caller** — the CLI, or a test that wants a breakdown. Never a library
//! function.
//!
//! This is not a style preference. [`report`] clears the accumulator, so a library function that
//! reported would end the measurement at its own boundary. `run_am1` used to, and the effect was
//! that profiling a *gradient* printed only the SCF phases: the gradient's own work happened
//! after the SCF's report had already fired and cleared the map, so the single most expensive
//! phase of that command was invisible in the profile meant to find it.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

/// Whether `AM1_TIMING` is set. Resolved once per process.
pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("AM1_TIMING").is_some())
}

fn totals() -> &'static Mutex<HashMap<&'static str, (f64, u64)>> {
    static T: OnceLock<Mutex<HashMap<&'static str, (f64, u64)>>> = OnceLock::new();
    T.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Add `secs` to the running total for `name`.
pub fn record(name: &'static str, secs: f64) {
    if !enabled() {
        return;
    }
    if let Ok(mut map) = totals().lock() {
        let entry = map.entry(name).or_insert((0.0, 0));
        entry.0 += secs;
        entry.1 += 1;
    }
}

/// Scope guard that records elapsed wall time under `name` when dropped.
///
/// Wall time is summed across threads, so a rayon-parallel phase reports more seconds than it
/// occupied. That is deliberate: it is the quantity that says where the *work* is, which is
/// what decides whether a phase is worth optimizing or worth parallelizing.
pub struct Timer {
    name: &'static str,
    start: Option<Instant>,
}

impl Timer {
    pub fn start(name: &'static str) -> Self {
        Self {
            name,
            start: if enabled() {
                Some(Instant::now())
            } else {
                None
            },
        }
    }
}

impl Drop for Timer {
    fn drop(&mut self) {
        if let Some(t) = self.start {
            record(self.name, t.elapsed().as_secs_f64());
        }
    }
}

/// Print the accumulated breakdown to stderr and reset it. No-op unless `AM1_TIMING` is set.
///
/// # These are thread-seconds, not wall-clock seconds
///
/// A [`Timer`] records the elapsed time of the scope it guards, and adds it to a global total.
/// When the guarded scope runs on many threads at once — which is most of them — every thread
/// contributes its own elapsed time, so a phase that took 3 s of wall clock on sixteen threads
/// is reported as roughly 48 s.
///
/// This is the right quantity for the question the report is for ("where does the work go"), and
/// the wrong one for "how long did this take". Read as wall clock it overstates a parallel phase
/// by the thread count, which makes a well-parallelized phase look like the bottleneck — the
/// CPHF Fock builds in a frequency run report 39 s against a 4.8 s calculation.
///
/// The header says so, because the numbers do not.
pub fn report(context: &str) {
    if !enabled() {
        return;
    }
    let Ok(mut map) = totals().lock() else { return };
    if map.is_empty() {
        return;
    }
    let mut rows: Vec<(&'static str, f64, u64)> = map.iter().map(|(k, v)| (*k, v.0, v.1)).collect();
    rows.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let total: f64 = rows.iter().map(|(_, s, _)| *s).sum();
    eprintln!("am1-rs timing ({context})");
    eprintln!("  thread-seconds, summed over threads -- a phase running on N threads reports ~N x");
    eprintln!("  its wall clock, and the percentages are shares of the summed total, not of a run");
    for (name, secs, calls) in &rows {
        let pct = if total > 0.0 {
            100.0 * secs / total
        } else {
            0.0
        };
        eprintln!("  {name:<22} {secs:9.3} s  {pct:5.1} %  ({calls} calls)");
    }
    eprintln!("  {:<22} {total:9.3} s", "TOTAL (thread-seconds)");
    map.clear();
}

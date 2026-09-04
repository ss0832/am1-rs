// SPDX-License-Identifier: GPL-3.0-or-later

//! The per-method parameter cache, and the fixed per-call cost it removes.
//!
//! `Am1Parameters::for_method` is called at the top of **every** function on the Python surface.
//! Before 0.2.2 each of those calls re-parsed the embedded CSV and re-ran the secant solves for
//! `rho1`/`rho2` on every element. Against an 800-atom cluster that is invisible; against a water
//! molecule in a molecular-dynamics loop it is the step.
//!
//! That is the shape of cost a large-system profile cannot find, which is why it is asserted here
//! rather than left to a benchmark nobody runs.

use am1_rs::{Am1Parameters, NddoMethod};

/// Minimum over repetitions, not a mean: interference only ever makes a sample slower, so the
/// minimum is the least-contended estimate. The project uses the same rule in `linalg.rs`.
fn best<F: FnMut()>(reps: usize, mut f: F) -> f64 {
    f(); // warm up
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t = std::time::Instant::now();
        f();
        best = best.min(t.elapsed().as_secs_f64());
    }
    best
}

#[test]
fn the_cached_parameter_set_costs_far_less_than_building_one() {
    let csv_work = best(20, || {
        // What `for_method` used to do on every call: parse and re-derive.
        let p = Am1Parameters::from_csv(include_str!("../src/data/am1_parameters.csv")).unwrap();
        std::hint::black_box(p);
    });
    let cached = best(20, || {
        std::hint::black_box(Am1Parameters::for_method(NddoMethod::Am1).unwrap());
    });
    let borrowed = best(20, || {
        std::hint::black_box(Am1Parameters::shared(NddoMethod::Am1).unwrap());
    });

    // The borrow is below the clock's resolution, so it is reported as such rather than as a
    // ratio against zero — "269500000x" would be a statement about the timer, not the code.
    eprintln!(
        "    build from CSV {:.1} us, cached clone {:.1} us ({:.0}x), borrowed {}",
        csv_work * 1e6,
        cached * 1e6,
        csv_work / cached.max(1e-12),
        if borrowed < 1e-8 {
            "below clock resolution".to_string()
        } else {
            format!("{:.3} us", borrowed * 1e6)
        },
    );

    // Deliberately loose. The point is an order of magnitude, and a tighter bound would be an
    // assertion about how busy the machine is.
    assert!(
        cached < csv_work / 3.0,
        "the cached set should be several times cheaper than building one; \
         build {:.1} us against cached {:.1} us",
        csv_work * 1e6,
        cached * 1e6
    );
}

/// The cache must not change what is returned, including across methods — one `OnceLock` per
/// method, not one shared slot that the second caller overwrites or reads stale.
#[test]
fn the_cache_returns_the_right_set_per_method() {
    for method in [NddoMethod::Am1, NddoMethod::Rm1] {
        let a = Am1Parameters::for_method(method).unwrap();
        let b = Am1Parameters::shared(method).unwrap();
        assert_eq!(a.method, method);
        assert_eq!(b.method, method);
        assert_eq!(a.elements.len(), b.elements.len());
        for (z, e) in &a.elements {
            let f = b.elements.get(z).expect("same elements");
            assert_eq!(e.u_ss, f.u_ss);
            assert_eq!(e.rho0, f.rho0);
            assert_eq!(e.rho1, f.rho1);
            assert_eq!(e.rho2, f.rho2);
        }
    }
    // Silicon is AM1-only; RM1 must still refuse it after both have been cached.
    assert!(Am1Parameters::shared(NddoMethod::Am1)
        .unwrap()
        .element(14)
        .is_ok());
    assert!(Am1Parameters::shared(NddoMethod::Rm1)
        .unwrap()
        .element(14)
        .is_err());
}

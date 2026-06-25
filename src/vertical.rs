// AI Generated Dataset Producer

//! Vertical dataset generator, port of nnfs `vertical_data`.
//!
//! Each class is a vertical Gaussian blob: x centered at `class / 3`,
//! y centered at 0.5, both with stddev 0.1.

/// A single 2D sample with its integer class label.
#[derive(Debug, Clone, Copy)]
pub struct Sample {
    pub x: f64,
    pub y: f64,
    pub label: usize,
}

/// Minimal xorshift64 RNG so we don't pull in the `rand` crate.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        // Avoid the all-zero state, which xorshift can't escape.
        Rng(if seed == 0 { 0x9E3779B97F4A7C15 } else { seed })
    }

    /// Uniform f64 in [0, 1).
    fn next_f64(&mut self) -> f64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        // Top 53 bits -> uniform double.
        (x >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Standard normal sample via Box-Muller.
    fn next_gaussian(&mut self) -> f64 {
        // u1 must be > 0 for ln; nudge away from 0.
        let u1 = self.next_f64().max(f64::MIN_POSITIVE);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// Generate `samples` points per class for `classes` classes.
///
/// Returns `samples * classes` labeled points, grouped by class.
pub fn vertical_data(samples: usize, classes: usize, seed: u64) -> Vec<Sample> {
    let mut rng = Rng::new(seed);
    let mut out = Vec::with_capacity(samples * classes);

    for class in 0..classes {
        let cx = class as f64 / 3.0;
        for _ in 0..samples {
            out.push(Sample {
                x: rng.next_gaussian() * 0.1 + cx,
                y: rng.next_gaussian() * 0.1 + 0.5,
                label: class,
            });
        }
    }

    out
}

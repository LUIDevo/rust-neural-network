// AI Generated Dataset Producer

//! Vertical dataset generator, port of nnfs `vertical_data`.
//!
//! Each class is a vertical Gaussian blob: x centered at `class / 3`,
//! y centered at 0.5, both with stddev 0.1.

use crate::rng::Rng;

/// A single 2D sample with its integer class label.
#[derive(Debug, Clone, Copy)]
pub struct Sample {
    pub x: (f64, f64),
    pub y: usize,
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
                x: (
                    rng.next_gaussian() * 0.1 + cx,
                    rng.next_gaussian() * 0.1 + 0.5,
                ),
                y: class,
            });
        }
    }

    out
}

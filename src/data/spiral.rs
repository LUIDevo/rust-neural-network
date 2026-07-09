// AI Generated Dataset Producer

//! Spiral dataset generator, port of nnfs `spiral_data`.
//!
//! Each class is one arm of an intertwined spiral — not linearly
//! separable, so it actually stresses the optimiser.

use crate::math::matrix::Matrix;
use crate::math::rng::Rng;

/// Generate `samples` points per class for `classes` classes.
///
/// Returns `(X, y)`:
/// - `X`: `(samples * classes) x 2` matrix of positions, grouped by class.
/// - `y`: matching class label per row.
pub fn spiral_data(samples: usize, classes: usize, seed: u64) -> (Matrix, Vec<usize>) {
    let mut rng = Rng::new(seed);
    let n = samples * classes;
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);

    let denom = (samples - 1).max(1) as f64;
    for class in 0..classes {
        for i in 0..samples {
            let frac = i as f64 / denom;
            // r: 0 -> 1 along the arm.
            let r = frac;
            // t: angle sweeps class*4 -> (class+1)*4 (linspace), plus jitter.
            let t = class as f64 * 4.0 + 4.0 * frac + rng.next_gaussian() * 0.2;
            x.push(vec![r * (t * 2.5).sin(), r * (t * 2.5).cos()]);
            y.push(class);
        }
    }

    (x, y)
}

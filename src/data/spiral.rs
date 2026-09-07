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
/// - `X`: row-major flat `Vec<f32>` of `samples * classes` positions, grouped
///   by class, wrapped in a `(samples * classes) x 2` [`Matrix`].
/// - `y`: matching class label per row.
pub fn spiral_data(samples: usize, classes: usize, seed: u64) -> (Matrix, Vec<usize>) {
    let mut rng = Rng::new(seed);
    let n = samples * classes;
    let mut x = Vec::with_capacity(n * 2);
    let mut y = Vec::with_capacity(n);

    let denom = (samples - 1).max(1) as f32;
    for class in 0..classes {
        for i in 0..samples {
            let frac = i as f32 / denom;
            // r: 0 -> 1 along the arm.
            let r = frac;
            // t: angle sweeps class*4 -> (class+1)*4 (linspace), plus jitter.
            let t = class as f32 * 4.0 + 4.0 * frac + rng.next_gaussian() * 0.2;
            x.push(r * (t * 2.5).sin());
            x.push(r * (t * 2.5).cos());
            y.push(class);
        }
    }

    (Matrix::new(x, n, 2), y)
}

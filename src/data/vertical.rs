// AI Generated Dataset Producer

//! Vertical dataset generator, port of nnfs `vertical_data`.
//!
//! Each class is a vertical Gaussian blob: x centered at `class / 3`,
//! y centered at 0.5, both with stddev 0.1.

use crate::math::matrix::Matrix;
use crate::math::rng::Rng;

/// Generate `samples` points per class for `classes` classes.
///
/// Returns `(X, y)`:
/// - `X`: row-major flat `Vec<f32>` of `samples * classes` positions, grouped
///   by class, wrapped in a `(samples * classes) x 2` [`Matrix`].
/// - `y`: matching class label per row.
pub fn vertical_data(samples: usize, classes: usize, seed: u64) -> (Matrix, Vec<usize>) {
    let mut rng = Rng::new(seed);
    let n = samples * classes;
    let mut x = Vec::with_capacity(n * 2);
    let mut y = Vec::with_capacity(n);

    for class in 0..classes {
        let cx = class as f32 / 3.0;
        for _ in 0..samples {
            x.push(rng.next_gaussian() * 0.1 + cx);
            x.push(rng.next_gaussian() * 0.1 + 0.5);
            y.push(class);
        }
    }

    (Matrix::new(x, n, 2), y)
}

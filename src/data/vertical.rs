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
/// - `X`: `(samples * classes) x 2` matrix of positions, grouped by class.
/// - `y`: matching class label per row.
pub fn vertical_data(samples: usize, classes: usize, seed: u64) -> (Matrix, Vec<usize>) {
    let mut rng = Rng::new(seed);
    let n = samples * classes;
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);

    for class in 0..classes {
        let cx = class as f64 / 3.0;
        for _ in 0..samples {
            x.push(vec![
                rng.next_gaussian() * 0.1 + cx,
                rng.next_gaussian() * 0.1 + 0.5,
            ]);
            y.push(class);
        }
    }

    (x, y)
}

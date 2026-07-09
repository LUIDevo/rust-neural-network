//! AI Generated sine dataset generator, port of nnfs `sine_data`.
//!
//! A single continuous curve `y = sin(2*pi*x)` — the canonical toy
//! regression problem for a linear-output + MSE model.

use crate::math::matrix::Matrix;

/// Generate `samples` points along `y = sin(2*pi*x)`.
///
/// Deterministic linspace, no noise.
///
/// Returns `(X, y)`:
/// - `X`: `samples x 1` matrix of inputs, evenly spaced in `[0, 1)`.
/// - `y`: `samples x 1` matrix of matching sine targets.
pub fn sine_data(samples: usize) -> (Matrix, Matrix) {
    let denom = samples as f64;
    let mut x = Vec::with_capacity(samples);
    let mut y = Vec::with_capacity(samples);

    for i in 0..samples {
        let xi = i as f64 / denom;
        x.push(vec![xi]);
        y.push(vec![(2.0 * std::f64::consts::PI * xi).sin()]);
    }

    (x, y)
}

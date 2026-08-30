//! AI Generated sine dataset generator, port of nnfs `sine_data`.
//!
//! A single continuous curve `y = sin(2*pi*x)` — the canonical toy
//! regression problem for a linear-output + MSE model.

/// Generate `samples` points along `y = sin(2*pi*x)`.
///
/// Deterministic linspace, no noise.
///
/// Returns `(X, y)`:
/// - `X`: flat `Vec<f32>` of `samples` inputs, evenly spaced in `[0, 1)`.
/// - `y`: flat `Vec<f32>` of matching sine targets.
pub fn sine_data(samples: usize) -> (Vec<f32>, Vec<f32>) {
    let denom = samples as f32;
    let mut x = Vec::with_capacity(samples);
    let mut y = Vec::with_capacity(samples);

    for i in 0..samples {
        let xi = i as f32 / denom;
        x.push(xi);
        y.push((2.0 * std::f32::consts::PI * xi).sin());
    }

    (x, y)
}

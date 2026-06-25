// Linear alg, custom built replacement for numpy

use crate::rng::Rng;

pub type Matrix = Vec<Vec<f64>>;

pub fn randn_matrix(rows: usize, cols: usize, stddev: f64, rng: &mut Rng) -> Matrix {
    (0..rows)
        .map(|_| (0..cols).map(|_| rng.next_gaussian() * stddev).collect())
        .collect()
}

pub fn dot(a: &Matrix, b: &Matrix) -> Matrix {}

pub fn transpose(a: &Matrix) -> Matrix {}

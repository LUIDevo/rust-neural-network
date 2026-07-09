// Linear alg, custom built replacement for numpy

use crate::math::rng::Rng;

pub type Matrix = Vec<Vec<f64>>;

pub fn randn_matrix(rows: usize, cols: usize, stddev: f64, rng: &mut Rng) -> Matrix {
    (0..rows)
        .map(|_| (0..cols).map(|_| rng.next_gaussian() * stddev).collect())
        .collect()
}

pub fn dot(a: &Matrix, b: &Matrix) -> Matrix {
    let (m, n, p) = (a.len(), b.len(), b[0].len());
    let mut out = vec![vec![0.0; p]; m];
    for i in 0..m {
        for j in 0..n {
            let aij = a[i][j];
            for k in 0..p {
                out[i][k] += aij * b[j][k];
            }
        }
    }
    out
}

pub fn sum(a: &Matrix, b: &Vec<f64>) -> Matrix {
    a.iter()
        .map(|row| row.iter().zip(b).map(|(z, bias)| z + bias).collect())
        .collect()
}

pub fn transpose(a: &Matrix) -> Matrix {
    let (r, c) = (a.len(), a[0].len());
    let mut out = vec![vec![0.0; r]; c];
    for i in 0..r {
        for j in 0..c {
            out[j][i] = a[i][j];
        }
    }
    out
}

pub fn row_sum(a: &Matrix) -> Vec<f64> {
    let (n, m) = (a.len(), a[0].len());
    let mut out = vec![0.0; n];
    for r in 0..n {
        for c in 0..m {
            out[r] += a[r][c];
        }
    }
    out
}

pub fn divide(a: &Matrix, b: &Vec<f64>) -> Matrix {
    a.iter()
        .zip(b)
        .map(|(m, n)| m.into_iter().map(|l| l / n).collect())
        .collect()
}

pub fn col_sum(a: &Matrix) -> Vec<f64> {
    let cols = a[0].len();
    let mut out = vec![0.0; cols];
    for row in a {
        for (j, &v) in row.iter().enumerate() {
            out[j] += v;
        }
    }
    out
}

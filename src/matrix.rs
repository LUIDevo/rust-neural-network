// Linear alg, custom built replacement for numpy

use crate::rng::Rng;

pub type Matrix = Vec<Vec<f64>>;

pub fn randn_matrix(rows: usize, cols: usize, stddev: f64, rng: &mut Rng) -> Matrix {
    (0..rows)
        .map(|_| (0..cols).map(|_| rng.next_gaussian() * stddev).collect())
        .collect()
}

pub fn dot(a: &Matrix, b: &Matrix) -> Matrix {
    let (m,n,p)=(a.len(), b.len(), b[0].len());
    let mut out=vec!(vec![0.0; p]; m);
    for i in 0..m {
        for j in 0..n { 
            let aij=a[i][j];
            for k in 0..p {
                out[i][k]+=aij*b[j][k];
            }
        }
    }
    out
}

pub fn transpose(a: &Matrix) -> Matrix {
    todo!()
}

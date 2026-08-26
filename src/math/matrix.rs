// Linear alg, custom built replacement for numpy

use crate::math::rng::Rng;

#[derive(Clone, Debug, PartialEq)]
pub struct Matrix {
    data: Vec<f32>,
    rows: usize,
    cols: usize,
}

impl Matrix {
    pub fn new(data: Vec<f32>, rows: usize, cols: usize) -> Self {
        assert_eq!(data.len(), rows * cols);
        Matrix { data, rows, cols }
    }
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Matrix { data: vec![0.0; rows*cols], rows, cols }
    }
    pub fn rows(&self) -> usize { self.rows }
    pub fn cols(&self) -> usize { self.cols }
}

pub fn randn_matrix(rows: usize, cols: usize, stddev: f64, rng: &mut Rng) -> Matrix {
    (0..rows)
        .map(|_| (0..cols).map(|_| rng.next_gaussian() * stddev).collect())
        .collect()
}

pub fn dot(a: &Matrix, b: &Matrix) -> Matrix {
    let (m, n, p) = (a.rows(), b.rows(), b.cols());
    let mut out = vec![0.0; m*p];
    for i in 0..m {
        for k in 0..n {
            let aik = a.data[i*n+k];
            for j in 0..p {
                out[i*p+j] += aik * b.data[k*p+j];
            }
        }
    }
    Matrix::new(out, m, p)
}

pub fn sum(a: &Matrix, b: &Vec<f64>) -> Matrix {
    let cols = a.cols();
    let data = a.iter().enumerate()
        .map(|(i,v)| v+b[i%cols])
        .collect();
    Matrix::new(data, a.rows(), cols)
}

pub fn transpose(a: &Matrix) -> Matrix {
    let (r,c)=(a.rows(), a.cols());
    let mut out = vec![0.0; r*c];
    for i in 0..r {
        for j in 0..c{
            out[j*r+i]=a.data[i*c+j];
        }
    }
    Matrix::new(out, c,r)
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

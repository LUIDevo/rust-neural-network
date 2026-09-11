use crate::math::matrix::Matrix;

#[derive(Clone, Debug, PartialEq, Default)]
pub struct Tensor {
    pub data: Vec<f32>,
    n: usize, // batch
    c: usize, // channels
    h: usize, // height
    w: usize, // width
}

impl Tensor {
    pub fn new(data: Vec<f32>, n: usize, c: usize, h: usize, w: usize) -> Self {
        assert_eq!(data.len(), n * c * h * w);
        Tensor { data, n, c, h, w }
    }
    pub fn zeros(n: usize, c: usize, h: usize, w: usize) -> Self {
        Tensor { data: vec![0.0; n * c * h * w], n, c, h, w }
    }
    pub fn n(&self) -> usize { self.n }
    pub fn c(&self) -> usize { self.c }
    pub fn h(&self) -> usize { self.h }
    pub fn w(&self) -> usize { self.w }

    pub fn idx(&self, n: usize, c: usize, y: usize, x: usize) -> usize {
        ((n * self.c + c) * self.h + y) * self.w + x
    }

    pub fn to_matrix(&self) -> Matrix {
        Matrix::new(self.data.clone(), self.n, self.c * self.h * self.w)
    }
}

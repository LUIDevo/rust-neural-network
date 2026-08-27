use crate::math::matrix::Matrix;

pub struct ActivationReLU {
    pub inputs: Matrix,
}

impl ActivationReLU {
    pub fn forward(&mut self, inputs: &Matrix) -> Matrix {
        self.inputs = inputs.clone();
        Matrix::new(inputs.data
            .iter()
            .map(|x| {
                    if *x < 0.0 { 0.0 } else { *x }
            })
            .collect(), inputs.rows(), inputs.cols())
    }
    pub fn backward(&mut self, dvalues: &Matrix) -> Matrix {
        self.inputs
            .iter()
            .zip(dvalues)
            .map(|(row, d_row)| {
                row.into_iter()
                    .zip(d_row)
                    .map(|(&x, &d)| if x > 0.0 { d } else { 0.0 })
                    .collect::<Vec<f64>>()
            })
            .collect()
    }
}

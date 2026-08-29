use crate::math::matrix::Matrix;

#[derive(Default)]
pub struct ActivationReLU {
    pub inputs: Matrix,
}

impl ActivationReLU {
    pub fn forward(&mut self, inputs: &Matrix) -> Matrix {
        self.inputs = inputs.clone();
        Matrix::new(
            inputs
                .data
                .iter()
                .map(|x| if *x < 0.0 { 0.0 } else { *x })
                .collect(),
            inputs.rows(),
            inputs.cols(),
        )
    }
    pub fn backward(&mut self, dvalues: &Matrix) -> Matrix {
        Matrix::new(
            self.inputs
                .data
                .iter()
                .zip(dvalues.data.clone())
                .map(|(row, d_row)| if *row > 0.0 { d_row } else { 0.0 })
                .collect(),
            dvalues.rows(),
            dvalues.cols(),
        )
    }
}

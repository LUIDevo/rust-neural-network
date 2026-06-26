//! Dense (fully-connected) layer.

use crate::matrix::{Matrix, dot, randn_matrix, sum, row_sum, transpose};
use crate::rng::Rng;

pub struct LayerDense {
    pub weights: Matrix,
    pub biases: Vec<f64>,
    pub dweights: Matrix,
    pub dbiases: Vec<f64>,
    pub dinputs: Matrix,
}

impl LayerDense {
    pub fn new(n_inputs: usize, n_neurons: usize, rng: &mut Rng) -> Self {
        LayerDense {
            weights: randn_matrix(n_inputs, n_neurons, 0.01, rng),
            biases: vec![0.0; n_neurons],
            dweights: vec![vec![0.0; n_neurons]; n_inputs],
            dbiases: vec![0.0; n_neurons],
            dinputs: Vec::new(),
        }
    }
    pub fn forward(self, inputs: Matrix) -> Matrix {
        sum(&dot(&self.weights, &inputs), &self.biases)
    }
    pub fn backward(mut self, dvalues: &Matrix) -> Matrix {
        self.dweights=dot(&transpose(&self.weights), &dvalues);
        self.dbiases=row_sum(&dvalues);
        todo!()
    }
}

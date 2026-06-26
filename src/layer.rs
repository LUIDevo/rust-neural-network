//! Dense (fully-connected) layer.

use crate::matrix::{Matrix, dot, randn_matrix, row_sum, sum, transpose};
use crate::rng::Rng;

pub struct LayerDense {
    pub inputs: Matrix,
    pub weights: Matrix,
    pub biases: Vec<f64>,
    pub dweights: Matrix,
    pub dbiases: Vec<f64>,
}

impl LayerDense {
    pub fn new(n_inputs: usize, n_neurons: usize, rng: &mut Rng) -> Self {
        LayerDense {
            inputs: Vec::new(),
            weights: randn_matrix(n_inputs, n_neurons, 0.01, rng),
            biases: vec![0.0; n_neurons],
            dweights: vec![vec![0.0; n_neurons]; n_inputs],
            dbiases: vec![0.0; n_neurons],
        }
    }
    pub fn forward(&mut self, inputs: &Matrix) -> Matrix {
        self.inputs = inputs.clone();
        sum(&dot(&inputs, &self.weights), &self.biases)
    }
    pub fn backward(mut self, dvalues: &Matrix) -> Matrix {
        self.dweights = dot(&transpose(&self.inputs), &dvalues);
        self.dbiases = row_sum(&dvalues);
        return dot(&dvalues, &transpose(&self.weights));
    }
}

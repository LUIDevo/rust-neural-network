//! Dense (fully-connected) layer.

use crate::math::matrix::{Matrix, col_sum, dot, randn_matrix, sum, transpose};
use crate::math::rng::Rng;
use crate::nn::activation::ActivationReLU;

pub enum Layer {
    Dense(LayerDense),
    ReLU(ActivationReLU),
    Dropout(LayerDropout),
}

pub struct LayerDense {
    pub inputs: Matrix,
    pub weights: Matrix,
    pub biases: Vec<f32>,
    pub dweights: Vec<f32>,
    pub dbiases: Vec<f32>,
    pub v_weights: Vec<f32>,
    pub v_biases: Vec<f32>,
    pub cache_weights: Matrix,
    pub cache_biases: Vec<f32>,
}

pub struct LayerDropout {
    pub rate: f32,
    pub mask: Matrix,
    pub rng: Rng,
}

impl Layer {
    pub fn forward(&mut self, inputs: &Matrix) -> Matrix {
        match self {
            Layer::Dense(l) => l.forward(inputs),
            Layer::ReLU(l) => l.forward(inputs),
            Layer::Dropout(l) => l.forward(inputs),
        }
    }
    pub fn backward(&mut self, dvalues: &Matrix) -> Matrix {
        match self {
            Layer::Dense(l) => l.backward(dvalues),
            Layer::ReLU(l) => l.backward(dvalues),
            Layer::Dropout(l) => l.backward(dvalues),
        }
    }
}

impl LayerDense {
    pub fn new(n_inputs: usize, n_neurons: usize, rng: &mut Rng) -> Self {
        LayerDense {
            inputs: Matrix::zeros(0, 0),
            weights: randn_matrix(n_inputs, n_neurons, 0.1, rng),
            biases: vec![0.0; n_neurons],
            dweights: vec![0.0; n_neurons],
            dbiases: vec![0.0; n_neurons],
            v_weights: vec![0.0; n_neurons],
            v_biases: vec![0.0; n_neurons],
            cache_weights: Matrix::zeros(n_inputs, n_neurons),
            cache_biases: vec![0.0; n_neurons],
        }
    }
    pub fn forward(&mut self, inputs: &Matrix) -> Matrix {
        self.inputs = inputs.clone();
        sum(&dot(&inputs, &self.weights), &self.biases)
    }
    pub fn backward(&mut self, dvalues: &Matrix) -> Matrix {
        self.dweights = dot(&transpose(&self.inputs), &dvalues);
        self.dbiases = col_sum(&dvalues);
        return dot(&dvalues, &transpose(&self.weights));
    }
}

impl LayerDropout {
    pub fn new(rate: f32, seed: u64) -> Self {
        LayerDropout {
            rate: 1.0 - rate,
            mask: Matrix::zeros(0, 0),
            rng: Rng::new(seed),
        }
    }
    pub fn forward(&mut self, inputs: &Matrix) -> Matrix {
        let (r,c)= (inputs.rows(), inputs.cols());
        self.mask = Matrix::new(inputs.data
            .iter()
            .map(|_| {
                if self.rng.next_f32() < self.rate as f32 {
                    1.0 / self.rate
                } else {
                    0.0
                }
            })
            .collect(), r, c);
        Matrix::new(inputs.data.clone()
            .iter()
            .zip(&self.mask.data)
            .map(|(i, m)| i*m)
            .collect(), r, c)
    }
    pub fn backward(&mut self, dvalues: &Matrix) -> Matrix {
        Matrix::new(dvalues.data
            .iter()
            .zip(&self.mask.data)
            .map(|(dv, m)| dv*m)
            .collect(), dvalues.rows(), dvalues.cols())
    }
}

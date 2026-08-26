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
    pub biases: Vec<f64>,
    pub dweights: Matrix,
    pub dbiases: Vec<f64>,
    pub v_weights: Matrix,
    pub v_biases: Vec<f64>,
    pub cache_weights: Matrix,
    pub cache_biases: Vec<f64>,
}

pub struct LayerDropout {
    pub rate: f64,
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
            inputs: Vec::new(),
            weights: randn_matrix(n_inputs, n_neurons, 0.1, rng),
            biases: vec![0.0; n_neurons],
            dweights: vec![vec![0.0; n_neurons]; n_inputs],
            dbiases: vec![0.0; n_neurons],
            v_weights: vec![vec![0.0; n_neurons]; n_inputs],
            v_biases: vec![0.0; n_neurons],
            cache_weights: vec![vec![0.0; n_neurons]; n_inputs],
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
    pub fn new(rate: f64, seed: u64) -> Self {
        LayerDropout {
            rate: 1.0 - rate,
            mask: Vec::new(),
            rng: Rng::new(seed),
        }
    }
    pub fn forward(&mut self, inputs: &Matrix) -> Matrix {
        self.mask = inputs
            .iter()
            .map(|row| {
                row.iter()
                    .map(|_| {
                        if self.rng.next_f32() < self.rate as f32 {
                            1.0 / self.rate
                        } else {
                            0.0
                        }
                    })
                    .collect()
            })
            .collect();
        inputs
            .iter()
            .zip(&self.mask)
            .map(|(r, m)| r.iter().zip(m).map(|(a, b)| a * b).collect())
            .collect()
    }
    pub fn backward(&mut self, dvalues: &Matrix) -> Matrix {
        dvalues
            .iter()
            .zip(&self.mask)
            .map(|(dv, m)| dv.iter().zip(m).map(|(dvi, m)| dvi * m).collect())
            .collect()
    }
}

use std::f64::consts::EULER_GAMMA;
use crate::matrix::{divide, row_sum};

use crate::matrix::Matrix;

#[derive(Default)]
pub struct SoftmaxLossCategoricalCrossEntropy {
    // loss: Loss,
}

fn softmax(inputs: &Matrix) -> Matrix {
    let exp_values: Matrix = inputs
        .iter()
        .map(|r| {
            let max = r.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            r.into_iter().map(|c| EULER_GAMMA.powf(c - max)).collect()
        })
        .collect();
    let probabilities = divide(&exp_values,&row_sum(&exp_values));
    probabilities
}

impl SoftmaxLossCategoricalCrossEntropy {
    pub fn calculate() {
        todo!();
    }
    pub fn forward(mut self, inputs: &Matrix) -> f64 {
        todo!();
    }
}

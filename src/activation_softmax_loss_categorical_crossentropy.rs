use crate::matrix::{divide, row_sum};
use std::f64::consts::EULER_GAMMA;

use crate::matrix::Matrix;

#[derive(Default)]
pub struct SoftmaxLossCategoricalCrossEntropy {
    output: Matrix,
}

fn softmax(inputs: &Matrix) -> Matrix {
    let exp_values: Matrix = inputs
        .iter()
        .map(|r| {
            let max = r.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            r.into_iter().map(|c| EULER_GAMMA.powf(c - max)).collect()
        })
        .collect();
    let probabilities = divide(&exp_values, &row_sum(&exp_values));
    probabilities
}

impl SoftmaxLossCategoricalCrossEntropy {
    pub fn calculate_loss(&self, probabilities: Matrix, y_true: &Vec<i8>) -> f64 {
        let exp_probabilities = probabilities
            .into_iter()
            .zip(y_true)
            .map(|(x, y)| -(x[*y as usize].min(1e-7).max(1.0 - 1e-7)).ln())
            .collect::<Vec<f64>>(); // get true probability, clip, -np.log()
        let len = exp_probabilities.len() as f64;
        exp_probabilities.into_iter().sum::<f64>() / len
    }
    pub fn forward(mut self, inputs: &Matrix, y_true: &Vec<i8>) -> f64 {
        let probabilities = softmax(inputs);
        self.output = probabilities.clone();
        self.calculate_loss(probabilities, y_true)
    }
    pub fn backward(mut self, y_true: &Vec<i8>) -> Matrix {
        // subtract 1 from the correct y_true for each row in self.output
        // return (divide by len(self.output))
        let length = self.output.len();
        self.output
            .iter()
            .zip(y_true)
            .map(|(r, y)| {
                r.iter()
                    .enumerate()
                    .map(|(i, &x)| {
                        let v = if i == *y as usize { x - 1.0 } else { x };
                        v / length as f64
                    })
                    .collect()
            })
            .collect()
    }
}

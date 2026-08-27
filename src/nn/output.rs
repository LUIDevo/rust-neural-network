use crate::math::matrix::{divide, row_sum};

use crate::math::matrix::Matrix;

pub enum Output {
    SoftmaxCCE(SoftmaxLossCategoricalCrossEntropy),
    LinearMSE(LinearMeanSquaredError),
}

pub enum Target {
    Sparse(Vec<usize>),
    Dense(Matrix),
}

impl Output {
    pub fn forward(&mut self, inputs: &Matrix, target: &Target) -> (f32, f32) {
        match (self, target) {
            (Output::SoftmaxCCE(o), Target::Sparse(y)) => o.forward(inputs, y),
            (Output::LinearMSE(o), Target::Dense(y)) => o.forward(inputs, y),
            _ => panic!("output head and target type mismatch"),
        }
    }
    pub fn backward(&mut self, target: &Target) -> Matrix {
        match (self, target) {
            (Output::SoftmaxCCE(o), Target::Sparse(y)) => o.backward(y),
            (Output::LinearMSE(o), Target::Dense(y)) => o.backward(y),
            _ => panic!("output head and target type mismatch"),
        }
    }
}

#[derive(Default)]
pub struct LinearMeanSquaredError {
    output: Matrix,
}

#[derive(Default)]
pub struct SoftmaxLossCategoricalCrossEntropy {
    output: Matrix,
}

fn softmax(inputs: &Matrix) -> Matrix {
    let exp_values: Matrix = inputs
        .iter()
        .map(|r| {
            let max = r.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            r.into_iter().map(|c| (c - max).exp()).collect()
        })
        .collect();
    divide(&exp_values, &row_sum(&exp_values))
}

impl SoftmaxLossCategoricalCrossEntropy {
    pub fn calculate_accuracy(&self, probabilities: &Matrix, y_true: &Vec<usize>) -> f64 {
        // find mean of percentage correct predictions
        let samples = probabilities.len();
        let correct = probabilities
            .into_iter()
            .zip(y_true)
            .filter(|(p, y)| {
                let pred = p
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i, _)| i)
                    .unwrap();
                pred == **y
            })
            .count();
        correct as f64 / samples as f64
    }
    pub fn calculate_loss(&self, probabilities: Matrix, y_true: &Vec<usize>) -> f32 {
        let exp_probabilities = probabilities
            .into_iter()
            .zip(y_true)
            .map(|(x, y)| -(x[*y].min(1.0 - 1e-7).max(1e-7)).ln())
            .collect::<Vec<f32>>(); // get true probability, clip, -np.log()
        let len = exp_probabilities.len() as f32;
        exp_probabilities.into_iter().sum::<f32>() / len
    }
    pub fn forward(&mut self, inputs: &Matrix, y_true: &Vec<usize>) -> (f32, f32) {
        let probabilities = softmax(inputs);
        self.output = probabilities.clone();
        (
            self.calculate_loss(probabilities.clone(), &y_true),
            self.calculate_accuracy(&probabilities, &y_true),
        )
    }
    pub fn backward(&mut self, y_true: &Vec<usize>) -> Matrix {
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
                        let v = if i == *y { x - 1.0 } else { x };
                        v / length as f32
                    })
                    .collect()
            })
            .collect()
    }
}

impl LinearMeanSquaredError {
    pub fn calculate_accuracy(&self, predictions: &Matrix, y_true: &Matrix) -> f64 {
        let flat: Vec<f32> = y_true.iter().flatten().copied().collect();
        let n = flat.len() as f32;
        let mean = flat.iter().sum::<f32>() / n;
        let std = (flat.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n).sqrt();
        let precision = std / 250.0;

        let mut total = 0.0;
        let mut correct = 0.0;
        for (p, y) in predictions.iter().zip(y_true) {
            for (pi, yi) in p.iter().zip(y) {
                total += 1.0;
                if (pi - yi).abs() < precision {
                    correct += 1.0;
                }
            }
        }
        correct / total
    }
    pub fn calculate_loss(&self, predictions: &Matrix, y_true: &Matrix) -> f32 {
        let samples = predictions.len() as f32;
        predictions
            .iter()
            .zip(y_true)
            .map(|(p, y)| {
                let outputs = p.len() as f32;
                p.iter()
                    .zip(y)
                    .map(|(pi, yi)| (pi - yi).powi(2))
                    .sum::<f32>()
                    / outputs
            })
            .sum::<f32>()
            / samples
    }
    pub fn forward(&mut self, inputs: &Matrix, y_true: &Matrix) -> (f64, f64) {
        self.output = inputs.clone();
        (
            self.calculate_loss(inputs, &y_true),
            self.calculate_accuracy(inputs, &y_true),
        )
    }
    pub fn backward(&mut self, y_true: &Matrix) -> Matrix {
        let samples = self.output.len() as f64;
        self.output
            .iter()
            .zip(y_true)
            .map(|(p, y)| {
                let outputs = p.len() as f64;
                p.iter()
                    .zip(y)
                    .map(|(pi, yi)| 2.0 * (pi - yi) / outputs / samples)
                    .collect()
            })
            .collect()
    }
}

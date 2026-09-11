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

fn softmax(inputs: &Matrix) -> Vec<f32> {
    let mut out = Vec::with_capacity(inputs.data.len());
    for row in inputs.data.chunks(inputs.cols()) {
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = row.iter().map(|c| (c - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        out.extend(exps.iter().map(|e| e / sum));
    }
    out
}

impl SoftmaxLossCategoricalCrossEntropy {
    pub fn calculate_accuracy(&self, probabilities: &Matrix, y_true: &Vec<usize>) -> f32 {
        // find mean of percentage correct predictions
        let mut count = 0;
        for (row, y) in probabilities.data.chunks(probabilities.cols()).zip(y_true) {
            let pred = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap();
            if pred == *y {
                count += 1;
            }
        }
        count as f32 / y_true.len() as f32
    }
    pub fn calculate_loss(&self, probabilities: &Matrix, y_true: &Vec<usize>) -> f32 {
        let mut sum: f32 = 0.0;
        for (i, row) in probabilities.data.chunks(probabilities.cols()).enumerate() {
            sum += -(row[y_true[i]].min(1.0 - 1e-7).max(1e-7)).ln()
        }
        sum / y_true.len() as f32
    }
    pub fn forward(&mut self, inputs: &Matrix, y_true: &Vec<usize>) -> (f32, f32) {
        let probabilities = softmax(inputs);
        self.output = Matrix::new(probabilities.clone(), inputs.rows(), inputs.cols());
        (
            self.calculate_loss(&self.output, &y_true),
            self.calculate_accuracy(&self.output, &y_true),
        )
    }
    pub fn backward(&mut self, y_true: &Vec<usize>) -> Matrix {
        // subtract 1 from the correct y_true for each row in self.output,
        // scale by 1 / batch size (rows), then wrap in one Matrix.
        let (rows, cols) = (self.output.rows(), self.output.cols());
        let samples = rows as f32;
        let mut data = Vec::with_capacity(self.output.data.len());
        for (ri, row) in self.output.data.chunks(cols).enumerate() {
            for (i, &x) in row.iter().enumerate() {
                let v = if i == y_true[ri] { x - 1.0 } else { x };
                data.push(v / samples);
            }
        }
        Matrix::new(data, rows, cols)
    }
}

impl LinearMeanSquaredError {
    pub fn calculate_accuracy(&self, predictions: &Matrix, y_true: &Matrix) -> f32 {
        let n = y_true.data.len() as f32;
        let mean = y_true.data.iter().sum::<f32>() / n;
        let std = (y_true.data.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n).sqrt();
        let precision = std / 250.0;

        let total = y_true.data.len() as f32;
        let correct = predictions
            .data
            .iter()
            .zip(&y_true.data)
            .filter(|(p, y)| (*p - *y).abs() < precision)
            .count() as f32;
        // for (p, y) in predictions.data.iter().zip(y_true) {
        //     for (pi, yi) in p.iter().zip(y) {
        //         total += 1.0;
        //         if (pi - yi).abs() < precision {
        //             correct += 1.0;
        //         }
        //     }
        // }
        correct / total
    }
    pub fn calculate_loss(&self, predictions: &Matrix, y_true: &Matrix) -> f32 {
        let samples = predictions.data.len() as f32;
        let mut sum: f32 = 0.0;
        for (x, row) in predictions.data.chunks(predictions.cols()).enumerate() {
            let cols = row.len() as f32;
            let mut sub_sum: f32 = 0.0;
            for (y, col) in row.iter().enumerate() {
                sub_sum += (col - y_true.data[x * (y_true.cols()) + y]).powi(2);
            }
            sum += sub_sum / cols;
        }
        sum / samples
        // predictions
        //     .iter()
        //     .zip(y_true)
        //     .map(|(p, y)| {
        //         let outputs = p.len() as f32;
        //         p.iter()
        //             .zip(y)
        //             .map(|(pi, yi)| (pi - yi).powi(2))
        //             .sum::<f32>()
        //             / outputs
        //     })
        //     .sum::<f32>()
        //     / samples
    }
    pub fn forward(&mut self, inputs: &Matrix, y_true: &Matrix) -> (f32, f32) {
        self.output = inputs.clone();
        (
            self.calculate_loss(inputs, &y_true),
            self.calculate_accuracy(inputs, &y_true),
        )
    }
    pub fn backward(&mut self, y_true: &Matrix) -> Matrix {
        let samples = self.output.data.len() as f32;
        let mut out = Vec::new();
        let row_length = self.output.cols();
        for (x, row) in self.output.data.chunks(self.output.cols()).enumerate() {
            for (y, item) in row.iter().enumerate() {
                out.push(
                    2.0 * (item - y_true.data[x * row_length + y]) / row_length as f32 / samples,
                )
            }
        }
        Matrix::new(out, y_true.rows(), y_true.cols())
        // self.output
        //     .iter()
        //     .zip(y_true)
        //     .map(|(p, y)| {
        //         let outputs = p.len() as f32;
        //         p.iter()
        //             .zip(y)
        //             .map(|(pi, yi)| 2.0 * (pi - yi) / outputs / samples)
        //             .collect()
        //     })
        //     .collect()
    }
}

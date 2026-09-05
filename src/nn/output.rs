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
        let exps: Vec<f32> = row.iter().map(|c| (c-max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        out.extend(exps.iter().map(|e| e/sum));
    }
    out
}

impl SoftmaxLossCategoricalCrossEntropy {
    pub fn calculate_accuracy(&self, probabilities: &Matrix, y_true: &Vec<usize>) -> f32 {
        // find mean of percentage correct predictions
        let mut count= 0;
        for (row, y) in probabilities.data.chunks(probabilities.cols()).zip(y_true) {
            let pred = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            if pred==*y as f32 { count+=1; }
        }
        count as f32/probabilities.data.len() as f32
        // let samples = probabilities.data.len();
        // let correct = probabilities.data
        //     .into_iter()
        //     .zip(y_true)
        //     .filter(|(p, y)| {
        //         let pred = p
        //             .iter()
        //             .enumerate()
        //             .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        //             .map(|(i, _)| i)
        //             .unwrap();
        //         pred == **y
        //     })
        //     .count();
        // correct as f32 / samples as f32
    }
    pub fn calculate_loss(&self, probabilities: Matrix, y_true: &Vec<usize>) -> f32 {
        let mut sum: f32=0.0;
        for (i,row) in probabilities.data.chunks(probabilities.cols()).enumerate() {
            sum+=-(row[y_true[i]].min(1.0 - 1e-7).max(1e-7)).ln()
        }
        // let exp_probabilities = probabilities.data
        //     .into_iter()
        //     .zip(y_true)
        //     .map(|(x, y)| -(x[*y].min(1.0 - 1e-7).max(1e-7)).ln())
        //     .collect::<Vec<f32>>(); // get true probability, clip, -np.log()
        sum / y_true.len()
    }
    pub fn forward(&mut self, inputs: &Matrix, y_true: &Vec<usize>) -> (f32, f32) {
        let probabilities = softmax(inputs.clone().data);
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
    pub fn calculate_accuracy(&self, predictions: &Matrix, y_true: &Matrix) -> f32 {
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
    pub fn forward(&mut self, inputs: &Matrix, y_true: &Matrix) -> (f32, f32) {
        self.output = inputs.clone();
        (
            self.calculate_loss(inputs, &y_true),
            self.calculate_accuracy(inputs, &y_true),
        )
    }
    pub fn backward(&mut self, y_true: &Matrix) -> Matrix {
        let samples = self.output.len() as f32;
        self.output
            .iter()
            .zip(y_true)
            .map(|(p, y)| {
                let outputs = p.len() as f32;
                p.iter()
                    .zip(y)
                    .map(|(pi, yi)| 2.0 * (pi - yi) / outputs / samples)
                    .collect()
            })
            .collect()
    }
}

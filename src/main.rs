mod data;
mod math;
mod nn;
#[cfg(test)]
mod tests;

// use std::process::Output;

use std::fs::{self, File};
use std::path::Path;

use crate::data::decode;
use crate::math::matrix::Matrix;
use crate::math::rng::Rng;
use crate::nn::activation::ActivationReLU;
use crate::nn::layer::{Layer, LayerDense, LayerDropout};
use crate::nn::optimiser::{AdaGrad, Adam, Optimiser, RMSProp, SGD};
use crate::nn::output::{LinearMeanSquaredError, Output, Target};
use crate::data::decode::decode_png;

fn create_dataset() -> (Matrix, Vec<f64>) {
    let mut x: Matrix = Vec::new();
    let mut y = Vec::new();

    let root = Path::new("fashion_mnist_images/train");
    for label in 0..10 {
        let class_dir = root.join(label.to_string());

        let mut paths: Vec<_> = fs::read_dir(&class_dir)
            .expect("read class dir")
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|ext| ext == "png"))
            .collect();
        paths.sort();

        for path in paths {
            let features = decode_png(path);
            x.push(features);
            y.push(label as f64);
        }
    }

    (x, y)
}

fn main() {
    let (x, y) = create_dataset();
    let (test_x, test_y) = data::sine::sine_data(100);
    let mut rng = Rng::new(0);
    // define layers, activation, loss function
    let mut layers: Vec<Layer> = vec![
        Layer::Dense(LayerDense::new(1, 64, &mut rng)),
        Layer::ReLU(ActivationReLU::default()),
        Layer::Dense(LayerDense::new(64, 64, &mut rng)),
        Layer::ReLU(ActivationReLU::default()),
        Layer::Dense(LayerDense::new(64, 1, &mut rng)),
    ];
    let mut output = Output::LinearMSE(LinearMeanSquaredError::default());
    let target = Target::Dense(y);
    let mut optimiser = Adam {
        lr: 0.01,
        moment_decay: 0.9,
        variance_decay: 0.999,
        lambda_reg: 0.001,
        iterations: 0,
    };
    let mut out;
    for iterations in 0..=1000 {
        optimiser.pre_update();
        // define forward pass
        out = x.clone();
        for layer in layers.iter_mut() {
            out = layer.forward(&out);
        }
        let (loss, accuracy) = output.forward(&out, &target);
        if iterations % 100 == 0 {
            println!(
                "Iteration: {}, Loss: {}, Accuracy: {}",
                iterations, loss, accuracy
            );
        }
        // define backward pass & update weights
        let mut grad = output.backward(&target);
        for layer in layers.iter_mut().rev() {
            grad = layer.backward(&grad);
        }
        for layer in layers.iter_mut() {
            if let Layer::Dense(dense) = layer {
                optimiser.update_params(dense);
            }
        }
    }
    out = test_x;
    for layer in layers.iter_mut() {
        match layer {
            Layer::Dropout(_) => (),
            _ => out = layer.forward(&out),
        }
    }
    let test_target = Target::Dense(test_y);
    let (test_loss, test_acc) = output.forward(&out, &test_target);
    println!("TEST  loss {:.3}  acc {:.3}", test_loss, test_acc);
}

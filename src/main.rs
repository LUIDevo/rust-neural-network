mod data;
mod math;
mod nn;
#[cfg(test)]
mod tests;

// use std::process::Output;

use std::fs::{self, File};
use std::path::Path;
use std::time::{Instant,Duration};

use crate::data::decode::{decode_png, shuffle_dataset};
use crate::data::fashion_mnist::prepare_dataset;
use crate::math::matrix::Matrix;
use crate::math::rng::Rng;
use crate::nn::activation::ActivationReLU;
use crate::nn::layer::{Layer, LayerDense, LayerDropout};
use crate::nn::optimiser::{AdaGrad, Adam, Optimiser, RMSProp, SGD};
use crate::nn::output::{
    LinearMeanSquaredError, Output, SoftmaxLossCategoricalCrossEntropy, Target,
};

const EPOCHS: usize = 10;
const BATCH_SIZE: usize = 128;

fn create_dataset(root: &Path, rng: &mut Rng) -> (Matrix, Vec<usize>) {
    let mut x: Matrix = Vec::new();
    let mut y = Vec::new();
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
            y.push(label as usize);
        }
    }
    shuffle_dataset(&mut x, &mut y, rng);
    (x, y)
}

fn main() {
    let mut rng = Rng::new(0);
    if !Path::new("fashion_mnist_images/").is_dir() {
        prepare_dataset().expect("prepare dataset");
    }
    let (mut x, mut y) = create_dataset(Path::new("fashion_mnist_images/train"), &mut rng);
    let (test_x, test_y) = create_dataset(Path::new("fashion_mnist_images/test"), &mut rng);
    let mut steps = x.len() / BATCH_SIZE;
    if steps * BATCH_SIZE < x.len() {
        steps += 1;
    }
    // define layers, activation, loss function
    let mut layers: Vec<Layer> = vec![
        Layer::Dense(LayerDense::new(784, 128, &mut rng)),
        Layer::ReLU(ActivationReLU::default()),
        Layer::Dense(LayerDense::new(128, 128, &mut rng)),
        Layer::ReLU(ActivationReLU::default()),
        Layer::Dense(LayerDense::new(128, 10, &mut rng)),
    ];
    let mut output = Output::SoftmaxCCE(SoftmaxLossCategoricalCrossEntropy::default());
    let mut optimiser = Adam {
        lr: 0.001,
        moment_decay: 0.9,
        variance_decay: 0.999,
        lambda_reg: 0.0,
        iterations: 0,
    };
    // training loop
    let mut start = Instant::now();
    for epoch in 0..EPOCHS {
        shuffle_dataset(&mut x, &mut y, &mut rng);
        let (mut ep_loss, mut ep_acc) = (0.0, 0.0);
        for (step, (bx, by)) in x.chunks(BATCH_SIZE).zip(y.chunks(BATCH_SIZE)).enumerate() {
            let target = Target::Sparse(by.to_vec());

            // forward pass
            let mut out = bx.to_vec();
            for layer in layers.iter_mut() {
                out = layer.forward(&out);
            }

            let (loss, acc) = output.forward(&out, &target);
            ep_loss += loss;
            ep_acc += acc;

            // backward pass
            let mut dvalues = output.backward(&target);
            for layer in layers.iter_mut().rev() {
                dvalues = layer.backward(&dvalues);
            }

            // optimiser update
            optimiser.pre_update();
            for layer in layers.iter_mut() {
                if let Layer::Dense(l) = layer {
                    optimiser.update_params(l);
                }
            }

            if step % 100 == 0 {
                println!("epoch {epoch} step {step}/{steps} loss {loss:.4} acc {acc:.4}");
            }
        }
        println!(
            "epoch {epoch}: loss {:.4} acc {:.4}",
            ep_loss / steps as f64,
            ep_acc / steps as f64
        );
    }
    println!("Training time: {:?}", start.elapsed());
    // test loop
    let (mut t_loss, mut t_acc) = (0.0, 0.0);
    let test_steps = test_x.len().div_ceil(BATCH_SIZE);

    start = Instant::now();
    for (bx, by) in test_x.chunks(BATCH_SIZE).zip(test_y.chunks(BATCH_SIZE)) {
        let mut out = bx.to_vec();

        for layer in layers.iter_mut() {
            out = layer.forward(&out);
        }

        let (loss, acc) = output.forward(&out, &Target::Sparse(by.to_vec()));
        t_loss += loss;
        t_acc += acc;
    }

    println!(
        "TEST loss {:.4} acc {:.4}",
        t_loss / test_steps as f64,
        t_acc / test_steps as f64
    );

    println!("Test time: {:?}", start.elapsed());
}

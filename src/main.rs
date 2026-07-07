mod activation;
mod activation_softmax_loss_categorical_crossentropy;
mod layer;
mod matrix;
mod optimiser;
mod rng;
mod sine;
mod spiral;
mod tests;
mod vertical;

use crate::optimiser::{AdaGrad, Adam, Optimiser, RMSProp, SGD};
use activation::ActivationReLU;
use activation_softmax_loss_categorical_crossentropy::SoftmaxLossCategoricalCrossEntropy;
use layer::{Layer, LayerDense, LayerDropout};
use rng::Rng;

fn create_dataset() -> (matrix::Matrix, Vec<usize>) {
    let (x, y) = spiral::spiral_data(100, 3, 42);
    println!("generated {} samples", x.len());
    (x, y)
}

fn main() {
    let (x, y) = create_dataset();
    let (test_x, test_y) = spiral::spiral_data(100, 3, 99);
    let mut rng = Rng::new(0);
    // define layers, activation, loss function
    let mut layers: Vec<Layer> = vec![
        Layer::Dense(LayerDense::new(2, 64, &mut rng)),
        Layer::ReLU(ActivationReLU::default()),
        Layer::Dropout(LayerDropout::new(0.0, 12345)),
        Layer::Dense(LayerDense::new(64, 3, &mut rng)),
    ];
    let mut output = SoftmaxLossCategoricalCrossEntropy::default();
    let mut optimiser = Adam {
        lr: 0.05,
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
        let (loss, accuracy) = output.forward(&out, &y);
        if iterations % 100 == 0 {
            println!(
                "Iteration: {}, Loss: {}, Accuracy: {}",
                iterations, loss, accuracy
            );
        }
        // define backward pass & update weights
        let mut grad = output.backward(&y);
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
    let (test_loss, test_acc) = output.forward(&out, &test_y);
    println!("TEST  loss {:.3}  acc {:.3}", test_loss, test_acc);
}

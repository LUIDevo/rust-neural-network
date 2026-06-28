mod activation;
mod activation_softmax_loss_categorical_crossentropy;
mod layer;
mod matrix;
mod optimiser;
mod rng;
mod spiral;
mod tests;
mod vertical;

use crate::optimiser::{AdaGrad, Adam, Optimiser, RMSProp, SGD};
use activation::ActivationReLU;
use activation_softmax_loss_categorical_crossentropy::SoftmaxLossCategoricalCrossEntropy;
use layer::LayerDense;
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
    let mut dense1 = LayerDense::new(2, 64, &mut rng);
    let mut activation1 = ActivationReLU::default();
    let mut dense2 = LayerDense::new(64, 3, &mut rng);
    let mut output = SoftmaxLossCategoricalCrossEntropy::default();
    let mut optimiser = Adam {
        lr: 0.05,
        moment_decay: 0.9,
        variance_decay: 0.999,
        iterations: 0,
    };

    for iterations in 0..=1000 {
        optimiser.pre_update();
        // define forward pass
        let dense1_output = dense1.forward(&x);
        let activation1_output = activation1.forward(&dense1_output);
        let dense2_output = dense2.forward(&activation1_output);
        let (loss, accuracy) = output.forward(&dense2_output, &y);
        if iterations % 100 == 0 {
            println!(
                "Iteration: {}, Loss: {}, Accuracy: {}",
                iterations, loss, accuracy
            );
        }
        // define backward pass & update weights
        let mut dinputs = output.backward(&y);
        dinputs = dense2.backward(&dinputs);
        dinputs = activation1.backward(&dinputs);
        dinputs = dense1.backward(&dinputs);
        optimiser.update_params(&mut dense2);
        optimiser.update_params(&mut dense1);
    }
    let a = dense1.forward(&test_x);
    let b = activation1.forward(&a);
    let c = dense2.forward(&b);
    let (test_loss, test_acc) = output.forward(&c, &test_y);
    println!("TEST  loss {:.3}  acc {:.3}", test_loss, test_acc);
}

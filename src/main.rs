mod activation;
mod activation_softmax_loss_categorical_crossentropy;
mod layer;
mod matrix;
mod optimiser;
mod rng;
mod tests;
mod vertical;

use crate::optimiser::{AdaGrad, Optimiser, RMSProp, SGD, Adam};
use activation::ActivationReLU;
use activation_softmax_loss_categorical_crossentropy::SoftmaxLossCategoricalCrossEntropy;
use layer::LayerDense;
use rng::Rng;

fn create_dataset() -> (matrix::Matrix, Vec<usize>) {
    let (x, y) = vertical::vertical_data(100, 3, 42);
    println!("generated {} samples", x.len());
    (x, y)
}

fn main() {
    let (x, y) = create_dataset();
    let mut rng = Rng::new(0);
    // define layers, activation, loss function
    let mut dense1 = LayerDense::new(2, 3, &mut rng);
    let mut activation1 = ActivationReLU::default();
    let mut dense2 = LayerDense::new(3, 3, &mut rng);
    let mut output = SoftmaxLossCategoricalCrossEntropy::default();
    let mut optimiser = Adam {
        lr: 0.2,
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
}

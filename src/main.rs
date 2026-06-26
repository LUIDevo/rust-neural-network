mod activation;
mod activation_softmax_loss_categorical_crossentropy;
mod layer;
mod matrix;
mod optimiser;
mod rng;
mod tests;
mod vertical;

use crate::optimiser::Optimiser;
use activation::ActivationReLU;
use activation_softmax_loss_categorical_crossentropy::SoftmaxLossCategoricalCrossEntropy;
use layer::LayerDense;
use rng::Rng;

fn create_dataset() -> Vec<vertical::Sample> {
    let data = vertical::vertical_data(100, 3, 42);
    println!("generated {} samples", data.len());
    data
}

fn main() {
    let data = create_dataset();
    let mut rng = Rng::new(0);
    // define layers, activation, loss function
    let mut dense1 = LayerDense::new(2, 3, &mut rng);
    let mut activation1 = ActivationReLU::default();
    let mut dense2 = LayerDense::new(3, 3, &mut rng);
    let mut output = SoftmaxLossCategoricalCrossEntropy::default();
    let mut optimiser = Optimiser::default();

    for _ in 0..=1000 {
        // define forward pass
        // get loss
        // define backward pass & update weights
    }
}

mod layer;
mod matrix;
mod rng;
mod tests;
mod activation;
mod vertical;

use layer::LayerDense;
use activation::ActivationReLU;
use rng::Rng;

fn create_dataset() -> Vec<vertical::Sample> {
    let data = vertical::vertical_data(100, 3, 42);
    println!("generated {} samples", data.len());
    data
}

fn main() {
    let data = create_dataset();
    let mut rng = Rng::new(0);
    // define layers
    let mut dense1 = LayerDense::new(2,3, &mut rng);
    // define activation function
    let mut activation1 = ActivationReLU::default();
    // define forward pass
    // get loss
    // define backward pass & update weights
}

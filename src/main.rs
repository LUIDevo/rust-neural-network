mod activation;
mod layer;
mod matrix;
mod rng;
mod tests;
mod vertical;

use activation::ActivationReLU;
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

    for _ in 0..=1001 {
        // define forward pass
        // get loss
        // define backward pass & update weights
    }
}

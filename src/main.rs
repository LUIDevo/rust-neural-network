mod layer;
mod matrix;
mod rng;
mod vertical;

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
    // define forward pass
}

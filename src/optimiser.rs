use crate::{layer::LayerDense, matrix::Matrix};

#[derive(Default)]
pub struct Optimiser {
    pub lr: f64,
}

impl Optimiser {
    pub fn update_params(&self, layer: &mut LayerDense) {
        layer.weights = layer
            .weights
            .iter()
            .zip(&layer.dweights)
            .map(|(w, dw)| {
                w.iter()
                    .zip(dw)
                    .map(|(&i, &dwi)| i - dwi * self.lr)
                    .collect()
            })
            .collect();
        layer.biases = layer
            .biases
            .iter()
            .zip(&layer.dbiases)
            .map(|(b, db)| b - db * self.lr)
            .collect();
    }
}

use crate::{layer::LayerDense, matrix::Matrix};

pub trait Optimiser {
    fn update_params(&self, layer: &mut LayerDense);
    fn pre_update(&mut self, iterations: u32);
}

#[derive(Default)]
pub struct SGD {
    pub lr: f64,
    pub lr_decay: f64,
    pub momentum: f64,
}

impl Optimiser for SGD {
    fn pre_update(&mut self, iterations: u32) {
        if iterations > 900 {
            self.lr -= self.lr_decay * self.lr
        }
    }
    fn update_params(&self, layer: &mut LayerDense) {
        layer.v_weights=layer.v_weights.iter().zip(&layer.dweights).map(|(vw,dw)| vw.iter().zip(dw).map(|(&vi,&di)| self.momentum*vi - self.lr*di).collect()).collect();
        layer.v_biases=layer.v_biases.iter().zip(&layer.dbiases).map(|(vb,db)| self.momentum*vb-self.lr*db).collect();
        layer.weights = layer
            .weights
            .iter()
            .zip(&layer.v_weights)
            .map(|(w, dv)| {
                w.iter()
                    .zip(dv)
                    .map(|(&i, &dvi)| i + dvi)
                    .collect()
            })
            .collect();
        layer.biases = layer
            .biases
            .iter()
            .zip(&layer.v_biases)
            .map(|(b, dv)| b + dv)
            .collect();
    }
}

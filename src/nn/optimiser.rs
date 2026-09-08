use crate::math::matrix::Matrix;
use crate::nn::layer::LayerDense;

pub trait Optimiser {
    fn update_params(&self, layer: &mut LayerDense);
}

pub struct SGD {
    pub lr: f32,
    pub lr_decay: f32,
    pub momentum: f32,
}

pub struct AdaGrad {
    pub lr: f32,
}

pub struct RMSProp {
    pub lr: f32,
    pub lr_decay: f32,
}

pub struct Adam {
    pub lr: f32,
    pub moment_decay: f32,
    pub variance_decay: f32,
    pub lambda_reg: f32,
    pub iterations: i32,
}

impl Adam {
    pub fn pre_update(&mut self) {
        self.iterations += 1;
    }
}

impl Optimiser for Adam {
    // technically AdamW now
    fn update_params(&self, layer: &mut LayerDense) {
        layer.v_weights = layer
            .v_weights
            .iter()
            .zip(&layer.dweights)
            .map(|(&vw, dw)| {
                    self.moment_decay * vw + (1.0 - self.moment_decay) * dw
            })
            .collect();
        layer.v_biases = layer
            .v_biases
            .iter()
            .zip(&layer.dbiases)
            .map(|(vb, db)| self.moment_decay * vb + (1.0 - self.moment_decay) * db)
            .collect();
        layer.cache_weights = layer
            .cache_weights
            .iter()
            .zip(&layer.dweights)
            .map(|(cw, dw)| self.variance_decay * cw + (1.0 - self.variance_decay) * dw.powi(2) )
            .collect();
        layer.cache_biases = layer
            .cache_biases
            .iter()
            .zip(&layer.dbiases)
            .map(|(cb, db)| cb * self.variance_decay + (1.0 - self.variance_decay) * db.powi(2))
            .collect();
        let vw_hat: Vec<f32> = layer
            .v_weights
            .iter()
            .map(|vw| vw / (1.0 - (self.moment_decay).powi(self.iterations)))
            .collect();
        let vb_hat: Vec<f32> = layer
            .v_biases
            .iter()
            .map(|vb| vb / (1.0 - (self.moment_decay).powi(self.iterations)))
            .collect();
        let cw_hat: Vec<f32> = layer
            .cache_weights
            .iter()
            .map(|cw| cw / (1.0 - (self.variance_decay).powi(self.iterations)))
            .collect();
        let cb_hat: Vec<f32> = layer
            .cache_biases
            .iter()
            .map(|cb| cb / (1.0 - (self.variance_decay).powi(self.iterations)))
            .collect();
        for i in 0..layer.weights.data.len() { 
            layer.weights.data[i]=layer.weights.data[i] - vw_hat[i] * self.lr / (cw_hat[i].sqrt() + 1e-7) - self.lambda_reg * layer.weights.data[i];
        }
        // layer.weights = layer
        //     .weights.data
        //     .iter_mut()
        //     .zip(vw_hat)
        //     .zip(cw_hat)
        //     .map(|((w, vwh), cwh): ((&f32, f32), f32)| w - vwh * self.lr / (cwh.sqrt() + 1e-7) - self.lambda_reg * w)
        //     .collect();
        layer.biases = layer
            .biases
            .iter()
            .zip(vb_hat)
            .zip(cb_hat)
            .map(|((b, vbh), cbh)| b - vbh * self.lr / (cbh.sqrt() + 1e-7))
            .collect();
    }
}

impl Optimiser for RMSProp {
    fn update_params(&self, layer: &mut LayerDense) {
        layer.cache_weights = layer
            .cache_weights
            .iter()
            .zip(&layer.dweights)
            .map(|(cw, dw)| self.lr_decay * cw + (1.0 - self.lr_decay) * dw.powi(2))
            .collect();
        layer.cache_biases = layer
            .cache_biases
            .iter()
            .zip(&layer.dbiases)
            .map(|(cb, db)| cb * self.lr_decay + (1.0 - self.lr_decay) * db.powi(2))
            .collect();
        // layer.weights = layer
        //     .weights
        //     .iter()
        //     .zip(&layer.dweights)
        //     .zip(&layer.cache_weights)
        //     .map(|((w, dw), cw)| {
        //         w.iter()
        //             .zip(dw)
        //             .zip(cw)
        //             .map(|((&wi, &dwi), &cwi)| wi - dwi * self.lr / (cwi.sqrt() + 1e-7))
        //             .collect()
        //     })
        //     .collect();
        for i in 0..layer.weights.data.len() { 
            layer.weights.data[i]=layer.weights.data[i] - layer.dweights[i] * self.lr / (layer.cache_weights[i].sqrt() + 1e-7);
        }
        layer.biases = layer
            .biases
            .iter()
            .zip(&layer.dbiases)
            .zip(&layer.cache_biases)
            .map(|((b, db), cb)| b - db * self.lr / (cb.sqrt() + 1e-7))
            .collect();
    }
}

impl Optimiser for AdaGrad {
    fn update_params(&self, layer: &mut LayerDense) {
        layer.cache_weights = layer
            .cache_weights
            .iter()
            .zip(&layer.dweights)
            .map(|(cw, dw)| cw + dw.powi(2))
            .collect();
        layer.cache_biases = layer
            .cache_biases
            .iter()
            .zip(&layer.dbiases)
            .map(|(cb, db)| cb + db.powi(2))
            .collect();

        for i in 0..layer.weights.data.len() { 
            layer.weights.data[i]=layer.weights.data[i] - layer.dweights[i] * self.lr / (layer.cache_weights[i].sqrt() + 1e-7);
        }
        // layer.weights = layer
        //     .weights.data
        //     .iter()
        //     .zip(&layer.dweights)
        //     .zip(&layer.cache_weights)
        //     .map(|((w, dw), cw)| {
        //         w.iter()
        //             .zip(dw)
        //             .zip(cw)
        //             .map(|((&wi, &dwi), &cwi)| wi - dwi * self.lr / (cwi.sqrt() + 1e-7))
        //             .collect()
        //     })
        //     .collect();
        layer.biases = layer
            .biases
            .iter()
            .zip(&layer.dbiases)
            .zip(&layer.cache_biases)
            .map(|((b, db), cb)| b - db * self.lr / (cb.sqrt() + 1e-7))
            .collect();
    }
}

impl SGD {
    fn pre_update(&mut self, iterations: u32) {
        if iterations > 900 {
            self.lr -= self.lr_decay * self.lr
        }
    }
}

impl Optimiser for SGD {
    fn update_params(&self, layer: &mut LayerDense) {
        layer.v_weights = layer
            .v_weights
            .iter()
            .zip(&layer.dweights)
            .map(|(vw, dw)| self.momentum * vw - self.lr * dw)
            .collect();
        layer.v_biases = layer
            .v_biases
            .iter()
            .zip(&layer.dbiases)
            .map(|(vb, db)| self.momentum * vb - self.lr * db)
            .collect();
        for i in 0..layer.weights.data.len() { 
            layer.weights.data[i]=layer.weights.data[i] + layer.v_weights[i];
        }
        // layer.weights = layer
        //     .weights
        //     .iter()
        //     .zip(&layer.v_weights)
        //     .map(|(w, dv)| w.iter().zip(dv).map(|(&wi, &dvi)| wi + dvi).collect())
        //     .collect();
        layer.biases = layer
            .biases
            .iter()
            .zip(&layer.v_biases)
            .map(|(b, dv)| b + dv)
            .collect();
    }
}

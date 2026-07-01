use crate::layer::LayerDense;
use crate::matrix::Matrix;

pub trait Optimiser {
    fn update_params(&self, layer: &mut LayerDense);
}

pub struct SGD {
    pub lr: f64,
    pub lr_decay: f64,
    pub momentum: f64,
}

pub struct AdaGrad {
    pub lr: f64,
}

pub struct RMSProp {
    pub lr: f64,
    pub lr_decay: f64,
}

pub struct Adam {
    pub lr: f64,
    pub moment_decay: f64,
    pub variance_decay: f64,
    pub lambda_reg: f64,
    pub iterations: i32,
}

impl Adam {
    pub fn pre_update(&mut self) {
        self.iterations += 1;
    }
}

impl Optimiser for Adam { // technically AdamW now
    fn update_params(&self, layer: &mut LayerDense) {
        layer.v_weights = layer
            .v_weights
            .iter()
            .zip(&layer.dweights)
            .map(|(vw, dw)| {
                vw.iter()
                    .zip(dw)
                    .map(|(&vwi, &dwi)| self.moment_decay * vwi + (1.0 - self.moment_decay) * dwi)
                    .collect()
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
            .map(|(cw, dw)| {
                cw.iter()
                    .zip(dw)
                    .map(|(&cwi, &dwi)| {
                        self.variance_decay * cwi + (1.0 - self.variance_decay) * dwi.powi(2)
                    })
                    .collect()
            })
            .collect();
        layer.cache_biases = layer
            .cache_biases
            .iter()
            .zip(&layer.dbiases)
            .map(|(cb, db)| cb * self.variance_decay + (1.0 - self.variance_decay) * db.powi(2))
            .collect();
        let vw_hat: Matrix = layer
            .v_weights
            .iter()
            .map(|vw| {
                vw.iter()
                    .map(|vwi| vwi / (1.0 - (self.moment_decay).powi(self.iterations)))
                    .collect()
            })
            .collect();
        let vb_hat: Vec<f64> = layer
            .v_biases
            .iter()
            .map(|vb| vb / (1.0 - (self.moment_decay).powi(self.iterations)))
            .collect();
        let cw_hat: Matrix = layer
            .cache_weights
            .iter()
            .map(|cw| {
                cw.iter()
                    .map(|cwi| cwi / (1.0 - (self.variance_decay).powi(self.iterations)))
                    .collect()
            })
            .collect();
        let cb_hat: Vec<f64> = layer
            .cache_biases
            .iter()
            .map(|cb| cb / (1.0 - (self.variance_decay).powi(self.iterations)))
            .collect();
        layer.weights = layer
            .weights
            .iter()
            .zip(vw_hat)
            .zip(cw_hat)
            .map(|((w, vwh), cwh)| {
                w.iter()
                    .zip(vwh)
                    .zip(cwh)
                    .map(|((wi, vwhi), cwhi)| wi - vwhi * self.lr / (cwhi.sqrt() + 1e-7) - self.lambda_reg * wi)
                    .collect()
            })
            .collect();
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
            .map(|(cw, dw)| {
                cw.iter()
                    .zip(dw)
                    .map(|(&cwi, &dwi)| self.lr_decay * cwi + (1.0 - self.lr_decay) * dwi.powi(2))
                    .collect()
            })
            .collect();
        layer.cache_biases = layer
            .cache_biases
            .iter()
            .zip(&layer.dbiases)
            .map(|(cb, db)| cb * self.lr_decay + (1.0 - self.lr_decay) * db.powi(2))
            .collect();
        layer.weights = layer
            .weights
            .iter()
            .zip(&layer.dweights)
            .zip(&layer.cache_weights)
            .map(|((w, dw), cw)| {
                w.iter()
                    .zip(dw)
                    .zip(cw)
                    .map(|((&wi, &dwi), &cwi)| wi - dwi * self.lr / (cwi.sqrt() + 1e-7))
                    .collect()
            })
            .collect();
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
            .map(|(cw, dw)| {
                cw.iter()
                    .zip(dw)
                    .map(|(&cwi, &dwi)| cwi + dwi.powi(2))
                    .collect()
            })
            .collect();
        layer.cache_biases = layer
            .cache_biases
            .iter()
            .zip(&layer.dbiases)
            .map(|(cb, db)| cb + db.powi(2))
            .collect();
        layer.weights = layer
            .weights
            .iter()
            .zip(&layer.dweights)
            .zip(&layer.cache_weights)
            .map(|((w, dw), cw)| {
                w.iter()
                    .zip(dw)
                    .zip(cw)
                    .map(|((&wi, &dwi), &cwi)| wi - dwi * self.lr / (cwi.sqrt() + 1e-7))
                    .collect()
            })
            .collect();
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
            .map(|(vw, dw)| {
                vw.iter()
                    .zip(dw)
                    .map(|(&vwi, &dwi)| self.momentum * vwi - self.lr * dwi)
                    .collect()
            })
            .collect();
        layer.v_biases = layer
            .v_biases
            .iter()
            .zip(&layer.dbiases)
            .map(|(vb, db)| self.momentum * vb - self.lr * db)
            .collect();
        layer.weights = layer
            .weights
            .iter()
            .zip(&layer.v_weights)
            .map(|(w, dv)| w.iter().zip(dv).map(|(&wi, &dvi)| wi + dvi).collect())
            .collect();
        layer.biases = layer
            .biases
            .iter()
            .zip(&layer.v_biases)
            .map(|(b, dv)| b + dv)
            .collect();
    }
}

use crate::layer::LayerDense;

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
            .map(|(cb, db)| cb*self.lr_decay + (1.0-self.lr_decay)*db.powi(2))
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

use crate::matrix::Matrix;

#[derive(Default)]
pub struct ActivationReLU {
    pub inputs: Matrix,
    pub output: Matrix,
    pub dinputs: Matrix,
}

impl ActivationReLU {
    pub fn forward(inputs: &Matrix)->Matrix{ 
        inputs.iter().map(|x| x.into_iter().map(|y| { if *y<0.0 { 0.0 } else { *y } } ).collect()).collect()
    }
    pub fn backward(){ 
        todo!()
    }
}

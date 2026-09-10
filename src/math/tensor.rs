use crate::math::rng::Rng;

#[derive(Clone, Debug, PartialEq, Default)]
pub struct Tensor {
    pub data: Vec<f32>,
    n: usize, // batch
    c: usize, // channels
    h: usize, // height
    w: usize, // width
}

impl Tensor {
    fn new(data: Vec<f32>, n:usize, c: usize, h: usize, w:usize) -> Self {
        assert_eq!(data.len(), n*c*h*w);
        Tensor { data, n, c, h, w}    
    }
    pub fn zeros(n: usize, c: usize, h:usize, w:usize) -> Self { 
        Tensor { vec![0.0 as f32; n*c*h*w], n, c, h, w } 
    }
}

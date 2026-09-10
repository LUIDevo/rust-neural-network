use crate::math::rng::Rng;

#[derive(Clone, Debug, PartialEq, Default)]
pub struct Tensor {
    pub data: Vec<f32>,
    n: usize, // batch
    c: usize, // channels
    h: usize, // height
    w: usize, // width
}

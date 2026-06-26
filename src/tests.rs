use crate::activation::ActivationReLU;
use crate::matrix::dot;
use crate::matrix::transpose;

#[test]
fn dot_works() {
    let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
    let b = vec![vec![5.0, 6.0], vec![7.0, 8.0]];
    assert_eq!(dot(&a, &b), vec![vec![19.0, 22.0], vec![43.0, 50.0]]);
}

#[test]
fn relu_works() {
    let a = vec![vec![-1.0, 2.0], vec![-3.0, 4.0]];
    let activation = ActivationReLU::default();
    assert_eq!(activation.forward(&a), vec![vec![0.0, 2.0], vec![0.0, 4.0]]);
}

#[test]
fn transpose_works() {
    let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
    assert_eq!(transpose(&a), vec![vec![1.0, 3.0], vec![2.0, 4.0]]);
}

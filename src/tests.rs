use crate::math::matrix::{dot, transpose, Matrix};
use crate::nn::output::{LinearMeanSquaredError, Output, Target};

#[test]
fn dot_works() {
    let a = Matrix::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
    let b = Matrix::new(vec![5.0, 6.0, 7.0, 8.0], 2, 2);
    let expected = Matrix::new(vec![19.0, 22.0, 43.0, 50.0], 2, 2);
    assert_eq!(dot(&a, &b), expected);
}

#[test]
fn transpose_works() {
    let a = Matrix::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
    let expected = Matrix::new(vec![1.0, 3.0, 2.0, 4.0], 2, 2);
    assert_eq!(transpose(&a), expected);
}

#[test]
fn Linear_MSE_accuracy_works() {
    let a = Matrix::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
    let b = Matrix::new(vec![5.0, 6.0, 7.0, 8.0], 2, 2);
    let t = Target::Dense(b);
    let mut output = Output::LinearMSE(LinearMeanSquaredError::default());
    let result = output.forward(&a, &t);
    println!("{}, {}", result.0, result.1); // test earlier implementation vs after
    // assert_eq!(output.forward(&a,&t), );
}

#[test]
fn Linear_MSE_backward_pass() {
    let a = Matrix::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
    let b = Matrix::new(vec![5.0, 6.0, 7.0, 8.0], 2, 2);
    let t = Target::Dense(b);
    let mut output = Output::LinearMSE(LinearMeanSquaredError::default());
    let result = output.backward(&t);
    println!("{:?}", result.data); // test earlier implementation vs after
    // assert_eq!(output.forward(&a,&t), );
}

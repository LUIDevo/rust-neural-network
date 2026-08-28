use crate::math::matrix::{dot, transpose, Matrix};

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

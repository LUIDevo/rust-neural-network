use crate::matrix::dot;
#[test]
fn dot_works() {
    let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
    let b = vec![vec![5.0, 6.0], vec![7.0, 8.0]];
    assert_eq!(dot(&a, &b), vec![vec![19.0, 22.0], vec![43.0, 50.0]]);
}

mod layers {
    use math::linearalg::{Tensor};
    pub trait Layer<T> {
        fn call(input: &Tensor<T>) -> Tensor<T>;
        fn backprop(input: &Tensor<T>) -> Tensor<T>;
    }
}
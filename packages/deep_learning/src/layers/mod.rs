
use math::linearalg::{Tensor};

mod convolution;
pub use convolution::Conv;

mod dense;
pub use dense::Dense;

pub struct LayerGradients<T> {
    weights: Tensor<T>,
    bias: Tensor<T>,
    input: Tensor<T>,
}

pub trait Layer<T> {
    fn call(&self, input: &Tensor<T>) -> Tensor<T>;
    fn backprop(&self, input: &Tensor<T>, output_gradient: &Tensor<T>,) -> LayerGradients<T>;
}
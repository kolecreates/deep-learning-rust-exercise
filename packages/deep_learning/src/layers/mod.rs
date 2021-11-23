
use math::linearalg::{Tensor};

mod convolution;
mod dense;
mod maxpool;

pub use convolution::Conv;
pub use dense::Dense;
pub use maxpool::MaxPool;



pub struct LayerGradients<T> {
    weights: Tensor<T>,
    bias: Tensor<T>,
    input: Tensor<T>,
}

pub trait Layer<T> {
    fn call(&self, input: &Tensor<T>) -> Tensor<T>;
    fn backprop(&self, input: &Tensor<T>, output_gradient: &Tensor<T>,) -> LayerGradients<T>;
}
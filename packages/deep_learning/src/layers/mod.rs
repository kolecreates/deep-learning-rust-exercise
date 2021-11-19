
use math::linearalg::{Tensor};

mod convolution;
pub use convolution::ConvLayer;

pub trait Layer<T> {
    fn call(&self, input: &Tensor<T>) -> Tensor<T>;
    fn backprop(&self, input: &Tensor<T>) -> Tensor<T>;
}
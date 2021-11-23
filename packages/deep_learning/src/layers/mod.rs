
use math::linearalg::{Tensor};

mod convolution;
mod dense;
mod maxpool;

pub use convolution::Conv;
pub use dense::Dense;
pub use maxpool::MaxPool;

use crate::optimizers::LayerLossGradients;

pub trait Layer<T> {
    fn call(&self, input: &Tensor<T>) -> Tensor<T>;
    fn backprop(&self, input: &Tensor<T>, output_gradient: &Tensor<T>,) -> (Option<LayerLossGradients<T>>, Tensor<T>);
}
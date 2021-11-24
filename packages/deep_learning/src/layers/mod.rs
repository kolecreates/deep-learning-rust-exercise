
use math::linearalg::{Tensor};

mod convolution;
mod dense;
mod maxpool;
pub mod activations;

pub use convolution::Conv;
pub use dense::Dense;
pub use maxpool::MaxPool;

use crate::optimizers::{LayerLossGradients, LayerOptimizations};

pub trait Layer<T> {
    fn call(&self, input: &Tensor<T>) -> Tensor<T>;
    fn backprop(&self, input: &Tensor<T>, output_gradient: &Tensor<T>,) -> (Option<LayerLossGradients<T>>, Option<Tensor<T>>);
    fn get_state(&mut self) -> Option<&mut dyn LayerState<T>>;
}

pub trait LayerState<T> {
    fn update(&mut self, optimization: &LayerOptimizations<T>);
}

pub struct StandardLayerState<T> {
    pub weights: Tensor<T>,
    pub bias: Tensor<T>,
}

impl LayerState<f32> for StandardLayerState<f32> {
    fn update(&mut self, optimization: &LayerOptimizations<f32>) {
        self.bias = self.bias.add(&optimization.bias);
        self.weights = self.weights.add(&optimization.weights);
    }
}

use math::linearalg::{Tensor};

mod convolution;
mod dense;
mod maxpool;
mod flatten;
pub mod activations;

pub use convolution::Conv;
pub use dense::Dense;
pub use maxpool::MaxPool;
pub use flatten::Flatten;

use crate::{initializers::Initializer, optimizers::{LayerLossGradients, LayerOptimizations}};

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

impl<T> StandardLayerState<T> {
    fn create(shape: &Vec<usize>, weight_initializer: &dyn Initializer<T>, bias_initializer: &dyn Initializer<T>) -> StandardLayerState<T> {
        StandardLayerState { weights: weight_initializer.initialize(shape), bias: bias_initializer.initialize(&vec![shape[0], 1])  }
    }
}

impl LayerState<f32> for StandardLayerState<f32> {
    fn update(&mut self, optimization: &LayerOptimizations<f32>) {
        self.bias = self.bias.add(&optimization.bias);
        self.weights = self.weights.add(&optimization.weights);
    }
}
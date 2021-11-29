
mod convolution;
mod dense;
mod maxpool;
mod flatten;
pub mod activations;

pub use convolution::Conv;
pub use dense::Dense;
pub use maxpool::MaxPool;
pub use flatten::Flatten;
use ndarray::{ArrayD, ArrayViewD};

use crate::{initializers::Initializer, optimizers::{LayerLossGradients, LayerOptimizations}};

pub trait Layer<T> {
    fn call(&self, input: &ArrayViewD<T>) -> ArrayD<T>;
    fn backprop(&self, input: &ArrayViewD<T>, output_gradient: &ArrayViewD<T>,) -> (Option<LayerLossGradients<T>>, Option<ArrayD<T>>);
    fn get_state(&mut self) -> Option<&mut dyn LayerState<T>>;
}

pub trait LayerState<T> {
    fn update(&mut self, optimization: &LayerOptimizations<T>);
}

pub struct StandardLayerState<T> {
    pub weights: ArrayD<T>,
    pub bias: ArrayD<T>,
}

impl<T> StandardLayerState<T> {
    fn create(shape: &Vec<usize>, weight_initializer: &dyn Initializer<T>, bias_initializer: &dyn Initializer<T>) -> StandardLayerState<T> {
        StandardLayerState { weights: weight_initializer.initialize(shape), bias: bias_initializer.initialize(&vec![shape[0], 1])  }
    }
}

impl LayerState<f32> for StandardLayerState<f32> {
    fn update(&mut self, optimization: &LayerOptimizations<f32>) {
        self.bias = &self.bias + &optimization.bias;
        self.weights = &self.weights + &optimization.weights;
    }
}
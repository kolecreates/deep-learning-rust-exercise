use ndarray::{ArrayD, ArrayViewD, Dimension, Ix2};

use crate::{initializers::Initializer, optimizers::LayerLossGradients};

use super::{Layer, LayerState, StandardLayerState};

pub struct Dense<D> {
    state: StandardLayerState<f32>,
}

impl<D> Dense<D> {
    pub fn create(shape: &Vec<usize>, weight_initializer: &dyn Initializer<f32>, bias_initializer: &dyn Initializer<f32>) -> Dense<D> {
        let state = StandardLayerState::create(shape, weight_initializer, bias_initializer);
        Dense { state }
    }
}

impl<D: Dimension> Layer<f32> for Dense<D> {
    fn call(&self, input: &ArrayViewD<f32>) -> ArrayD<f32> {
        self.state.weights.into_dimensionality::<Ix2>().unwrap().dot(&input).add(&self.state.bias)
    }

    fn backprop(&self, input: &ArrayViewD<f32>, output_gradient: &ArrayViewD<f32>, ) -> (Option<LayerLossGradients<f32>>, Option<ArrayD<f32>>) {
        let input_gradient = self.state.weights.transpose().dot(&output_gradient);
        (Some(LayerLossGradients {
            weights: output_gradient.dot(&input.transpose()),
            bias: output_gradient.sum_last_axis(),
        }), Some(input_gradient))
    }

    fn get_state(&mut self) -> Option<&mut dyn LayerState<f32>> {
        Some(&mut self.state)
    }
}
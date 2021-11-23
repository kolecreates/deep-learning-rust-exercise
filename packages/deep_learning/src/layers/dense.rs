use math::linearalg::Tensor;

use crate::optimizers::LayerLossGradients;

use super::{Layer, LayerState, StandardLayerState};

pub struct Dense {
    state: StandardLayerState<f32>,
}

impl Layer<f32> for Dense {
    fn call(&self, input: &Tensor<f32>) -> Tensor<f32> {
        self.state.weights.dot(&input).add(&self.state.bias)
    }

    fn backprop(&self, input: &Tensor<f32>, output_gradient: &Tensor<f32>, ) -> (Option<LayerLossGradients<f32>>, Tensor<f32>) {
        (Some(LayerLossGradients {
            weights: output_gradient.dot(&input.transpose()),
            bias: output_gradient.sum_last_axis(),
        }), self.state.weights.transpose().dot(&output_gradient))


    }

    fn get_state(&mut self) -> Option<&mut dyn LayerState<f32>> {
        Some(&mut self.state)
    }
}
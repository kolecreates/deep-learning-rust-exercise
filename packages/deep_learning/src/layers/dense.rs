use math::linearalg::Tensor;

use crate::optimizers::LayerLossGradients;

use super::{Layer};

pub struct Dense {
    weights: Tensor<f32>,
    bias: Tensor<f32>,
}

impl Layer<f32> for Dense {
    fn call(&self, input: &Tensor<f32>) -> Tensor<f32> {
        self.weights.dot(&input).add(&self.bias)
    }

    fn backprop(&self, input: &Tensor<f32>, output_gradient: &Tensor<f32>, ) -> (Option<LayerLossGradients<f32>>, Tensor<f32>) {
        (Some(LayerLossGradients {
            weights: output_gradient.dot(&input.transpose()),
            bias: output_gradient.sum_last_axis(),
        }), self.weights.transpose().dot(&output_gradient))


    }
}
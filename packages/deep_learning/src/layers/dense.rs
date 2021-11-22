use math::linearalg::Tensor;

use super::{Layer, LayerGradients};

pub struct Dense {
    weights: Tensor<f32>,
    bias: Tensor<f32>,
}

impl Layer<f32> for Dense {
    fn call(&self, input: &Tensor<f32>) -> Tensor<f32> {
        self.weights.dot(&input).add(&self.bias)
    }

    fn backprop(&self, input: &Tensor<f32>, output_gradient: &Tensor<f32>, ) -> LayerGradients<f32> {
        LayerGradients {
            weights: output_gradient.dot(&input.transpose()),
            bias: output_gradient.sum_last_axis(),
            input: self.weights.transpose().dot(&output_gradient)
        }
    }
}
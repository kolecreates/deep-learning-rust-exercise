use math::linearalg::{Tensor, tensor_exp};

use super::Layer;

pub struct Softmax;

impl Layer<f32> for Softmax {
    fn call(&self, input: &Tensor<f32>) -> Tensor<f32> {
        let output = tensor_exp(&input);
        output.scalar_divide(output.sum())
    }

    fn backprop(&self, _input: &Tensor<f32>, _output_gradient: &Tensor<f32>,) -> (Option<crate::optimizers::LayerLossGradients<f32>>, Option<Tensor<f32>>) {
        (None, None)
    }

    fn get_state(&mut self) -> Option<&mut dyn super::LayerState<f32>> {
        None
    }
}

pub struct ReLU;

impl ReLU {
    fn clamp(to_copy: &Tensor<f32>, to_check: &Tensor<f32>) -> Tensor<f32> {
        let mut out = Tensor::from_shape(to_copy.shape.clone(), 0f32);
        for i in 0..out.data.len() {
            if to_check.data[i] < 0.0 {
                out.data[i] = 0.0;
            }else{
                out.data[i] = to_copy.data[i];
            }
        }

        out
    }
}

impl Layer<f32> for ReLU {
    fn call(&self, input: &Tensor<f32>) -> Tensor<f32> {
        ReLU::clamp(&input, &input)
    }

    fn backprop(&self, input: &Tensor<f32>, output_gradient: &Tensor<f32>,) -> (Option<crate::optimizers::LayerLossGradients<f32>>, Option<Tensor<f32>>) {
        (None, Some(ReLU::clamp(input, output_gradient)))
    }

    fn get_state(&mut self) -> Option<&mut dyn super::LayerState<f32>> {
        None
    }
}
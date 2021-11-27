use math::linearalg::{Tensor, print_vec, tensor_exp};

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
        let out_shape = Tensor::get_broadcasted_shape(to_copy, to_check);
        let indices = Tensor::get_indices_for_broadcasting(&out_shape, to_copy, to_check);
        let mut out = Tensor::from_shape(out_shape, 0f32);
        for i in 0..indices.len() {
            let (to_copy_index, to_check_index) = indices[i];
            if to_check.data[to_check_index] < 0.0 {
                out.data[i] = 0.0;
            }else{
                out.data[i] = to_copy.data[to_copy_index];
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
        (None, Some(ReLU::clamp(output_gradient, input)))
    }

    fn get_state(&mut self) -> Option<&mut dyn super::LayerState<f32>> {
        None
    }
}
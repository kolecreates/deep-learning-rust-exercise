use ndarray::{ArrayD, ArrayViewD, Zip};

use super::Layer;

pub struct Softmax;

impl Layer<f32> for Softmax {
    fn call(&self, input: &ArrayViewD<f32>) -> ArrayD<f32> {
        let output = input.map(|x| x.exp());
        let sum = output.sum();
        output / sum
    }

    fn backprop(&self, _input: &ArrayViewD<f32>, _output_gradient: &ArrayViewD<f32>,) -> (Option<crate::optimizers::LayerLossGradients<f32>>, Option<ArrayD<f32>>) {
        (None, None)
    }

    fn get_state(&mut self) -> Option<&mut dyn super::LayerState<f32>> {
        None
    }
}

pub struct ReLU;

impl ReLU {
    fn clamp(to_copy: &ArrayViewD<f32>, to_check: &ArrayViewD<f32>) -> ArrayD<f32> {
        let mut out = to_copy.to_owned();

        Zip::from(&mut out).and(to_check).for_each(|out, to_check|{
            if to_check < &0f32 {
                *out = 0f32;
            }
        });


        out
    }
}

impl Layer<f32> for ReLU {
    fn call(&self, input: &ArrayViewD<f32>) -> ArrayD<f32> {
        ReLU::clamp(&input, &input)
    }

    fn backprop(&self, input: &ArrayViewD<f32>, output_gradient: &ArrayViewD<f32>,) -> (Option<crate::optimizers::LayerLossGradients<f32>>, Option<ArrayD<f32>>) {
        (None, Some(ReLU::clamp(output_gradient, input)))
    }

    fn get_state(&mut self) -> Option<&mut dyn super::LayerState<f32>> {
        None
    }
}
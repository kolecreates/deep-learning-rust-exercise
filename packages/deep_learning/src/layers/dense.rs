use ndarray::{ArrayD, ArrayViewD, Axis, Ix2};

use crate::{initializers::Initializer, optimizers::LayerLossGradients};

use super::{Layer, LayerState, StandardLayerState};

pub struct Dense {
    state: StandardLayerState<f32>,
}

impl Dense {
    pub fn create(shape: &Vec<usize>, weight_initializer: &dyn Initializer<f32>, bias_initializer: &dyn Initializer<f32>) -> Dense {
        let state = StandardLayerState::create(shape, weight_initializer, bias_initializer);
        Dense { state }
    }
}

impl Layer<f32> for Dense {
    fn call(&self, input: &ArrayViewD<f32>) -> ArrayD<f32> {
        self.state.weights.into_dimensionality::<Ix2>().unwrap().dot(&input.into_dimensionality::<Ix2>().unwrap()) + self.state.bias
    }

    fn backprop(&self, input: &ArrayViewD<f32>, output_gradient: &ArrayViewD<f32>, ) -> (Option<LayerLossGradients<f32>>, Option<ArrayD<f32>>) {
        let output_gradient_2d = output_gradient.into_dimensionality::<Ix2>().unwrap();
        let input_gradient = self.state.weights.t().into_dimensionality::<Ix2>().unwrap().dot(&output_gradient_2d);
        (Some(LayerLossGradients {
            weights: output_gradient_2d.dot(&input.t().into_dimensionality::<Ix2>().unwrap()).into_dyn(),
            bias: output_gradient_2d.sum_axis(Axis(1)).into_dyn(),
        }), Some(input_gradient.into_dyn()))
    }

    fn get_state(&mut self) -> Option<&mut dyn LayerState<f32>> {
        Some(&mut self.state)
    }
}
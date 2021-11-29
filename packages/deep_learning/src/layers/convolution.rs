use std::ops::{Add, Mul};

use ndarray::{ArrayD, ArrayView3, ArrayViewD, Axis, IndexLonger, Ix3, s};

use crate::{initializers::Initializer, optimizers::LayerLossGradients};

use super::{Layer, LayerState, StandardLayerState};

pub struct Conv {
    pub step: usize,
    state: StandardLayerState<f32>
}

impl Conv {
    pub fn create(step: usize, shape: &Vec<usize>, filter_initializer: &dyn Initializer<f32>, bias_initializer: &dyn Initializer<f32>) -> Conv {
        let state = StandardLayerState::create(shape, filter_initializer, bias_initializer);
        Conv { step, state }
    }
}

impl Layer<f32> for Conv {

    fn backprop(&self, input: &ArrayViewD<f32>,  output_gradient: &ArrayViewD<f32>, ) -> (Option<LayerLossGradients<f32>>, Option<ArrayD<f32>>) {
        let weight_shape = self.state.weights.shape();
        let number_of_filters = weight_shape[0];
        let filter_size = weight_shape[2];
        let input_shape = input.shape();
        let image_channels = input_shape[0];
        let image_size = input_shape[1];
        let mut loss_gradients = LayerLossGradients {
            weights: ArrayD::zeros(self.state.weights.raw_dim()),
            bias: ArrayD::zeros(self.state.bias.raw_dim()),
        };
        let mut input_gradient = ArrayD::zeros(input.raw_dim());

        for filter_index in 0..number_of_filters {
            let mut image_y = 0;
            let mut out_y = 0;
            let mut filter_gradient = loss_gradients.weights.slice_mut(s![filter_index, .., ..]).into_dimensionality::<Ix3>().unwrap();
            let filter = self.state.weights.index_axis(Axis(0), filter_index);
            while image_y + filter_size <= image_size {
                let mut image_x = 0;
                let mut out_x = 0;
                while image_x + filter_size <= image_size {
                    let input_patch_start = vec![0, image_y, image_x];
                    let input_patch_end = vec![image_channels, image_y+filter_size, image_x+filter_size];

                    let input_patch = input.slice(s![0..image_channels, image_y..image_y+filter_size, image_x..image_x+filter_size]);
                    let output_derivative = output_gradient[[filter_index, out_y, out_x]];
                    filter_gradient.scaled_add(output_derivative, &input_patch);

                    let mut input_gradient_patch = input_gradient.slice_mut(s![0..image_channels, image_y..image_y+filter_size, image_x..image_x+filter_size]);
                    input_gradient_patch.scaled_add(output_derivative, &filter);

                    image_x += self.step;
                    out_x += 1;
                    
                }

                image_y += self.step;
                out_y += 1;
            }

            loss_gradients.bias[[filter_index, 0]] = output_gradient.index_axis(Axis(0), filter_index).sum();
        }

        (Some(loss_gradients), Some(input_gradient))
    }
    fn call(&self, input: &Tensor<f32>) -> Tensor<f32> {
        let number_of_filters = self.state.weights.shape[0];
        let filter_channels = self.state.weights.shape[1];
        let filter_size = self.state.weights.shape[2];
        let image_channels = input.shape[0];
        let image_size = input.shape[1];

        assert!(filter_channels == image_channels);

        let out_size = ((image_size - filter_size) / self.step) + 1;

        let mut output = Tensor::from_shape(vec![number_of_filters, out_size, out_size], 0f32);

        for filter_index in 0..number_of_filters {
            let mut image_y = 0;
            let mut output_y = 0;
            while image_y + filter_size <= image_size {
                let mut image_x = 0;
                let mut output_x = 0;
                while image_x + filter_size <= image_size {
                    let filter = self.state.weights.get_at_first_axis_index(filter_index);
                    let image_patch = input.get_elements(&vec![0, image_y, image_x], &vec![image_channels, image_y+filter_size, image_x+filter_size]);
                    let product = filter.multiply(&image_patch);
                    let product_sum = product.sum();
                    let filter_bias = self.state.bias.data[filter_index];
                    output.set_element(&vec![filter_index, output_y, output_x], product_sum + filter_bias);
                    image_x += self.step;
                    output_x += 1;
                }
                image_y += self.step;
                output_y += 1;
            }
        }

        output
    }

    fn get_state(&mut self) -> Option<&mut dyn LayerState<f32>> {
        Some(&mut self.state)
    }
}
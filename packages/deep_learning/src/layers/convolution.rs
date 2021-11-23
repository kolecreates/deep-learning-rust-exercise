use math::linearalg::Tensor;

use crate::optimizers::LayerLossGradients;

use super::{Layer, LayerState, StandardLayerState};

pub struct Conv {
    pub step: usize,
    state: StandardLayerState<f32>
}

impl Layer<f32> for Conv {

    fn backprop(&self, input: &Tensor<f32>,  output_gradient: &Tensor<f32>, ) -> (Option<LayerLossGradients<f32>>, Tensor<f32>) {
        let number_of_filters = self.state.weights.shape[0];
        let filter_size = self.state.weights.shape[2];
        let image_channels = input.shape[0];
        let image_size = input.shape[1];
        let mut loss_gradients = LayerLossGradients {
            weights: Tensor::from_shape(self.state.weights.shape.clone(), 0.0),
            bias: Tensor::from_shape(self.state.bias.shape.clone(), 0.0),
        };
        let mut input_gradient = Tensor::from_shape(input.shape.clone(), 0.0);

        for filter_index in 0..number_of_filters {
            let mut image_y = 0;
            let mut out_y = 0;
            let mut filter_gradient = loss_gradients.weights.get_along_first_axis(filter_index);
            let filter = self.state.weights.get_along_first_axis(filter_index);
            while image_y + filter_size <= image_size {
                let mut image_x = 0;
                let mut out_x = 0;
                while image_x + filter_size <= image_size {
                    let input_patch_start = vec![0, image_y, image_x];
                    let input_patch_end = vec![image_channels, image_y+filter_size, image_x+filter_size];

                    let input_patch = input.get_elements(&input_patch_start, &input_patch_end);
                    let output_derivative = output_gradient.get_element(&vec![filter_index, out_y, out_x]);
                    filter_gradient = filter_gradient.add(&input_patch.scalar_multiply(output_derivative)); 

                    
                    let mut input_gradient_patch = input_gradient.get_elements(&input_patch_start, &input_patch_end);
                    input_gradient_patch = input_gradient_patch.add(&filter.scalar_multiply(output_derivative));
                    input_gradient.set_elements(&input_patch_start, &input_patch_end, &input_gradient_patch);

                    image_x += self.step;
                    out_x += 1;
                    
                }

                image_y += self.step;
                out_y += 1;
            }

            loss_gradients.weights.set_along_first_axis(filter_index, &filter_gradient);
            loss_gradients.bias.data[filter_index] = output_gradient.get_along_first_axis(filter_index).sum();
        }

        (Some(loss_gradients), input_gradient)
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
                    let filter = self.state.weights.get_along_first_axis(filter_index);
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
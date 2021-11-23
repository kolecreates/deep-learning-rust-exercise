use math::linearalg::Tensor;

use crate::optimizers::LayerLossGradients;

use super::{Layer};

pub struct MaxPool {
    kernal_size: usize,
    stride: usize,
}

impl Layer<f32> for MaxPool {
    fn call(&self, input: &Tensor<f32>) -> Tensor<f32> {
        let image_depth = input.shape[0];
        let image_height = input.shape[1];
        let image_width = input.shape[2];

        let output_height = ((image_height - self.kernal_size) / self.stride) + 1;
        let output_width = ((image_width - self.kernal_size) / self.stride) + 1;

        let mut output = Tensor::from_shape(vec![output_height, output_width], 0f32);

        for channel_index in 0..image_depth {
            let mut image_y = 0;
            let mut output_y = 0;
            while image_y + self.kernal_size <= image_height {
                let mut image_x = 0;
                let mut output_x = 0;
                while image_x + self.kernal_size < image_width {
                    let kernal = input.get_elements(&vec![channel_index, image_y, image_x], &vec![channel_index, image_y+self.kernal_size, image_x+self.kernal_size]);
                    output.set_element(&vec![channel_index, output_y, output_x], kernal.max());
                    image_x += self.stride;
                    output_x += 1;
                }
                image_y += self.stride;
                output_y += 1;
            }
        }
        output
    }

    fn backprop(&self, input: &Tensor<f32>, output_gradient: &Tensor<f32>,) -> (Option<LayerLossGradients<f32>>, Tensor<f32>){
        
        let mut input_gradient = Tensor::from_shape(input.shape.clone(), 0f32);

        let input_channels = input.shape[0];
        let input_size = input.shape[1];

        for channel_index in 0..input_channels {
            let mut input_y = 0;
            let mut output_y = 0;
            while input_y + self.kernal_size <= input_size {
                let mut input_x = 0;
                let mut output_x = 0;
                while input_x + self.kernal_size < input_size {
                    let window = input.get_elements(&vec![channel_index, input_y, input_x], &vec![channel_index, input_y+self.kernal_size, input_x+self.kernal_size]);
                    let indices_of_max = window.get_indices_of_max();
                    input_gradient.set_element(&vec![channel_index, input_y+indices_of_max[0], input_x+indices_of_max[1]], output_gradient.get_element(&vec![channel_index, output_y, output_x]));
                    input_x += self.stride;
                    output_x += 1;
                }
                input_y += self.stride;
                output_y += 1;
            }
        }

        (Option::None, input_gradient)
    }
}
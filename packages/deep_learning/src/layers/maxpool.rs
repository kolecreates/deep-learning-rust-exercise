use ndarray::{Array3, ArrayD, ArrayViewD, s};
use ndarray_stats::QuantileExt;

use crate::optimizers::LayerLossGradients;

use super::{Layer, LayerState};

pub struct MaxPool {
    pub kernal_size: usize,
    pub stride: usize,
}

impl Layer<f32> for MaxPool {
    
    fn call(&self, input: &ArrayViewD<f32>) -> ArrayD<f32> {
        let shape = input.shape();
        let image_depth = shape[0];
        let image_height = shape[1];
        let image_width = shape[2];

        let output_height = ((image_height - self.kernal_size) / self.stride) + 1;
        let output_width = ((image_width - self.kernal_size) / self.stride) + 1;

        let mut output: Array3<f32> = Array3::zeros((image_depth, output_height, output_width));

        for channel_index in 0..image_depth {
            let mut image_y = 0;
            let mut output_y = 0;
            while image_y + self.kernal_size <= image_height {
                let mut image_x = 0;
                let mut output_x = 0;
                while image_x + self.kernal_size < image_width {
                    let kernal = input.slice(s![channel_index, image_y..image_y+self.kernal_size, image_x..image_x+self.kernal_size]);
                    output[[channel_index, output_y, output_x]] = *kernal.max_skipnan();
                    image_x += self.stride;
                    output_x += 1;
                }
                image_y += self.stride;
                output_y += 1;
            }
        }
        output.into_dyn()
    }

    fn backprop(&self, input: &ArrayViewD<f32>, output_gradient: &ArrayViewD<f32>,) -> (Option<LayerLossGradients<f32>>, Option<ArrayD<f32>>){
        
        let shape = input.shape();
        let input_channels = shape[0];
        let input_size = shape[1];

        let mut input_gradient: Array3<f32> = Array3::zeros((input_channels, input_size, input_size));

        for channel_index in 0..input_channels {
            let mut input_y = 0;
            let mut output_y = 0;
            while input_y + self.kernal_size <= input_size {
                let mut input_x = 0;
                let mut output_x = 0;
                while input_x + self.kernal_size < input_size {
                    let window = input.slice(s![channel_index, input_y..input_y+self.kernal_size, input_x..input_x+self.kernal_size]);
                    let (max_y, max_x) = window.argmax_skipnan().unwrap();
                    input_gradient[[channel_index, input_y+max_y, input_x+max_x]] = output_gradient[[channel_index, output_y, output_x]];
                    input_x += self.stride;
                    output_x += 1;
                }
                input_y += self.stride;
                output_y += 1;
            }
        }

        (None, Some(input_gradient.into_dyn()))
    }

    fn get_state(&mut self) -> Option<&mut dyn LayerState<f32>> {
        None
    }
}
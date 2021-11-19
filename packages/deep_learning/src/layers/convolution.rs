use math::linearalg::Tensor;

use super::Layer;

pub struct ConvLayer {
    pub step: usize,
    filters: Tensor<f32>,
    bias: Tensor<f32>
}

impl Layer<f32> for ConvLayer {
    fn backprop(&self, input: &Tensor<f32> ) -> Tensor<f32> {
        input.clone()
    }
    fn call(&self, input: &Tensor<f32>) -> Tensor<f32> {
        let number_of_filters = self.filters.shape[0];
        let filter_size = self.filters.shape[1];
        let image_size = input.shape[0];
        let out_size = ((image_size - filter_size) / self.step) + 1;

        let mut output = Tensor::from_shape(vec![number_of_filters, out_size, out_size], 0f32);

        for filter_index in 0..number_of_filters {
            let mut image_y = 0;
            let mut output_y = 0;
            while image_y + filter_size <= image_size {
                let mut image_x = 0;
                let mut output_x = 0;
                while image_x + filter_size <= image_size {
                    let filter = self.filters.get_elements(&vec![filter_index, 0, 0], &vec![filter_index, filter_size, filter_size]);
                    let image_patch = input.get_elements(&vec![image_y, image_x], &vec![image_y+filter_size, image_x+filter_size]);
                    let product = filter.multiply(&image_patch);
                    let product_sum = product.sum();
                    let filter_bias = self.bias.data[filter_index];
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
}
mod activations {
    use math::linearalg::{Tensor, tensor_exp};
    pub fn softmax(input: &Tensor<f32>) -> Tensor<f32> {
        let output = tensor_exp(&input);
        output.scalar_divide(output.sum())
    }
}

mod losses {
    use math::linearalg::{Tensor, tensor_log};
    pub fn catagorical_cross_entryopy(probabilities: &Tensor<f32>, label: &Tensor<f32>) -> f32 {
        //label is a one-hot vector representing the correct catagory classification e.g. [0,0,1,0]
        -label.multiply(&tensor_log(probabilities)).sum()
    }
}

mod layers {
    use math::linearalg::{Tensor, tensor_dot};

    pub fn dense(input: &Tensor<f32>, weights: &Tensor<f32>, bias: &Tensor<f32>) -> Tensor<f32> {
        tensor_dot(weights, input).sum()
    }

    pub fn max_pool(image: &Tensor<f32>, kernal_size: usize, stride: usize) -> Tensor<f32> {
        let image_height = image.shape[0];
        let image_width = image.shape[1];

        let output_height = ((image_height - kernal_size) / stride) + 1;
        let output_width = ((image_width - kernal_size) / stride) + 1;

        let mut output = Tensor::from_shape(vec![output_height, output_width], 0f32);

        let mut image_y = 0;
        let mut output_y = 0;
        while image_y + kernal_size <= image_height {
            let mut image_x = 0;
            let mut output_x = 0;
            while image_x + kernal_size < image_width {
                let kernal = image.get_elements(&vec![image_y, image_x], &vec![image_y+kernal_size, image_x+kernal_size]);
                output.set_element(&vec![output_y, output_x], kernal.max());
                image_x += stride;
                output_x += 1;
            }
            image_y += stride;
            output_y += 1;
        }

        output
    }

    pub fn convolve_filters_over_image(image: &Tensor<f32>, filters: &Tensor<f32>, bias: &Tensor<f32>, step: usize) -> Tensor<f32> {
        assert!(filters.shape.len() > 0);
        let number_of_filters = filters.shape[0];
        let filter_size = filters.shape[1];
        let image_size = image.shape[0];
        let out_size = ((image_size - filter_size) / step) + 1;

        let mut output = Tensor::from_shape(vec![number_of_filters, out_size, out_size], 0f32);

        for filter_index in 0..number_of_filters {
            let mut image_y = 0;
            let mut output_y = 0;
            while image_y + filter_size <= image_size {
                let mut image_x = 0;
                let mut output_x = 0;
                while image_x + filter_size <= image_size {
                    let filter = filters.get_elements(&vec![filter_index, 0, 0], &vec![filter_index, filter_size, filter_size]);
                    let image_patch = image.get_elements(&vec![image_y, image_x], &vec![image_y+filter_size, image_x+filter_size]);
                    let product = filter.multiply(&image_patch);
                    let product_sum = product.sum();
                    let filter_bias = bias.data[filter_index];
                    output.set_element(&vec![filter_index, output_y, output_x], product_sum + filter_bias);
                    image_x += step;
                    output_x += 1;
                }
                image_y += step;
                output_y += 1;
            }
        }

        output
    }
}


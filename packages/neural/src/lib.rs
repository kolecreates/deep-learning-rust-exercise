pub mod architectures {
    pub mod cnn {
        use math::linearalg::Tensor;

        pub fn convolve_filters_over_image(image: &Tensor<f32>, filters: &Tensor<f32>, bias: &Tensor<f32>, step: i32) -> Tensor<f32> {
            assert!(filters.shape.len() > 0);
            let number_of_filters = filters.shape[0];
            let filter_size = filters.shape[1];
            let image_size = image.shape[0];
            let out_size = ((image_size - filter_size) / step) + 1;

            let output = Tensor { 
                shape: vec![number_of_filters, out_size, out_size], 
                data: vec![0f32; (number_of_filters*out_size*out_size).try_into().unwrap()]
            };

            for filter_index in 0..number_of_filters {
                let image_y = 0;
                let output_y = 0;
                while image_y + filter_size <= image_size {
                    let image_x = 0;
                    let output_x = 0;
                    while image_x + filter_size <= image_size {
                        let sum = 0f32;
                        let filter = filters.get_elements(&vec![filter_index, 0, 0], &vec![filter_index, filter_size, filter_size]);
                        let image_patch = image.get_elements(&vec![image_y, image_x], &vec![image_y+filter_size, image_x+filter_size]);
                        let product = filter.multiply(image_patch);
                        let product_sum = 0f32;
                        output.set_element(&vec![filter_index, output_y, output_x], product_sum);
                    }
                }
            }

            output
        }
    }
}

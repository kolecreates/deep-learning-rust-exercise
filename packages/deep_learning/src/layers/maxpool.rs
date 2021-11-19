mod layers {
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
}
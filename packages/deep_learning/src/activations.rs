mod activations {
    use math::linearalg::{Tensor, tensor_exp};
    pub fn softmax(input: &Tensor<f32>) -> Tensor<f32> {
        let output = tensor_exp(&input);
        output.scalar_divide(output.sum())
    }

    pub fn relu(input: &Tensor<f32>) -> Tensor<f32> {
        let mut out = input.clone();
        for i in 0..out.data.len() {
            if out.data[i] < 0.0 {
                out.data[i] = 0.0;
            }
        }

        out
    }
}
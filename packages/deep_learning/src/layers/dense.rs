mod layers {
    pub fn dense(input: &Tensor<f32>, weights: &Tensor<f32>, bias: &Tensor<f32>) -> Tensor<f32> {
        weights.dot(&input).add(&bias)
     }
}
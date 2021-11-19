mod losses {
    use math::linearalg::{Tensor, tensor_log};
    pub fn catagorical_cross_entryopy(probabilities: &Tensor<f32>, label: &Tensor<f32>) -> f32 {
        //label is a one-hot vector representing the correct catagory classification e.g. [0,0,1,0]
        -label.multiply(&tensor_log(probabilities)).sum()
    }
}
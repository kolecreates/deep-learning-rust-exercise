use math::linearalg::{Tensor, tensor_log};

pub trait Loss {
    fn compute(&self, output: &Tensor<f32>, label: &Tensor<f32>) -> f32;
}

pub struct CategoricalCrossEntropy;

impl Loss for CategoricalCrossEntropy {
    fn compute(&self, output: &Tensor<f32>, label: &Tensor<f32>) -> f32 {
        //label is a one-hot vector representing the correct catagory classification e.g. [0,0,1,0]
        -label.multiply(&tensor_log(output)).sum()
    }
}
use ndarray::ArrayViewD;

pub trait Loss {
    fn compute(&self, output: &ArrayViewD<f32>, label: &ArrayViewD<f32>) -> f32;
}

pub struct CategoricalCrossEntropy;

impl Loss for CategoricalCrossEntropy {
    fn compute(&self, output: &ArrayViewD<f32>, label: &ArrayViewD<f32>) -> f32 {
        //label is a one-hot vector representing the correct catagory classification e.g. [0,0,1,0]
        (label * output.map(|x|x.log(f32::EPSILON))).sum()
    }
}
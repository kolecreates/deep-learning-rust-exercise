use data::parse_u8_tensor_from_idx_file;
use math::linearalg::Tensor;

use crate::models::{Model, cnn::CNNClassifier};


pub fn run(){
    let mut classifer = CNNClassifier::create(10);

    let raw_samples = parse_u8_tensor_from_idx_file("../../assets/mnist-train-images");
    let mut samples  = Tensor::from_shape(raw_samples.shape, 0f32);
    for i in 0..samples.data.len() {
        samples.data[i] = raw_samples.data[i] as f32;
    }
    let mut labels = parse_u8_tensor_from_idx_file("../../assets/mnist-train-labels");

    println!("Loaded Training Data...");

    classifer.train(1, 32, &mut samples, &mut labels, 1);
}

#[cfg(test)]
mod tests {
    use super::run;

    #[test]
    fn test_run(){
        run();
    }
}
use data::parse_u8_ndarray_from_idx_file;
use ndarray::{Array2};

use crate::models::{Model, cnn::CNNClassifier};


pub fn run(){
    let mut classifer = CNNClassifier::create(10);

    let raw_samples = parse_u8_ndarray_from_idx_file("../../assets/mnist-train-images");
    let mut samples  = raw_samples.map(|x| *x as f32);
    let raw_labels = parse_u8_ndarray_from_idx_file("../../assets/mnist-train-labels");
    let mut labels = Array2::zeros((raw_labels.shape()[0], 10));

    for i in 0..raw_labels.shape()[0] {
        labels[[i, raw_labels[i] as usize]] = 1f32;
    }

    println!("Loaded Training Data...");

    classifer.train(1, 32, &mut samples, &mut labels.into_dyn(), 1);
}

#[cfg(test)]
mod tests {
    use super::run;

    #[test]
    fn test_run(){
        run();
    }
}
use math::linearalg::{Tensor, vec_product};

pub trait Initializer<T> {
    fn initialize(&self, shape: &Vec<usize>) -> Tensor<T>;
}

pub struct VarianceScaling {
    pub seed: u64,
}

impl Initializer<f32> for VarianceScaling {
    fn initialize(&self, shape: &Vec<usize>) -> Tensor<f32> {
        let units = vec_product(shape);
        let stddev = (1.0/(units as f32)).sqrt();
        Tensor::from_normal_distribution(shape, self.seed, 0f32, stddev)
    }
}

pub struct Zeros;

impl Initializer<f32> for Zeros {
    fn initialize(&self, shape: &Vec<usize>) -> Tensor<f32> {
        Tensor::from_shape(shape.clone(), 0f32)
    }
}

pub struct Constant {
    value: f32,
}

impl Initializer<f32> for Constant {
    fn initialize(&self, shape: &Vec<usize>) -> Tensor<f32> {
        Tensor::from_shape(shape.clone(), self.value)
    }
}


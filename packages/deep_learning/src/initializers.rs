use math::linearalg::vec_product;
use ndarray::{Array, Array2, ArrayBase, ArrayD, Dim, Dimension, OwnedRepr};
use ndarray_rand::RandomExt;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand::prelude::StdRng;
use ndarray_rand::rand_distr::Normal;

pub trait Initializer<T> {
    fn initialize(&self, shape: &[usize]) -> ArrayD<T>;
}

pub struct VarianceScaling {
    pub seed: u64,
}

impl Initializer<f32> for VarianceScaling {
    fn initialize(&self, shape: &[usize]) -> ArrayD<f32> {
        let units = vec_product(&shape.to_vec());
        let stddev = (1.0/(units as f32)).sqrt();
        let mut rng = StdRng::seed_from_u64(self.seed);
        let dist = Normal::new(0f32, stddev).unwrap();
        Array::random_using(shape, dist, &mut rng)
    }
}

pub struct Zeros;

impl Initializer<f32> for Zeros {
    fn initialize(&self, shape: &[usize]) -> ArrayD<f32> {
        Array::zeros(shape)
    }
}

pub struct Constant {
    value: f32,
}

impl Initializer<f32> for Constant {
    fn initialize(&self, shape: &[usize]) -> ArrayD<f32> {
        Array::from_elem(shape, self.value)
    }
}


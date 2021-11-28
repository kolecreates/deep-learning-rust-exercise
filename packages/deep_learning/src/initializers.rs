use ndarray::{Array, ArrayBase, Dimension, OwnedRepr};
use ndarray_rand::RandomExt;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand::prelude::StdRng;
use ndarray_rand::rand_distr::Normal;

pub trait Initializer<T> {
    fn initialize<D: Dimension>(&self, shape: D) -> ArrayBase<OwnedRepr<T>, D>;
}

pub struct VarianceScaling {
    pub seed: u64,
}

impl Initializer<f32> for VarianceScaling {
    fn initialize<D: Dimension>(&self, shape: D) -> ArrayBase<OwnedRepr<f32>, D> {
        let units = shape.size();
        let stddev = (1.0/(units as f32)).sqrt();
        let mut rng = StdRng::seed_from_u64(self.seed);
        let dist = Normal::new(0f32, stddev).unwrap();
        Array::random_using(shape, dist, &mut rng)
    }
}

pub struct Zeros;

impl Initializer<f32> for Zeros {
    fn initialize<D: Dimension>(&self, shape: D) -> ArrayBase<OwnedRepr<f32>, D> {
        Array::zeros(shape)
    }
}

pub struct Constant {
    value: f32,
}

impl Initializer<f32> for Constant {
    fn initialize<D: Dimension>(&self, shape: D) -> ArrayBase<OwnedRepr<f32>, D> {
        Array::from_elem(shape, self.value)
    }
}


use ndarray::{ArrayD, ArrayViewD};

use super::Layer;

pub struct Flatten;


impl<T:Clone> Layer<T> for Flatten {
    fn call(&self, input: &ArrayViewD<T>) -> ArrayD<T> {
        input.to_owned().into_shape((input.len(), 1)).unwrap().into_dyn()
    }

    fn backprop(&self, input: &ArrayViewD<T>, output_gradient: &ArrayViewD<T>,) -> (Option<crate::optimizers::LayerLossGradients<T>>, Option<ArrayD<T>>) {
        (None, Some(output_gradient.to_owned().into_shape(input.raw_dim()).unwrap().into_dyn()))
    }

    fn get_state(&mut self) -> Option<&mut dyn super::LayerState<T>> {
        todo!()
    }
}
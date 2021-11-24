use super::Layer;

pub struct Flatten;


impl<T:Clone> Layer<T> for Flatten {
    fn call(&self, input: &math::linearalg::Tensor<T>) -> math::linearalg::Tensor<T> {
        input.flatten()
    }

    fn backprop(&self, input: &math::linearalg::Tensor<T>, output_gradient: &math::linearalg::Tensor<T>,) -> (Option<crate::optimizers::LayerLossGradients<T>>, Option<math::linearalg::Tensor<T>>) {
        (None, Some(output_gradient.reshape(&input.shape)))
    }

    fn get_state(&mut self) -> Option<&mut dyn super::LayerState<T>> {
        todo!()
    }
}
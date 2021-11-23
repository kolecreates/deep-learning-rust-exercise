use math::linearalg::Tensor;

use crate::{layers::{Layer}, losses::Loss, optimizers::{LayerLossGradients, Optimizer}};

pub struct Model{
    layers: Vec<Box<dyn Layer<f32>>>,
    loss: Box<dyn Loss>,
    optimizer: dyn Optimizer<f32>,
}

impl Model {
    pub fn add_layer(&mut self, layer: Box<dyn Layer<f32>>) {
        self.layers.push(layer);
    }

    pub fn train(&mut self, num_epochs: usize, batch_size: usize, samples: &Tensor<f32>, labels: &Tensor<f32>) -> f32 {


        let sample_count = samples.shape[0];
        let batch_count = sample_count/batch_size;
        let layer_count = self.layers.len();

        for epoch_index in 0..num_epochs {
            for batch_index in 0..batch_count {
                let mut cost = 0.0;
                let mut batch_gradients: Vec<LayerLossGradients<f32>> = vec![];
                for sample_index in 0..batch_size {
                    let scaled_index = batch_index * batch_size + sample_index;
                    let mut outputs: Vec<Tensor<f32>> = vec![samples.get_along_first_axis(scaled_index)];
                    for layer_index in 0..layer_count {
                        let layer = &self.layers[layer_index];
                        let layer_input = &outputs[outputs.len()-1];
                        outputs.push(layer.call(layer_input));
                    }

                    let label = &labels.get_along_first_axis(scaled_index);
                    let model_output = &outputs[outputs.len()-1];

                    let mut output_gradient = model_output.sub(label);

                    let mut j = 0;

                    for i in 0..layer_count {
                        let layer_index = layer_count - i - 1;
                        let layer = &self.layers[layer_index];
                        let layer_input = &outputs[outputs.len()-2];
                        let (loss_gradients_option, input_gradient)  = layer.backprop(layer_input, &output_gradient);
                        output_gradient = input_gradient;

                        match loss_gradients_option {
                            None => {},
                            Some(loss_gradients) => {
                                if sample_index > 0 {
                                    let batch_gradient = &mut batch_gradients[j];
                                    batch_gradient.bias = batch_gradient.bias.add(&loss_gradients.bias);
                                    batch_gradient.weights = batch_gradient.weights.add(&loss_gradients.weights);
                                    j += 1;
                                }else{
                                    batch_gradients.push(loss_gradients);
                                }
                            }
                        }
                    }

                    cost += self.loss.compute(model_output, label);
                }

                for i in 0..batch_gradients.len() {
                    let batch_gradient = &mut batch_gradients[i];
                    batch_gradient.bias = batch_gradient.bias.scalar_divide(batch_size as f32);
                    batch_gradient.weights = batch_gradient.weights.scalar_divide(batch_size as f32);
                }

                let optimizations = self.optimizer.optimize(&batch_gradients);

                let opt_index = 0;
                for i in 0..layer_count {
                    let layer_index = layer_count - i - 1;
                    if self.layers[layer_index].update(&optimizations[opt_index]) {
                        
                    }
                }

            }   
        }

        0.0
    }
}
pub mod cnn;

use std::time::Instant;

use ndarray::{ArrayD, ArrayViewD, Axis};

use crate::{layers::{Layer}, losses::Loss, optimizers::{LayerLossGradients, Optimizer}};

pub trait Model<T> {
    fn train(&mut self, num_epochs: usize, batch_size: usize, samples: &mut ArrayD<T>, labels: &mut ArrayD<T>, seed:u64);
}

pub struct SequentialModel<'a> {
    layers: Vec<&'a mut dyn Layer<f32>>,
    loss: &'a dyn Loss,
    optimizer: &'a mut dyn Optimizer<f32>,
}

impl<'a> Model<f32> for SequentialModel<'a> {
    fn train(&mut self, num_epochs: usize, batch_size: usize, samples: &mut ArrayD<f32>, labels: &mut ArrayD<f32>, seed:u64) {

        let sample_count = samples.shape()[0];
        let batch_count = sample_count/batch_size;
        let layer_count = self.layers.len();

        for epoch_index in 0..num_epochs {
            println!("epoch start {}", epoch_index);
            let mut cost = 0.0;
            // samples.shuffle_first_axis(seed);
            // labels.shuffle_first_axis(seed);
            for batch_index in 0..batch_count {
                println!("batch start {}", batch_index);
                cost = 0f32;
                let mut batch_gradients: Vec<LayerLossGradients<f32>> = vec![];
                for sample_index in 0..batch_size {
                    println!("sample start {}", sample_index);
                    let scaled_index = batch_index * batch_size + sample_index;
                    let mut sample = samples.index_axis(Axis(0), scaled_index).to_owned();
                    sample.insert_axis_inplace(Axis(0));
                    let mut outputs: Vec<ArrayD<f32>> = vec![sample];
                    let forward_pass_start = Instant::now();
                    for layer_index in 0..layer_count {
                        let layer = &self.layers[layer_index];
                        outputs.push(layer.call(&outputs[outputs.len()-1].view()));
                    }

                    let forward_pass_end = forward_pass_start.elapsed();

                    println!("Forward pass time elapsed is: {:?}", forward_pass_end);

                    let label = &labels.index_axis(Axis(0), scaled_index);
                    let model_output = &outputs[outputs.len()-1];

                    let mut output_gradient = model_output - label;

                    let mut j = 0;

                    let backprop_start = Instant::now();

                    for i in 0..layer_count {
                        let layer_index = layer_count - i - 1;
                        let layer_input = &outputs[outputs.len()-i-2];
                        let (loss_gradients_option, input_gradient)  = self.layers[layer_index].backprop(&layer_input.view(), &output_gradient.view());
                        
                        match input_gradient {
                            None => {},
                            Some(grad)=> {
                                output_gradient = grad;
                            }
                        }

                        match loss_gradients_option {
                            None => {},
                            Some(loss_gradients) => {
                                if sample_index > 0 {
                                    let batch_gradient = &mut batch_gradients[j];
                                    batch_gradient.bias = &batch_gradient.bias + &loss_gradients.bias;
                                    batch_gradient.weights = &batch_gradient.weights + &loss_gradients.weights;
                                    j += 1;
                                }else{
                                    batch_gradients.push(loss_gradients);
                                }
                            }
                        }
                    }

                    let backprop_end = backprop_start.elapsed();

                    println!("Backprop time elapsed is: {:?}", backprop_end);

                    cost += self.loss.compute(&model_output.view(), label);
                }

                cost /= batch_size as f32;

                for i in 0..batch_gradients.len() {
                    let batch_gradient = &mut batch_gradients[i];
                    batch_gradient.bias = &batch_gradient.bias / batch_size as f32;
                    batch_gradient.weights = &batch_gradient.weights / batch_size as f32;
                }

                let optimizations = self.optimizer.optimize(&batch_gradients);

                let mut opt_index = 0;
                for i in 0..layer_count {
                    let layer_index = layer_count - i - 1;
                    match self.layers[layer_index].get_state() {
                        None => {},
                        Some(state) => {
                            state.update(&optimizations[opt_index]);
                            opt_index += 1;
                        }
                    }
                }

                println!("batch {} - cost {}", batch_index, cost);
            }  
            
            println!("epoch {} - cost {}", epoch_index, cost);
        }
    }
}
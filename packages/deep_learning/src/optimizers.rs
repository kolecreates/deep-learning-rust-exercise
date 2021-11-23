use math::linearalg::{Tensor, tensor_sqrt};

pub struct LayerOptimizations<T> {
    pub weights: Tensor<T>,
    pub bias: Tensor<T>
}

pub struct LayerLossGradients<T> {
    pub weights: Tensor<T>,
    pub bias: Tensor<T>
}

pub trait Optimizer<T> {
    fn optimize(&mut self, batch_avg_loss_gradients: &Vec<LayerLossGradients<T>>) -> Vec<LayerOptimizations<T>>;
}

//https://machinelearningmastery.com/adam-optimization-from-scratch/
pub struct AdamOptimizer {
    moment_w1: Vec<Tensor<f32>>,
    moment_w2: Vec<Tensor<f32>>,
    moment_b1: Vec<Tensor<f32>>,
    moment_b2: Vec<Tensor<f32>>,
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
}

impl AdamOptimizer {
    fn calc_moment_1(&self, moment1: &Tensor<f32>, gradient: &Tensor<f32>) -> Tensor<f32> {
        moment1.scalar_multiply(self.beta1).add(&gradient.scalar_multiply(1.0-self.beta1))
    }

    fn calc_moment_2(&self, moment2: &Tensor<f32>, gradient: &Tensor<f32>) -> Tensor<f32> {
        moment2.scalar_multiply(self.beta2).add(&gradient.multiply(&gradient).scalar_multiply(1.0-self.beta2))
    }

    fn calc_optimization(&self, moment1: &Tensor<f32>, moment2: &Tensor<f32>) -> Tensor<f32> {
        moment1.scalar_multiply(self.learning_rate).divide(&tensor_sqrt(&moment2).scalar_add(self.epsilon)).scalar_multiply(-1.0)
    }
}

impl Optimizer<f32> for AdamOptimizer {
    fn optimize(&mut self, batch_avg_loss_gradients: &Vec<LayerLossGradients<f32>>) -> Vec<LayerOptimizations<f32>> {

        if self.moment_w1.len() == 0 {
            //initialize moment vectors
            for i in 0..batch_avg_loss_gradients.len() {
                let layer_grads = &batch_avg_loss_gradients[i];
                self.moment_w1.push(Tensor::from_shape(layer_grads.weights.shape.clone(), 0f32));
                self.moment_b1.push(Tensor::from_shape(layer_grads.bias.shape.clone(), 0f32));
                self.moment_w2.push(Tensor::from_shape(layer_grads.weights.shape.clone(), 0f32));
                self.moment_b2.push(Tensor::from_shape(layer_grads.bias.shape.clone(), 0f32));
            }
        }
        
        let mut out: Vec<LayerOptimizations<f32>> = vec![];
        for layer_index in 0..batch_avg_loss_gradients.len() {
            let layer_grads = &batch_avg_loss_gradients[layer_index];
            self.moment_w1[layer_index] = self.calc_moment_1(&self.moment_w1[layer_index], &layer_grads.weights);
            self.moment_b1[layer_index] = self.calc_moment_1(&self.moment_b1[layer_index],&layer_grads.bias);
            self.moment_w2[layer_index] = self.calc_moment_2(&self.moment_w2[layer_index], &layer_grads.weights);
            self.moment_b2[layer_index] = self.calc_moment_2(&self.moment_b2[layer_index],&layer_grads.bias);
            let opt = LayerOptimizations {
                weights: self.calc_optimization(&self.moment_w1[layer_index], &self.moment_w2[layer_index]),
                bias: self.calc_optimization(&self.moment_b1[layer_index], &self.moment_b2[layer_index]),
            };

            out.push(opt);
        }

        out
    }
}
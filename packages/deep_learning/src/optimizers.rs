use ndarray::{Array, ArrayD};

pub struct LayerOptimizations<T> {
    pub weights: ArrayD<T>,
    pub bias: ArrayD<T>
}

pub struct LayerLossGradients<T> {
    pub weights: ArrayD<T>,
    pub bias: ArrayD<T>,
}

pub trait Optimizer<T> {
    fn optimize(&mut self, batch_avg_loss_gradients: &Vec<LayerLossGradients<T>>) -> Vec<LayerOptimizations<T>>;
}

//https://machinelearningmastery.com/adam-optimization-from-scratch/
pub struct AdamOptimizer {
    moment_w1: Vec<ArrayD<f32>>,
    moment_w2: Vec<ArrayD<f32>>,
    moment_b1: Vec<ArrayD<f32>>,
    moment_b2: Vec<ArrayD<f32>>,
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
}

impl AdamOptimizer {
    pub fn create(learning_rate: f32, beta1: f32, beta2: f32, epsilon: f32) -> AdamOptimizer {
        AdamOptimizer { moment_w1: vec![], moment_w2: vec![], moment_b1: vec![], moment_b2: vec![], learning_rate, beta1, beta2, epsilon }
    }

    fn calc_moment_1(&self, moment1: &ArrayD<f32>, gradient: &ArrayD<f32>) -> ArrayD<f32> {
        (moment1 * self.beta1) + (gradient * (1.0-self.beta1))
    }

    fn calc_moment_2(&self, moment2: &ArrayD<f32>, gradient: &ArrayD<f32>) -> ArrayD<f32> {
        (moment2 * self.beta2) + (gradient * gradient * (1.0-self.beta2))
    }

    fn calc_optimization(&self, moment1: &ArrayD<f32>, moment2: &ArrayD<f32>) -> ArrayD<f32> {
        ((moment1 * self.learning_rate) / (moment2.map(f32::sqrt) + self.epsilon)) * -1.0
    }
}

impl Optimizer<f32> for AdamOptimizer {
    fn optimize(&mut self, batch_avg_loss_gradients: &Vec<LayerLossGradients<f32>>) -> Vec<LayerOptimizations<f32>> {

        if self.moment_w1.len() == 0 {
            //initialize moment vectors
            for i in 0..batch_avg_loss_gradients.len() {
                let layer_grads = &batch_avg_loss_gradients[i];
                self.moment_w1.push(Array::zeros(layer_grads.weights.shape()));
                self.moment_b1.push(Array::zeros(layer_grads.bias.shape()));
                self.moment_w2.push(Array::zeros(layer_grads.weights.shape()));
                self.moment_b2.push(Array::zeros(layer_grads.bias.shape()));
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
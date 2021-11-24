use crate::{initializers::{VarianceScaling, Zeros}, layers, losses::CategoricalCrossEntropy, models::{Model, SequentialModel}, optimizers::AdamOptimizer};

pub struct CNNModel {
    conv_1: layers::Conv,
    relu_1: layers::activations::ReLU,
    conv_2: layers::Conv,
    relu_2: layers::activations::ReLU,
    pooling: layers::MaxPool,
    flatten: layers::Flatten,
    dense_1: layers::Dense,
    relu_3: layers::activations::ReLU,
    dense_2: layers::Dense,
    softmax: layers::activations::Softmax,
    optimizer: AdamOptimizer,
    loss: CategoricalCrossEntropy,
}

impl CNNModel {
    pub fn create() -> CNNModel {
        let conv_step = 1;
        let num_filters_1 = 8;
        let num_filters_2 = 8;
        let image_depth = 1;
        let filter_size = 5;
        let weight_initializer = VarianceScaling { seed: 1 };
        let bias_initializer = Zeros;
        let conv_shape_1 = vec![num_filters_1, image_depth, filter_size, filter_size];
        let conv_shape_2 = vec![num_filters_2, num_filters_1, filter_size, filter_size];
        let dense_shape_1 = vec![128,800];
        let dense_shape_2 = vec![10, 128];
        CNNModel {
            conv_1: layers::Conv::create(conv_step, &conv_shape_1, &weight_initializer, &bias_initializer),
            relu_1: layers::activations::ReLU,
            conv_2: layers::Conv::create(conv_step, &conv_shape_2, &weight_initializer, &bias_initializer),
            relu_2: layers::activations::ReLU,
            pooling: layers::MaxPool{ kernal_size: 2, stride: 2 },
            flatten: layers::Flatten,
            dense_1: layers::Dense::create(&dense_shape_1, &weight_initializer, &bias_initializer),
            relu_3: layers::activations::ReLU,
            dense_2: layers::Dense::create(&dense_shape_2, &weight_initializer, &bias_initializer),
            softmax: layers::activations::Softmax,
            optimizer: AdamOptimizer::create(0.01, 0.95, 0.99, 1e-7),
            loss: CategoricalCrossEntropy,
        }
        
    }
}

impl Model for CNNModel {
    fn train(&mut self, num_epochs: usize, batch_size: usize, samples: &mut math::linearalg::Tensor<f32>, labels: &mut math::linearalg::Tensor<f32>, seed:u64) {
        
        let mut model = SequentialModel { 
            layers: vec![
                &mut self.conv_1,
                &mut self.relu_1,
                &mut self.conv_2,
                &mut self.relu_2,
                &mut self.pooling,
                &mut self.flatten,
                &mut self.dense_1,
                &mut self.relu_3,
                &mut self.dense_2,
                &mut self.softmax,
            ], 
            loss: &self.loss, 
            optimizer: &mut self.optimizer,
        };

        model.train(num_epochs, batch_size, samples, labels, seed);
    }
}
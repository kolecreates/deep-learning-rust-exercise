use crate::{initializers::{VarianceScaling, Zeros}, layers, losses::CategoricalCrossEntropy, models::{Model, SequentialModel}, optimizers::AdamOptimizer};

pub struct CNNModel;

impl Model for CNNModel {
    fn train(&mut self, num_epochs: usize, batch_size: usize, samples: &mut math::linearalg::Tensor<f32>, labels: &mut math::linearalg::Tensor<f32>, seed:u64) {
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
        let mut conv_1 = layers::Conv::create(conv_step, &conv_shape_1, &weight_initializer, &bias_initializer);
        let mut relu_1 = layers::activations::ReLU;
        let mut conv_2 = layers::Conv::create(conv_step, &conv_shape_2, &weight_initializer, &bias_initializer);
        let mut relu_2 = layers::activations::ReLU;
        let mut pooling = layers::MaxPool{ kernal_size: 2, stride: 2 };
        let mut flatten = layers::Flatten;
        let mut dense_1 = layers::Dense::create(&dense_shape_1, &weight_initializer, &bias_initializer);
        let mut relu_3 = layers::activations::ReLU;
        let mut dense_2 = layers::Dense::create(&dense_shape_2, &weight_initializer, &bias_initializer);
        let mut softmax = layers::activations::Softmax;

        let mut optimizer = AdamOptimizer::create(0.01, 0.95, 0.99, 1e-7);
        let mut model = SequentialModel { 
            layers: vec![
                &mut conv_1,
                &mut relu_1,
                &mut conv_2,
                &mut relu_2,
                &mut pooling,
                &mut flatten,
                &mut dense_1,
                &mut relu_3,
                &mut dense_2,
                &mut softmax,
            ], 
            loss: &CategoricalCrossEntropy, 
            optimizer: &mut optimizer,
        };

        model.train(num_epochs, batch_size, samples, labels, seed);
    }
}
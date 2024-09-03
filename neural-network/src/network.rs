use matrix::matrix::Matrix;

use crate::activations::Activation;
use serde::{Deserialize, Serialize}; // Add this line at the top with other imports

#[derive(Serialize, Deserialize, Builder)]
pub struct Network {
    layers: Vec<usize>, // amount of neurons in each layer, [72,16,10]
    weights: Vec<Matrix>,
    biases: Vec<Matrix>,
    data: Vec<Matrix>,
    activation: Activation,
    learning_rate: f64,
}

impl Network {
    pub fn new(layers: Vec<usize>, activation: Activation, learning_rate: f64) -> Self {
        let mut weights = vec![];

        let mut biases = vec![];

        for i in 0..layers.len() - 1 {
            weights.push(Matrix::random(layers[i + 1], layers[i]));
            biases.push(Matrix::random(layers[i + 1], 1));
        }

        Network {
            layers,
            weights,
            biases,
            data: vec![],
            activation,
            learning_rate,
        }
    }

    pub fn save(&self) -> Vec<u8> {
        bincode::serialize(self).expect("Failed to serialize network")
    }

    // Load the network's state from a binary format
    pub fn load(&mut self, data: &[u8]) {
        let loaded_network: Network =
            bincode::deserialize(data).expect("Failed to deserialize network");
        *self = loaded_network;
    }

    // Feed the network forward
    pub fn feed_forward(&mut self, inputs: Matrix) -> Matrix {
        assert!(self.layers[0] == inputs.rows, "Invalid Number of Inputs");

        let mut current = inputs;

        self.data = vec![current.clone()];

        for i in 0..self.layers.len() - 1 {
            current = self.weights[i]
                .dot_multiply(&current)
                .add(&self.biases[i])
                .map(|x| self.activation.function(*x)); // Use closure to call the function

            self.data.push(current.clone());
        }

        current
    }

    pub fn back_propogate(&mut self, inputs: Matrix, targets: Matrix) {
        // Compute initial errors
        let mut errors = targets.subtract(&inputs);

        // Compute initial gradients
        let mut gradients = inputs.map(|x| self.activation.derivative(*x)); // Use closure to call derivative function

        for i in (0..self.layers.len() - 1).rev() {
            // Update gradients
            gradients = gradients
                .elementwise_multiply(&errors)
                .map(|x| x * self.learning_rate); // learning rate

            // Update weights and biases
            self.weights[i] =
                self.weights[i].add(&gradients.dot_multiply(&self.data[i].transpose()));
            self.biases[i] = self.biases[i].add(&gradients);

            // Update errors for the next layer
            errors = self.weights[i].transpose().dot_multiply(&errors);
            gradients = self.data[i].map(|x| self.activation.derivative(*x)); // Use closure to call derivative function
        }
    }

    pub fn train(&mut self, inputs: Vec<Vec<f64>>, targets: Vec<Vec<f64>>, epochs: u32) {
        for _i in 1..=epochs {
            for j in 0..inputs.len() {
                let outputs = self.feed_forward(Matrix::from(inputs[j].clone()));
                self.back_propogate(outputs, Matrix::from(targets[j].clone()));
            }
        }
    }
}

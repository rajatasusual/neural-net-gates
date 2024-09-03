use serde::{Serialize, Deserialize};
use std::f64::consts::E;

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum ActivationType {
    Sigmoid,
    // Add other activation functions here
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Activation {
    pub activation_type: ActivationType,
}

impl Activation {
    pub fn function(&self, x: f64) -> f64 {
        match self.activation_type {
            ActivationType::Sigmoid => 1.0 / (1.0 + E.powf(-x)),
            // Handle other activation functions here
        }
    }

    pub fn derivative(&self, x: f64) -> f64 {
        match self.activation_type {
            ActivationType::Sigmoid => x * (1.0 - x),
            // Handle derivatives for other activation functions here
        }
    }
}

// Constants for different activation functions
pub const SIGMOID: Activation = Activation {
    activation_type: ActivationType::Sigmoid,
    // Add other constants here
};
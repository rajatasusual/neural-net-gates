use neural_network::activations::SIGMOID;
use neural_network::network::Network;
use neural_network::matrix::Matrix;
use rand::seq::SliceRandom;
use std::fs::File;
use std::io::{Write, Read};
use std::f64;

fn main() {
    // Define the input-output pairs for each logic gate
    let logic_gates = vec![
        ("AND", vec![vec![0.0, 0.0], vec![0.0, 1.0], vec![1.0, 0.0], vec![1.0, 1.0]], vec![vec![0.0], vec![0.0], vec![0.0], vec![1.0]]),
        ("OR", vec![vec![0.0, 0.0], vec![0.0, 1.0], vec![1.0, 0.0], vec![1.0, 1.0]], vec![vec![0.0], vec![1.0], vec![1.0], vec![1.0]]),
        ("XOR", vec![vec![0.0, 0.0], vec![0.0, 1.0], vec![1.0, 0.0], vec![1.0, 1.0]], vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]]),
        ("NOT", vec![vec![0.0], vec![1.0]], vec![vec![1.0], vec![0.0]]),
        ("NAND", vec![vec![0.0, 0.0], vec![0.0, 1.0], vec![1.0, 0.0], vec![1.0, 1.0]], vec![vec![1.0], vec![1.0], vec![1.0], vec![0.0]]),
        ("NOR", vec![vec![0.0, 0.0], vec![0.0, 1.0], vec![1.0, 0.0], vec![1.0, 1.0]], vec![vec![1.0], vec![0.0], vec![0.0], vec![0.0]]),
        ("XNOR", vec![vec![0.0, 0.0], vec![0.0, 1.0], vec![1.0, 0.0], vec![1.0, 1.0]], vec![vec![1.0], vec![0.0], vec![0.0], vec![1.0]]),
    ];

    let mut rng = rand::thread_rng();

    for (gate_name, inputs, targets) in logic_gates {
        println!("Training network for {} gate...", gate_name);

        let input_size = inputs[0].len();
        let output_size = targets[0].len();
        let layers = if gate_name == "NOT" {
            vec![1, 2, 1] // Adjust architecture for NOT gate
        } else {
            vec![input_size, 2 * input_size, output_size]
        };

        // Initialize the network with the desired architecture and sigmoid activation function
        let mut network = Network::new(layers, SIGMOID, 0.5);

        // Load model if exists
        if let Ok(mut file) = File::open(format!("{}_model.bin", gate_name)) {
            let mut data = Vec::new();
            file.read_to_end(&mut data).unwrap();
            network.load(&data);
        }

        let mut best_loss = f64::MAX;
        let mut epochs_without_improvement = 0;
        let validation_split = 0.8;
        let train_size = (inputs.len() as f64 * validation_split).ceil() as usize;

        for epoch in 1..=100000 {
            // Shuffle data
            let mut combined: Vec<_> = inputs.clone().into_iter().zip(targets.clone()).collect();
            combined.shuffle(&mut rng);
            let (shuffled_inputs, shuffled_targets): (Vec<_>, Vec<_>) = combined.into_iter().unzip();

            // Split into training and validation sets
            let (train_inputs, val_inputs) = shuffled_inputs.split_at(train_size);
            let (train_targets, val_targets) = shuffled_targets.split_at(train_size);

            network.train(train_inputs.to_vec(), train_targets.to_vec(), 1);

            // Calculate loss on validation set
            let mut val_loss = 0.0;
            let mut count = 0;

            for (input, target) in val_inputs.iter().zip(val_targets.iter()) {
                let output = network.feed_forward(Matrix::from(input.clone()));

                if output.data.len() != target.len() {
                    println!("Mismatch in output and target sizes.");
                    continue;
                }

                let loss: f64 = target.iter()
                    .zip(output.data.iter())
                    .map(|(t, o)| {
                        if o.is_nan() || t.is_nan() {
                            println!("Encountered NaN value in output or target.");
                            return 0.0;
                        }
                        let diff = t - o;
                        if diff.is_nan() {
                            println!("NaN detected in loss calculation.");
                        }
                        diff * diff
                    })
                    .sum();
                
                if loss.is_nan() {
                    println!("NaN detected in loss calculation.");
                    continue;
                }

                val_loss += loss;
                count += 1;
            }

            if count > 0 {
                val_loss /= count as f64;
            } else {
                val_loss = f64::INFINITY;
            }

            if val_loss < best_loss {
                best_loss = val_loss;
                epochs_without_improvement = 0;

                // Save model
                let mut file = File::create(format!("{}_model.bin", gate_name)).unwrap();
                let data = network.save();
                file.write_all(&data).unwrap();
            } else {
                epochs_without_improvement += 1;
            }

            if epoch % 10000 == 0 {
                println!("Epoch: {}, Validation Loss: {:.6}", epoch, val_loss);
            }

            if epochs_without_improvement > 1000 {
                println!("Early stopping at epoch {} for {} gate.", epoch, gate_name);
                break;
            }
        }

        // Test the network with the input values
        for input in &inputs {
            let output = network.feed_forward(Matrix::from(input.clone()));
            let binary_output: Vec<f64> = output.data.iter().map(|&x| if x > 0.5 { 1.0 } else { 0.0 }).collect();
            println!("Input: {:?} => Output: {:?}", input, binary_output);
        }

        println!("Finished training for {} gate.\n", gate_name);
    }
}
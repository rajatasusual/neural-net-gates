use neural_network::activations::SIGMOID;
use neural_network::matrix::Matrix;
use neural_network::network::Network;
use std::env;

fn main() {
    env::set_var("RUST_BACKTRACE", "1");

    // Inputs and targets for all logic gates
    let all_inputs = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];

    let and_targets = vec![vec![0.0], vec![0.0], vec![0.0], vec![1.0]];
    let or_targets = vec![vec![0.0], vec![1.0], vec![1.0], vec![1.0]];
    let xor_targets = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];
    let not_targets = vec![vec![1.0], vec![0.0]]; // NOT gate only has two inputs
    let nand_targets = vec![vec![1.0], vec![1.0], vec![1.0], vec![0.0]];
    let nor_targets = vec![vec![1.0], vec![0.0], vec![0.0], vec![0.0]];
    let xnor_targets = vec![vec![1.0], vec![0.0], vec![0.0], vec![1.0]];

    let mut network = Network::new(vec![2, 3, 1], SIGMOID, 0.5);

	train(&mut network, all_inputs.clone(), and_targets, 100000, "AND");
	train(&mut network, all_inputs.clone(), or_targets, 100000, "OR");
	train(&mut network, all_inputs.clone(), xor_targets, 100000, "XOR");
	train(&mut network, all_inputs[0..2].to_vec(), not_targets, 100000, "NOT");
	train(&mut network, all_inputs.clone(), nand_targets, 100000, "NAND");
	train(&mut network, all_inputs.clone(), nor_targets, 100000, "NOR");
	train(&mut network, all_inputs.clone(), xnor_targets, 100000, "XNOR");
}


fn train( network: &mut Network, all_inputs: Vec<Vec<f64>>, targets: Vec<Vec<f64>>, epochs: u32, gate: &str) {

	print!("Training {}...\n", gate);

	network.train(all_inputs.clone(), targets, epochs);
    // Test on all inputs
    for input in &all_inputs {
        println!(
            "{:?} -> {:?}",
            input,
            network.feed_forward(Matrix::from(input.clone()))
        );
    }
}
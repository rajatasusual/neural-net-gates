use neural_network::activations::SIGMOID;
use neural_network::matrix::Matrix;
use neural_network::network::Network;
use std::collections::HashMap;
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

    // 7-dimensional target vectors
    let and_targets = vec![
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ];
    let or_targets = vec![
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ];
    let xor_targets = vec![
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ];

    let not_targets = vec![
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    ];

    let nand_targets = vec![
        vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ];

    let nor_targets = vec![
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ];

    let xnor_targets = vec![
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], // 1 for XNOR when both inputs are 1 or 0
    ];

    let mut network = Network::new(vec![2, 3, 7], SIGMOID, 0.5);

    train_and_test(
        &mut network,
        all_inputs.clone(),
        and_targets,
        100000,
        "AND",
    );
    train_and_test(
        &mut network,
        all_inputs.clone(),
        or_targets,
        100000,
        "OR",
    );
    train_and_test(
        &mut network,
        all_inputs.clone(),
        xor_targets,
        100000,
        "XOR",
    );
    train_and_test(
        &mut network,
        all_inputs[0..2].to_vec(),
        not_targets,
        100000,
        "NOT",
    );
    train_and_test(
        &mut network,
        all_inputs.clone(),
        nand_targets,
        100000,
        "NAND",
    );
    train_and_test(
        &mut network,
        all_inputs.clone(),
        nor_targets,
        100000,
        "NOR",
    );
    train_and_test(
        &mut network,
        all_inputs.clone(),
        xnor_targets,
        100000,
        "XNOR",
    );
}

fn train_and_test(
    network: &mut Network,
    all_inputs: Vec<Vec<f64>>,
    targets: Vec<Vec<f64>>,
    epochs: u32,
    gate: &str,
) {
    print!("Training {}...\n", gate);

    network.train(all_inputs.clone(), targets, epochs);

    // Map gate names to their corresponding output neuron index
    let gate_map = HashMap::from([
        ("AND", 0),
        ("OR", 1),
        ("XOR", 2),
        ("NOT", 3),
        ("NAND", 4),
        ("NOR", 5),
        ("XNOR", 6),
    ]);

    // Test on all inputs for the specified gate
    for input in &all_inputs {
        let output = network.feed_forward(Matrix::from(input.clone()));
        let gate_index = gate_map.get(gate).expect("Invalid gate name");
        let result = output.data[*gate_index];
        println!("Input: {:?}, Gate: {}, Output: {}", input, gate, result);
    }
}

use neural_network::activations::SIGMOID;
use neural_network::network::Network;
use neural_network::matrix::Matrix;

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

    for (gate_name, inputs, targets) in logic_gates {
        println!("Training network for {} gate...", gate_name);

        let layers = if gate_name == "NOT" {
            vec![1, 2, 1] // Adjust architecture for NOT gate
        } else {
            vec![2, 4, 1]
        };

        // Initialize the network with the desired architecture and sigmoid activation function
        let mut network = Network::new(layers, SIGMOID, 0.5);

        // Train the network for 10,000 epochs
        network.train(inputs.clone(), targets.clone(), 100000);

        // Test the network with the input values
        for input in &inputs {
            let output = network.feed_forward(Matrix::from(input.clone()));
            println!("Input: {:?} => Output: {:?}", input, output);
        }

        println!("Finished training for {} gate.\n", gate_name);
    }
}
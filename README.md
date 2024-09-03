
# Neural Network for Logic Gates

[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/rajatasusual/neural-net-gates/blob/main/LICENSE)


This project implements a neural network from scratch in Rust to train and test on seven basic logic gates: AND, OR, XOR, NOT, NAND, NOR, and XNOR.

## Project Structure

- `matrix.rs`: Contains the implementation of matrix operations, including addition, subtraction, multiplication, and transposition.
- `activations.rs`: Defines activation functions and their derivatives. Currently, only the sigmoid function is implemented.
- `network.rs`: Implements the neural network structure, including forward propagation, backpropagation, and training.
- `main.rs`: Initializes and trains the neural network on the logic gates data, then tests and prints the results.

## Dependencies

The project uses the following crates:

- `rand`: For generating random numbers in matrix initialization.

Make sure to add this to your `Cargo.toml`:

```toml
[dependencies]
rand = "0.8"
```

## Usage

To run the project:

1. Clone the repository:

    ```sh
    git clone https://github.com/rajatasusual/neural-net-gates.git
    ```

2. Navigate to the project directory:

    ```sh
    cd neural-net-gates
    ```

3. Build and run the project:

    ```sh
    cargo run
    ```

## Training Details

- **Logic Gates**: AND, OR, XOR, NOT, NAND, NOR, XNOR
- **Epochs**: 100,000 for each gate
- **Learning Rate**: 0.5
- **Activation Function**: Sigmoid function

## Code Explanation

- **Matrix Operations**: `matrix.rs` handles basic matrix operations essential for neural network computations.
- **Activation Functions**: `activations.rs` defines the sigmoid function and its derivative.
- **Neural Network**: `network.rs` manages the architecture, forward propagation, backpropagation, and training.
- **Main Program**: `main.rs` sets up the network, trains it on the logic gates, and prints the results.

## Contributing

If you want to contribute to the project, please fork the repository and submit a pull request. Any improvements or suggestions are welcome!

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
# neural-net-gates

[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/rajatasusual/neural-net-gates/blob/main/LICENSE)

A Rust-based neural network implementation for learning and recognizing logic gates (AND, OR, XOR, NOT, NAND, NOR, XNOR).

**Author:** rajatasusual

## Overview

This project demonstrates the fundamental concepts of neural networks in Rust by training a network to identify different logic gates based on their input-output patterns. The network architecture and training process are designed to be simple and understandable, making it an excellent educational resource for those new to neural networks.

## Features

* Implements a basic feedforward neural network with backpropagation for training.
* Supports multiple logic gates: AND, OR, XOR, NOT, NAND, NOR, XNOR.
* Uses the sigmoid activation function.
* Provides a custom `matrix!` macro for convenient matrix creation.

## Getting Started

1. **Clone the repository:**

   ```bash
   git clone [https://github.com/rajatasusual/neural-net-gates.git](https://github.com/rajatasusual/neural-net-gates.git)
   ```

2. **Navigate to the project directory:**

   ```bash
   cd neural-net-gates
   ```

3. **Build and run:**

   ```bash
   cargo run
   ```

## Usage

The `main.rs` file demonstrates how to train and test the network on the logic gates. The network architecture and training parameters can be customized within the code.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests to improve the project.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
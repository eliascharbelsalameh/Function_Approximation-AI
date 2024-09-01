# Neural Network for Function Approximation

## Project Overview
This project implements a fully connected feedforward neural network from scratch in Python. The goal is to approximate a function based on input-output data provided in a CSV file. The model uses the hyperbolic tangent activation function and employs backpropagation for training, with adjustable learning rate and momentum factors.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Neural Network Architecture](#neural-network-architecture)
- [Training Process](#training-process)
- [How to Use](#how-to-use)
- [Results](#results)
- [File Descriptions](#file-descriptions)

## Features
- **Customizable Architecture**: Define your own network architecture by specifying the number of layers and neurons per layer.
- **Training with Backpropagation**: The network is trained using backpropagation with momentum.
- **Real-Time Visualization**: During training, the network's progress is visualized in real-time, displaying both the desired and actual outputs.
- **Mean Squared Error (MSE) Monitoring**: The MSE is calculated and plotted over the epochs to evaluate the model's performance.

## Neural Network Architecture
- **Layers**: The network's architecture can be customized by defining the number of layers and neurons per layer. For example, `[1, 8, 8, 8, 8, 1]` represents a network with one input neuron, four hidden layers with 8 neurons each, and one output neuron.
- **Activation Function**: The network uses the hyperbolic tangent (`tanh`) function for activation.
- **Learning Rate and Momentum**: These hyperparameters control the speed and stability of the training process.

## Training Process
1. **Forward Propagation**: The input data is passed through the network to generate predictions.
2. **Backward Propagation**: The error is propagated backward through the network, calculating gradients for weight updates.
3. **Weight Update**: Weights are updated using gradient descent with momentum.
4. **Real-Time Visualization**: The desired vs. actual output is plotted in real-time during each epoch.
5. **MSE Calculation**: The MSE is calculated for each epoch and plotted at the end of training to show the model's performance over time.

## How to Use
1. **Prepare Data**: Ensure your input data is stored in a CSV file (`function.csv`), with the features and labels in separate columns.
2. **Adjust Hyperparameters**: Modify the layers, learning rate, momentum factor, and epochs according to your needs.
3. **Run the Script**: Execute the script to train the neural network on your data.
4. **Visualize Results**: Watch the real-time plots to see how well the network is learning, and review the final MSE plot to evaluate overall performance.

### Example Usage:
```python
# Define network architecture
layers = [1, 8, 8, 8, 8, 1]
learning_rate = 0.001
momentum_factor = 0.9
epochs = 50000

# Initialize and train the neural network
nn = NeuralNetwork(layers, learning_rate, momentum_factor, epochs)
nn.train(X, y)
```

## Results
- **Real-Time Output Visualization**: During training, a plot shows the desired vs. actual outputs, updated after each epoch.
- **MSE vs. Epochs Plot**: After training, a plot of the MSE over epochs provides insight into the model's performance and convergence.

## File Descriptions
- **`NeuralNetwork.py`**: The main script containing the `NeuralNetwork` class and training logic.
- **`function.csv`**: The CSV file containing the dataset used for training the neural network.

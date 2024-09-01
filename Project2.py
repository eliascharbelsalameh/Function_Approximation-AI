#   Artificial Neural Network from scratch - Project 2
#   GEL521 - Machine Learning
#   Presented to: Dr. Hayssam SERHAN
#   Presented by:   Antonio HADDAD          - 202200238
#                   Elias-Charbel SALAMEH   - 202201047

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

class NeuralNetwork:
    def __init__(self, layers, learning_rate, momentum_factor, epochs): 
        self.layers             = layers
        self.num_layers         = len(layers)
        self.learning_rate      = learning_rate
        self.momentum_factor    = momentum_factor
        self.epochs             = epochs
        self.weights            = [np.random.randn(layers[i-1], layers[i]) for i in range(1, len(layers))]
        self.old_weights        = [np.zeros((layers[i-1], layers[i])) for i in range(1, len(layers))]

    def save_weights(self, filename="project2_weights.npy"):
        np.save(filename, self.weights)

    def load_weights(self, filename="project2_weights.npy"):
        if os.path.isfile(filename):
            self.weights = np.load(filename, allow_pickle=True)
        else:
            print("File not found. Loading default weights.")

    def tanh(self, z):
        return (1.0 - np.exp(-z)) / (1.0 + np.exp(-z))

    def tanh_derivative(self, z):
        return 0.5 * ((1 + self.tanh(z)) * (1 - self.tanh(z)))
    
    def forward_propagation(self, x):
        activations = [x]           # assign input to be activated
        zs = []                     # effective Y in the GEL521 course

        for w in self.weights:
            
            z = np.dot(activations[-1],w) #v
            zs.append(z)
            activations.append(self.tanh(z)) #y

        return activations, zs      # V_layer,Y_layer

    def backward_propagation(self, x, y, activations, zs):
        deltas = [None] * self.num_layers
        gradients_w = [np.zeros(w.shape) for w in self.weights] 

        # Calculate output layer error
        deltas[-1] = (y - activations[-1]) * self.tanh_derivative(zs[-1])  # output_gradient

        # Backpropagation for hidden layers
        for l in range( self.num_layers - 2, 0, -1):  # Exclude input and output layers
            
            deltas[l] = np.dot(deltas[l+1], self.weights[l].T) * self.tanh_derivative(zs[l-1])

        # Compute gradients
        for l in range(self.num_layers - 1):
            gradients_w[l] = np.dot(np.array(activations[l]).T, deltas[l+1])

        return gradients_w

    def calculate_mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def train(self, X, y):
        plt.ion()
        fig, ax = plt.subplots()

        mse_values = []  # List to store MSE values at each epoch

        for epoch in range(self.epochs):
            total_gradients_w = [np.zeros(w.shape) for w in self.weights]

            activations, zs = self.forward_propagation(X)
            gradients_w = self.backward_propagation(X, y, activations, zs)

            total_gradients_w = [tw + gw for tw, gw in zip(total_gradients_w, gradients_w)]

            # Update weights
            for l in range(self.num_layers - 1):
                self.weights[l] += self.momentum_factor * (self.weights[l] - self.old_weights[l]) + \
                                   self.learning_rate * total_gradients_w[l]

            # Update old_weights
            self.old_weights = [w.copy() for w in self.weights]

            # Calculate MSE and append to list
            mse = self.calculate_mse(y, activations[-1])
            mse_values.append(mse)

            ax.clear()
            ax.plot(X, y, label='Desired Output')
            ax.plot(X, activations[-1], label='Actual Output')

            ax.set_xlabel('Input')
            ax.set_ylabel('Output')
            ax.set_title(f'Epoch {epoch + 1}/{self.epochs} \nMSE: {mse:.4f} - LR: {self.learning_rate} - MF: {self.momentum_factor} - Structure: {self.layers}')
            ax.legend()
            plt.pause(0.001)

        plt.ioff()
        plt.show()

        # Display MSE values
        plt.figure()
        plt.plot(range(1, self.epochs + 1), mse_values, linestyle='-')
        plt.title('Mean Squared Error (MSE) vs. Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('MSE')
        plt.grid(True)
        plt.show()

data = pd.read_csv("Project2/function.csv")
X = data.iloc[:,0].values     # Features, inputs
y = data.iloc[:,3].values       # Target labels

X = (X   -   X.min(axis=0)) /   (X.max(axis=0) - X.min(axis=0))   # normalize inputs

X = np.array(X).reshape(-1,1)
y = np.array(y).reshape(-1,1)

# Plotting
layers          = [1, 8, 8, 8, 8, 1]
learning_rate   = 0.001
momentum_factor = 0.9
epochs          = 50000
nn = NeuralNetwork(layers,learning_rate,momentum_factor,epochs)
nn.train(X,y)
import numpy as np
from nnfs.datasets import spiral_data
import nnfs

nnfs.init()
import matplotlib.pyplot as plt


# plt.scatter(X[:,0], X[:,1])
# plt.show()

# number of neurons (let's say 3)
# number of inputs (let's say 2)
# class will initialize random weights and biases of each neuron
# class will take in an input and give an output
# output is np.dot(X, W) + b
# where X is input vector (Horizontal), W is weight matrix (vertical), b is bias
class Layer_Dense:
    # Will intialize weight matrix with correct dimensions and random weights and biases
    def __init__(self, num_inputs, num_neurons):
        # Rows = weight each neuron (y)
        # Columns = input each neuron (x)
        # Dimension of layer: inputs * neurons
        # Each layer will be a weight matrix and bias matrix
        # the bias matrix represents each neurons weight (one row)
        self.weights = 0.01 * np.random.randn(num_inputs, num_neurons)
        self.biases = np.zeros((1, num_neurons))  # Initialized as 0 for now

    # will return an output based on the inputs and weight + bias matrix
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_Softmax:
    def forward(self, inputs):
        # Takes in a matrix and returns the probability of each value
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True)) # Makes the exponent matrix [e^1, e^2, ...]
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True) # Divides each exponent with sum of exponent rows
        self.output = probabilities

class Loss_CategoricalCrossentropy:
    def forward(self, y_pred, y_true):
        # Ignore predictions which are too close to 0 or 1
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_true.shape) == 1: # Correct color array is in the form of [0] or [1] or [2]
            samples = len(y_pred)
            correct_confidences = y_pred_clipped[range(samples), y_true]

        elif len(y_true.shape) == 2: # Correct color array is in the form [1,0,0] or [0,1,0] or [0,0,1]
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1) # Result will be [0.7, 0.8, 0.4] instead of [[0.7 ,0 ,0], [0, 0.8, 0], [0, 0.4, 0]], by using np.sum

        # calculate loss:
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods





X, y = spiral_data(samples=100, classes=3)

dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3,3)
activation2 = Activation_Softmax()

dense1.forward(X)

activation1.forward(dense1.output)

dense2.forward(activation1.output)

activation2.forward(dense2.output)

print(activation2.output[0:5])


# A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
#print(np.sum(A))  # sums up each number 45

#print(np.sum(A, axis=0))  # sums the columns [12 15 18]
#print(np.sum(A, axis=1))  # sums the rows [6 15 24]

#print(np.sum(A, axis=0, keepdims=True))  # Sums the rows and keeps dimensions
#print(np.sum(A, axis=0, keepdims=True).shape)  # [[12 15 18]] (1,3)

#print(np.sum(A, axis=1, keepdims=True))  # sums the columns and keeps dimensions
#print(np.sum(A, axis=1, keepdims=True).shape)  # (3,1)
# [[6]
# [15]
# [24]]


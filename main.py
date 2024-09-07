import numpy as np
from nnfs.datasets import spiral_data, vertical_data
import nnfs
import matplotlib.pyplot as plt

# Layer class of Neural netowrk
class Layer_Dense:
    # Will intialize weight matrix with correct dimensions and random weights and biases
    def __init__(self, num_inputs, num_neurons):
        # Dimension of layer: inputs * neurons
        self.weights = 0.01 * np.random.randn(num_inputs, num_neurons)
        self.biases = np.zeros((1, num_neurons))  # Initialized as 0 for now

    # will return an output based on the inputs and weight + bias matrix
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

    # Backpropagation: dvalues is dL_dZ (derivative of loss respect to output of this layer)
    def backward(self, dvalues):
        # to get dL_dW (derivative loss respect weights) you take dL_dZ * X(T) 
        self.dweights = np.dot(dvalues, self.inputs.T)
        # to get dL_dB (derivative loss respect biases) you take dL_dZ
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True) # Sum rows
        # to get dL_dX (derivative loss respect inputs) you take dL_dZ * W(T) 
        self.dinputs = np.dot(dvalues, self.weights.T)

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        # dvalues = dL_da, dL_dz = dL_da * da_dZ (da_dz = 0 or 1 based on input Z)
        self.dinputs[self.inputs < 0] = 0

class Activation_Softmax:
    def forward(self, inputs):
        # Takes in a matrix and returns the probability of each value
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True)) # Makes the exponent matrix [e^1, e^2, ...]
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True) # Divides each exponent with sum of exponent rows
        self.output = probabilities

class Activation_Softmax_Loss_CategoricalCrossentropy:
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()
    
    def forward(self, inputs, y_true):
        # Calculate probabilities (Softmax)
        self.activation.forward(inputs)
        self.output = self.activation.output
        # Calculate loss of probabilities (Cross entropy)
        return self.loss.calculate(self.output, y_true)
    
    def backward(self, dvalues, y_true):
        # dvalues = predicted values [0.1, 0.7, 0.2], y_true = answer [0, 1, 0]
        samples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.dinputs = dvalues.copy()
        # dL_dz = predicted - ans
        self.dinputs[range(samples), y_true] -= 1
        # normalize dL_dz with number of samples
        self.dinputs = self.dinputs / samples

class Loss_CategoricalCrossentropy:
    def forward(self, y_pred, y_true):
        # Ignore predictions which are too close to 0 or 1
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_true.shape) == 1: # correct_color array is in the form of [0] or [1] or [2]
            samples = len(y_pred)
            correct_confidences = y_pred_clipped[range(samples), y_true]

        elif len(y_true.shape) == 2: # Correct color array is in the form [1,0,0] or [0,1,0] or [0,0,1]
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1) # Result will be [0.7, 0.8, 0.4] instead of [[0.7 ,0 ,0], [0, 0.8, 0], [0, 0.4, 0]], by using np.sum

        # calculate loss:
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        # dL_dy = y_ans / y_guess
        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples

    def calculate(self, output, y):
        # Calculate sample losses
        sample_losses = self.forward(output, y)
        # Calculate mean loss
        data_loss = np.mean(sample_losses)
        # Return loss
        return data_loss

# Adjusting weight and biases randomly:
nnfs.init()
X, y = vertical_data(samples=100, classes=3)
# Create model:
dense1 = Layer_Dense(2,3)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3,3)
activation2 = Activation_Softmax()

loss_function = Loss_CategoricalCrossentropy()

lowest_loss = float('inf')
best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.biases.copy()
best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.biases.copy()

for i in range(300):
    dense1.weights += 0.05 * np.random.randn(2,3) # Returns a normally distrubuted random number witht he shape 2x3 (3 neurons have 2 weights each)
    dense1.biases += 0.05 * np.random.randn(1,3) # Returns a normally distributed random number with the shape 1x3 (3 neurons, 1 for each)
    dense2.weights += 0.05 * np.random.randn(3,3)
    dense2.biases += 0.05 * np.random.randn(1, 3)

    dense1.forward(X)
    activation1.forward(dense1.output)

    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    loss = loss_function.calculate(activation2.output, y)

    # We chose the highest probability as the prediction
    predictions = np.argmax(activation2.output, axis=1)
    accuracy = np.mean(predictions == y)

    if loss < lowest_loss:
        print('new set of weights found iteration: ', i, 'loss: ', loss, 'accuracy: ', accuracy)
        best_dense1_weights = dense1.weights.copy()
        best_dense1_biases = dense1.biases.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense2_biases = dense2.biases.copy()
        lowest_loss = loss
    else:
        dense1.weights = best_dense1_weights.copy()
        dense1.biases = best_dense1_biases.copy()
        dense2.weights = best_dense2_weights.copy()
        dense2.biases = best_dense2_biases.copy()

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


from nnfs.datasets import spiral_data
import numpy as np
import nnfs
nnfs.init()
X, y = spiral_data(samples=200, classes=3)

# Dense layer
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # Weight matrix in shape: inputs * neurons
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    # Forward pass
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    # Backward pass
    def backward(self, dvalues):
        # to get dL_dW (derivative loss respect weights) you take dL_dZ * X(T) 
        self.dweights = np.dot(self.inputs.T, dvalues)
        # to get dL_dB (derivative loss respect biases) you take dL_dZ
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # to get dL_dX (derivative loss respect inputs) you take dL_dZ * W(T) 
        self.dinputs = np.dot(dvalues, self.weights.T)

# ReLU activation
class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        # (0 if inputs <=0 else inputs)
        self.output = np.maximum(0, inputs)

    # Backward pass
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        # d(ReLU)_d(input) = 0 (input < 0), else 1
        self.dinputs[self.inputs <= 0] = 0

# Softmax activation
class Activation_Softmax:
    def forward(self, inputs):
        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1,keepdims=True)
        self.output = probabilities

# Common loss class
class Loss:
    def calculate(self, output, y):
        # Calculate sample losses
        sample_losses = self.forward(output, y)
        # Calculate mean loss
        data_loss = np.mean(sample_losses)
        # Return loss
        return data_loss

class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        # Clip data to prevent division by 0
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Correct format
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(samples),
                y_true
            ]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped * y_true,
                axis=1
            )

        # Loss = -log(predicted) Example: -log(0.7)
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    # Backward pass
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])

        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        # dL_y = -ans / guess
        self.dinputs = -y_true / dvalues
        # Normalize gradient
        self.dinputs = self.dinputs / samples

# Softmax activation fucntion combined with crossentropy loss
class Activation_Softmax_Loss_CategoricalCrossentropy:
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    # Forward pass
    def forward(self, inputs, y_true):
        # Calcluate probabilities (Softmax)
        self.activation.forward(inputs)
        self.output = self.activation.output
        # Calculate loss of probabilties
        return self.loss.calculate(self.output, y_true)

    # Backward pass
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        self.dinputs = dvalues.copy()
        # dL_d(input softmax) = predicted - ans
        self.dinputs[range(samples), y_true] -= 1
        # Normalize gradient
        self.dinputs = self.dinputs / samples

# SGD optimizer
class Optimizer_SGD:
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate
    def update_params(self, layer):
        # Update weights and biases
        layer.weights += -self.learning_rate * layer.dweights
        layer.biases += -self.learning_rate * layer.dbiases

# Build Model
dense1 = Layer_Dense(2, 64)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(64, 3)
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
optimizer = Optimizer_SGD()

# Train model
for epoch in range(50001):
    # Forward pass
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss = loss_activation.forward(dense2.output, y)

    # Calculate accuracy
    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)
    if not epoch % 100:
        print(f'epoch: {epoch}, ' +
              f'acc: {accuracy:.3f}, ' +
              f'loss: {loss:.3f}')
        
    # Backward pass
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # Update weights and biases
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)

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


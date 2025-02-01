import numpy as np
import matplotlib.pyplot as plt

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons, is_first_layer=False):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.is_first_layer = is_first_layer

    # Forward pass
    def forward(self, inputs):
        self.inputs = inputs # stores the x values
        # a = w1 * x + b1
        self.output = np.dot(inputs, self.weights) + self.biases

    # Backward pass
    # dL/dw1 = dL/dy * dy/dz1 * dz1/da * da/dw1
    # dvalues = dL/dy * dy/dz1 * dz1/da
    # derivative of neuron: 
    # da/dw1 = x and da/db1 = 1

    # so we need to multiply dvalues with x (for weights) or 1 (for biases)
    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues) # multiply derivative with x
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True) # dbiases = dvalues (since we multiply with 1)
        # Only compute dinputs if not the first layer (since it's not used there)
        if not self.is_first_layer:
            self.dinputs = np.dot(dvalues, self.weights.T)

# ReLU activation
class Activation_ReLU:
    def forward(self, inputs): # takes in output of each neuron like: a1 = w1x + b1
        self.inputs = inputs # stores a1, a2, a3 (output of each neuron)
        # z1 = (a1: a1 > 0, else 0)
        self.output = np.maximum(0, inputs) # returns a or 0

    # Backward pass
    # dL/dw1 = dL/dy * dy/dz1 * dz1/da * da/dw1
    # dvalues = dL/dy * dy/dz1
    # derivative of relu: dz1/da = (1 or 0)

    # therefore we either multiple with 1 or 0
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        # if inputs (a) is less than 0, multiply dvalues with 0
        # else multiply dvalues with 1
        self.dinputs[self.inputs <= 0] = 0 

# Rest of the code remains the same...
class Loss_MeanSquaredError:
    def forward(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2, axis=-1)

    def backward(self, y_pred, y_true):
        self.dinputs = 2 * (y_pred - y_true) / len(y_pred)
    
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Optimizer_SGD:
    def __init__(self, learning_rate, decay):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0

    def update_params(self, layer):
        layer.weights -= self.current_learning_rate * layer.dweights
        layer.biases -= self.current_learning_rate * layer.dbiases

    def post_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate / (1. + self.decay * self.iterations)
        self.iterations += 1

# Data
X = np.linspace(-3, 3, 128).reshape(-1, 1)
Y = X**3 - 3*X + np.random.normal(0, 1, X.shape)

# Set up model with three layers
dense1 = Layer_Dense(1, 128, is_first_layer=True)  # Input layer -> Hidden layer
activation1 = Activation_ReLU()
dense2 = Layer_Dense(128, 64)  # Hidden layer -> Hidden layer
activation2 = Activation_ReLU()
dense3 = Layer_Dense(64, 1)    # Hidden layer -> Output layer

loss_function = Loss_MeanSquaredError()
optimizer = Optimizer_SGD(learning_rate=0.001, decay=1e-4)

# Training loop
loss_history = []

for epoch in range(5000):
    # Forward pass
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    dense3.forward(activation2.output)

    # Calculate loss
    loss = loss_function.calculate(dense3.output, Y)
    loss_history.append(loss)

    # Print loss every 100 epochs
    if epoch % 100 == 0:
        print(f'epoch: {epoch}, loss: {loss}')

    # Backward pass
    loss_function.backward(dense3.output, Y)
    dense3.backward(loss_function.dinputs)
    activation2.backward(dense3.dinputs)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)  # dinputs won't be computed here

    # Update weights and biases
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.update_params(dense3)
    optimizer.post_update_params()

# Generate predictions
dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
dense3.forward(activation2.output)
predictions = dense3.output

# Plotting code remains the same...
sort_indices = np.argsort(X.flatten())
X_sorted = X[sort_indices]
Y_sorted = Y[sort_indices]
predictions_sorted = predictions[sort_indices]

plt.figure(figsize=(10, 6))
plt.scatter(X, Y, color='blue', alpha=0.5, label='Actual Data')
plt.plot(X_sorted, predictions_sorted, color='red', linewidth=2, label='Model Predictions')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('3-Layer Neural Network Fit vs Actual Data')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(loss_history, color='blue', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Training Loss Over Time')
plt.grid(True)
plt.show()

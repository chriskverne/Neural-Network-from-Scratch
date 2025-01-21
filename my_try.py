import numpy as np
import matplotlib.pyplot as plt

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

# Common loss class
class Loss:
    def calculate(self, output, y):
        # Calculate sample losses
        sample_losses = self.forward(output, y)
        # Calculate mean loss
        data_loss = np.mean(sample_losses)
        # Return loss
        return data_loss

# Softmax activation fucntion combined with crossentropy loss
class Loss_MeanSquaredError(Loss):
    # Forward pass
    def forward(self, y_pred, y_true):
        # Calculate squared differences between predictions and true values
        return np.mean((y_pred - y_true) ** 2, axis=-1)

    # Backward pass
    def backward(self, dvalues, y_true):
        # Gradient of MSE Loss with respect to predictions
        samples = len(dvalues)
        outputs = len(dvalues[0])
        
        # dL/dy_pred = 2 * (y_pred - y_true) / samples
        self.dinputs = 2 * (dvalues - y_true) / samples

# SGD optimizer with learning rate decay
class Optimizer_SGD:
    def __init__(self, learning_rate=1., decay=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0

    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            # learning rate = intial_learning_rate / (1 + decay * current_iteration) (Will decrease over time)
            self.current_learning_rate = self.learning_rate / (1. + self.decay * self.iterations)

    # Update parameters
    def update_params(self, layer):
        # new_weight = old_weight - (a * dL_dW)
        weight_updates = -self.current_learning_rate * layer.dweights
        bias_updates = -self.current_learning_rate * layer.dbiases

        # Update weights and biases
        layer.weights += weight_updates
        layer.biases += bias_updates

    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1

# Data
X = np.linspace(-3, 3, 128).reshape(-1, 1)
Y = X**3 - 3*X + np.random.normal(0, 1, X.shape)

# Set up model
dense1 = Layer_Dense(1, 128) 
activation1 = Activation_ReLU()
dense2 = Layer_Dense(128, 64)  # New hidden layer with 64 neurons
activation2 = Activation_ReLU()
dense3 = Layer_Dense(64, 1)    # Output dim: 1 (We predict 1 value for regression)
loss_function = Loss_MeanSquaredError()
optimizer = Optimizer_SGD(learning_rate=0.001, decay=1e-5)

# Training loop
epochs = 30000
for epoch in range(epochs):
    # Forward pass
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    dense3.forward(activation2.output)

    # Calculate loss
    loss = loss_function.calculate(dense3.output, Y)

    # Print loss every 1000 epochs
    if epoch % 1000 == 0:
        print(f'epoch: {epoch}, loss: {loss}')

    # Backward pass
    loss_function.backward(dense3.output, Y)
    dense3.backward(loss_function.dinputs)
    activation2.backward(dense3.dinputs)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # Update weights and biases
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.update_params(dense3)
    optimizer.post_update_params()



# Plot results from trained network
dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
dense3.forward(activation2.output)
predictions = dense3.output

# Sort X and corresponding predictions/Y values for smooth plotting
sort_indices = np.argsort(X.flatten())
X_sorted = X[sort_indices]
Y_sorted = Y[sort_indices]
predictions_sorted = predictions[sort_indices]

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(X, Y, color='blue', alpha=0.5, label='Actual Data')
plt.plot(X_sorted, predictions_sorted, color='red', linewidth=2, label='Model Predictions')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Neural Network Fit vs Actual Data')
plt.legend()
plt.grid(True)
plt.show()
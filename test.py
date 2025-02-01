import numpy as np
import matplotlib.pyplot as plt

def state_zero():
    return np.array([1.0 + 0.0j, 0.0 + 0.0j])

def RY(theta):
    return np.array([
        [ np.cos(theta/2), -np.sin(theta/2)],
        [ np.sin(theta/2),  np.cos(theta/2)]
    ], dtype=complex)

def measure_z(state):
    prob0 = np.abs(state[0])**2
    prob1 = np.abs(state[1])**2
    return (prob0 - prob1).real

def quantum_circuit(x, params):
    w, b = params
    state = state_zero()
    angle = w*x + b
    final_state = RY(angle) @ state
    return measure_z(final_state)

def cost_function(params, x_data, y_data):
    predictions = np.array([quantum_circuit(x, params) for x in x_data])
    return np.mean((predictions - y_data)**2)

def compute_grad(cost_fn, params, x_data, y_data, delta=1e-4):
    grads = np.zeros_like(params, dtype=float)
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += delta

        params_minus = params.copy()
        params_minus[i] -= delta

        c_plus = cost_fn(params_plus, x_data, y_data)
        c_minus = cost_fn(params_minus, x_data, y_data)

        grads[i] = (c_plus - c_minus) / (2.0 * delta)
    return grads

# 1) Generate data
N = 30
x_data = np.linspace(0, 2*np.pi, N)
y_data = np.sin(x_data)

# 2) Initialize parameters
np.random.seed(42)
params = np.random.randn(2)  # [w, b]

# 3) Train
learning_rate = 0.1
epochs = 200
cost_history = []

for epoch in range(epochs):
    current_cost = cost_function(params, x_data, y_data)
    cost_history.append(current_cost)

    grads = compute_grad(cost_function, params, x_data, y_data)
    params -= learning_rate * grads

    if (epoch+1) % 20 == 0:
        print(f"Epoch {epoch+1}/{epochs} | Cost = {current_cost:.6f} | params = {params}")

# 4) Plot results
predictions = [quantum_circuit(x, params) for x in x_data]

plt.figure(figsize=(7,4))
plt.plot(cost_history, label='Training Cost')
plt.xlabel('Epoch')
plt.ylabel('MSE Cost')
plt.title('Cost During Training')
plt.legend()
plt.show()

plt.figure(figsize=(7,4))
plt.scatter(x_data, y_data, label='sin(x)', color='red')
plt.scatter(x_data, predictions, label='Circuit Output', color='blue')
plt.xlabel('x')
plt.ylabel('Value')
plt.title('Single-Qubit Circuit vs. sin(x)')
plt.legend()
plt.show()

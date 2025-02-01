import numpy as np
import matplotlib.pyplot as plt

# Generate data
X = np.linspace(-3, 3, 128).reshape(-1, 1)
Y = X**3 - 3*X + np.random.normal(0, 1, X.shape)

# Create the plot
plt.figure(figsize=(10, 6))
plt.scatter(X, Y, color='blue', alpha=0.6)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Data Points')
plt.grid(True)
plt.show()
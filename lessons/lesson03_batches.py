"""NumPy performs the same calculation for several samples simultaneously."""

import numpy as np

inputs = np.array([
    [1.0, 2.0, 3.0, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8],
])
weights_by_neuron = np.array([
    [0.2, 0.8, -0.5, 1.0],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87],
])
biases = np.array([2.0, 3.0, 0.5])

# Transpose changes (neurons, inputs) into (inputs, neurons).
# (3 samples, 4 inputs) @ (4 inputs, 3 neurons) -> (3 samples, 3 neurons).
outputs = inputs @ weights_by_neuron.T + biases
print('Input shape:', inputs.shape)
print('Weight shape after transpose:', weights_by_neuron.T.shape)
print('Outputs:\n', outputs)
np.testing.assert_allclose(outputs, [[4.8, 1.21, 2.385],
                                    [8.9, -1.81, 0.2],
                                    [1.41, 1.051, 0.026]])

# network.Dense stores weights in the transposed layout from the beginning.

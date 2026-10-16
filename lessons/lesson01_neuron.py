"""Start here: one neuron, three inputs, no NumPy required."""

inputs = [1.0, 2.0, 3.0]
weights = [0.2, 0.8, -0.5]
bias = 2.0

# Each weight controls how strongly its matching input affects this neuron.
output = inputs[0] * weights[0] + inputs[1] * weights[1] + inputs[2] * weights[2] + bias
print('Inputs:', inputs)
print('Weights:', weights)
print('Bias:', bias)
print('Output:', output)  # 2.3
assert abs(output - 2.3) < 1e-12

# Try: increase only bias by 1. The output should increase by exactly 1.

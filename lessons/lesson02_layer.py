"""Three neurons see the same four inputs, each with its own weights and bias."""

inputs = [1.0, 2.0, 3.0, 2.5]
weights = [
    [0.2, 0.8, -0.5, 1.0],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87],
]
biases = [2.0, 3.0, 0.5]

outputs = []
for neuron_weights, bias in zip(weights, biases):
    neuron_output = bias
    for input_value, weight in zip(inputs, neuron_weights):
        neuron_output += input_value * weight
    outputs.append(neuron_output)

print('Layer output:', outputs)
assert all(abs(actual - expected) < 1e-12
           for actual, expected in zip(outputs, [4.8, 1.21, 2.385]))

# Try: change the first neuron's bias. Only its output should change.

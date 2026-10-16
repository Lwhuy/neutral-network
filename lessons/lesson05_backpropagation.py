"""One plain SGD step, before the full training script introduces Adam."""

import numpy as np
from network import Classifier, softmax_cross_entropy

model = Classifier(seed=0)
inputs = np.array([[0.2, 0.8], [-0.5, 0.3], [0.7, -0.4]])
labels = np.array([0, 1, 2])

loss_before, _, dlogits = softmax_cross_entropy(model.forward(inputs), labels)
model.backward(dlogits)

# A gradient says how loss changes when a parameter increases a little.
# Subtracting it moves parameters in the direction of smaller loss.
learning_rate = 0.01
for layer in (model.hidden, model.output):
    layer.weights -= learning_rate * layer.dweights
    layer.biases -= learning_rate * layer.dbiases

loss_after, _, _ = softmax_cross_entropy(model.forward(inputs), labels)
print(f'Loss before one SGD step: {loss_before:.6f}')
print(f'Loss after one SGD step:  {loss_after:.6f}')
assert loss_after < loss_before

# Try: raise learning_rate substantially. Too large a step can increase loss.

"""ReLU adds nonlinearity; softmax gives probabilities; loss measures error."""

import numpy as np
from network import ReLU, softmax_cross_entropy

values = np.array([[-1.0, 0.0, 2.0]])
print('Before ReLU:', values)
print('After ReLU: ', ReLU().forward(values))

logits = np.array([[2.0, 1.0, 0.1], [0.1, 1.0, 2.0]])
labels = np.array([0, 2])  # First sample belongs to class 0, second to class 2.
loss, probabilities, _ = softmax_cross_entropy(logits, labels)
print('\nClass probabilities:\n', probabilities)
print('Predictions:', probabilities.argmax(axis=1))
print('Correct classes:', labels)
print('Mean cross-entropy loss:', round(loss, 4))
np.testing.assert_allclose(probabilities.sum(axis=1), 1)
np.testing.assert_array_equal(probabilities.argmax(axis=1), labels)

# Try: swap the labels to [2, 0]. Confident wrong predictions increase loss.

"""Small, readable building blocks. Read the numbered lessons first.

Array convention: rows are samples, columns are features (or neurons).
Only NumPy does the array arithmetic; all learning logic lives here.
"""

import numpy as np


class Dense:
    """Every input connects to every neuron: output = inputs @ weights + bias."""

    def __init__(self, n_inputs, n_neurons, rng):
        # He initialization keeps the scale useful for layers followed by ReLU.
        self.weights = rng.normal(0, np.sqrt(2 / n_inputs), (n_inputs, n_neurons))
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        return inputs @ self.weights + self.biases

    def backward(self, dvalues):
        # Sum each parameter's contribution across all samples in the batch.
        self.dweights = self.inputs.T @ dvalues
        self.dbiases = dvalues.sum(axis=0, keepdims=True)
        return dvalues @ self.weights.T


class ReLU:
    """A bend in the function lets stacked layers learn nonlinear patterns."""

    def forward(self, inputs):
        self.inputs = inputs
        return np.maximum(0, inputs)

    def backward(self, dvalues):
        return dvalues * (self.inputs > 0)


def softmax(logits):
    """Turn raw class scores into probabilities, independently for each row."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def softmax_cross_entropy(logits, labels):
    """Return mean loss, probabilities, and gradient with respect to logits.

    labels: one integer class index per row, from 0 to n_classes - 1.
    Computing log probabilities directly avoids log(0) after underflow.
    """
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_probabilities = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    probabilities = np.exp(log_probabilities)
    n_samples = len(logits)
    loss = -log_probabilities[np.arange(n_samples), labels].mean()
    # Softmax and cross-entropy simplify together to (probability - target) / N.
    dlogits = probabilities.copy()
    dlogits[np.arange(n_samples), labels] -= 1
    dlogits /= n_samples
    return loss, probabilities, dlogits


def spiral_data(samples_per_class=100, classes=3, seed=0):
    """Generate labeled 2D spiral arms locally; no data download needed."""
    if samples_per_class < 1 or classes < 2:
        raise ValueError('Use at least one sample per class and two classes.')
    rng = np.random.default_rng(seed)
    x = np.zeros((samples_per_class * classes, 2))
    y = np.repeat(np.arange(classes), samples_per_class)
    for class_index in range(classes):
        rows = slice(class_index * samples_per_class, (class_index + 1) * samples_per_class)
        radius = np.linspace(0.05, 1, samples_per_class)
        angle = np.linspace(class_index * 4, (class_index + 1) * 4, samples_per_class)
        angle += rng.normal(0, 0.15, samples_per_class)
        x[rows] = np.column_stack((radius * np.sin(angle * 2.5), radius * np.cos(angle * 2.5)))
    return x, y


class Classifier:
    """Two dense layers: 2 coordinates -> 64 hidden neurons -> 3 class scores."""

    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        self.hidden = Dense(2, 64, rng)
        self.relu = ReLU()
        self.output = Dense(64, 3, rng)

    def forward(self, x):
        return self.output.forward(self.relu.forward(self.hidden.forward(x)))

    def backward(self, dlogits):
        self.hidden.backward(self.relu.backward(self.output.backward(dlogits)))

    def predict(self, x):
        return self.forward(x).argmax(axis=1)


class Adam:
    """Track smoothed gradients and their squares to scale parameter updates."""

    def __init__(self, layers, learning_rate=0.02):
        self.learning_rate = learning_rate
        self.step = 0
        self.parameters = []
        for layer in layers:
            for name in ('weights', 'biases'):
                parameter = getattr(layer, name)
                self.parameters.append((layer, name, np.zeros_like(parameter), np.zeros_like(parameter)))

    def update(self):
        self.step += 1
        for layer, name, momentum, velocity in self.parameters:
            gradient = getattr(layer, 'd' + name)
            momentum[:] = 0.9 * momentum + 0.1 * gradient
            velocity[:] = 0.999 * velocity + 0.001 * gradient ** 2
            corrected_momentum = momentum / (1 - 0.9 ** self.step)
            corrected_velocity = velocity / (1 - 0.999 ** self.step)
            parameter = getattr(layer, name)
            parameter -= self.learning_rate * corrected_momentum / (np.sqrt(corrected_velocity) + 1e-7)

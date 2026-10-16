"""Train the complete network: python train.py [--epochs 3000]."""

import argparse
import numpy as np

from network import Adam, Classifier, softmax, softmax_cross_entropy, spiral_data


def train(epochs=3000, verbose=True):
    if epochs < 1:
        raise ValueError('epochs must be at least 1')
    x_train, y_train = spiral_data(seed=0)
    x_test, y_test = spiral_data(seed=99)
    model = Classifier(seed=0)
    optimizer = Adam([model.hidden, model.output])
    history = []

    for epoch in range(epochs + 1):
        # Forward: turn coordinates into scores, then measure how wrong they are.
        loss, probabilities, dlogits = softmax_cross_entropy(model.forward(x_train), y_train)
        accuracy = np.mean(probabilities.argmax(axis=1) == y_train)
        history.append((float(loss), float(accuracy)))
        if verbose and (epoch % 500 == 0 or epoch == epochs):
            print(f'epoch {epoch:4d} | train loss {loss:.4f} | train accuracy {accuracy:.1%}')
        if epoch == epochs:
            break
        # Backward: assign responsibility; update: move downhill a small step.
        model.backward(dlogits)
        optimizer.update()

    if verbose:
        test_loss, probabilities, _ = softmax_cross_entropy(model.forward(x_test), y_test)
        test_accuracy = np.mean(probabilities.argmax(axis=1) == y_test)
        print(f'\nHeld-out loss {test_loss:.4f} | held-out accuracy {test_accuracy:.1%}')
        print('First five held-out predictions:')
        for point, label in zip(x_test[:5], y_test[:5]):
            probabilities = softmax(model.forward(point[None, :]))[0]
            print(f'  x={point.round(3)} | probabilities={probabilities.round(3)} '
                  f'| predicted={probabilities.argmax()} | actual={label}')
    return model, history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=3000)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('--epochs must be at least 1')
    train(epochs=args.epochs)

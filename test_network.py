"""Run with: python -m unittest -v. Tests are also worked math examples."""

import unittest
import numpy as np

import network


class NetworkTests(unittest.TestCase):
    def test_dense_forward_and_backward_by_hand(self):
        layer = network.Dense(2, 2, np.random.default_rng(0))
        layer.weights[:] = [[1, 2], [3, 4]]
        layer.biases[:] = [0.5, -0.5]
        np.testing.assert_allclose(layer.forward(np.array([[2., 3.]])), [[11.5, 15.5]])
        np.testing.assert_allclose(layer.backward(np.array([[1., -1.]])), [[-1, -1]])
        np.testing.assert_allclose(layer.dweights, [[2, -2], [3, -3]])
        np.testing.assert_allclose(layer.dbiases, [[1, -1]])

    def test_relu_masks_negative_and_zero_inputs(self):
        relu = network.ReLU()
        np.testing.assert_array_equal(relu.forward(np.array([[-2., 0., 3.]])), [[0, 0, 3]])
        np.testing.assert_array_equal(relu.backward(np.ones((1, 3))), [[0, 0, 1]])

    def test_softmax_stays_finite_for_large_logits(self):
        probabilities = network.softmax(np.array([[1000., 1001., 999.], [-1000., 0., 1000.]]))
        self.assertTrue(np.isfinite(probabilities).all())
        np.testing.assert_allclose(probabilities.sum(axis=1), 1)
        np.testing.assert_allclose(probabilities[0], [0.24472847, 0.66524096, 0.09003057])

    def test_loss_and_gradient_match_finite_differences(self):
        logits = np.array([[0.2, -0.3, 0.7], [1.1, 0.1, -0.2]])
        labels = np.array([2, 0])
        loss, probabilities, gradient = network.softmax_cross_entropy(logits, labels)
        self.assertAlmostEqual(loss, -np.log(probabilities[[0, 1], labels]).mean())
        numerical = np.zeros_like(logits)
        for index in np.ndindex(logits.shape):
            plus, minus = logits.copy(), logits.copy()
            plus[index] += 1e-5
            minus[index] -= 1e-5
            numerical[index] = (network.softmax_cross_entropy(plus, labels)[0]
                                - network.softmax_cross_entropy(minus, labels)[0]) / 2e-5
        np.testing.assert_allclose(gradient, numerical, atol=1e-8)
        extreme_loss, _, _ = network.softmax_cross_entropy(np.array([[1000., -1000.]]), np.array([1]))
        self.assertAlmostEqual(extreme_loss, 2000.)

    def test_spiral_is_reproducible_and_balanced(self):
        x, y = network.spiral_data(20, seed=4)
        x_again, y_again = network.spiral_data(20, seed=4)
        self.assertEqual(x.shape, (60, 2))
        np.testing.assert_array_equal(np.bincount(y), [20, 20, 20])
        np.testing.assert_array_equal(x, x_again)
        np.testing.assert_array_equal(y, y_again)

    def test_training_learns_unseen_spiral_points(self):
        from train import train
        model, history = train(epochs=2000, verbose=False)
        self.assertLess(history[-1][0], history[0][0] * 0.6)
        x, y = network.spiral_data(100, seed=99)
        accuracy = np.mean(model.predict(x) == y)
        self.assertGreater(accuracy, 0.7)


if __name__ == '__main__':
    unittest.main()

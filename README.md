# Neural networks from scratch

A small Python learning project inspired by Sentdex's
[Neural Networks from Scratch, Part 1](https://www.youtube.com/watch?v=Wo5dMEP_BbI)
and the progression described on [NNFS](https://nnfs.io/).
Start with a single neuron and finish with a classifier that learns three spiral arms.

This is an original companion project, not an official transcript or an episode-by-episode
copy. Lesson numbers below are this project's reading order. Backpropagation and Adam
extend the starting video into a complete training example.

## Run it

Python 3.9 or newer. From this folder on macOS/Linux:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m lessons.lesson01_neuron
python train.py
python -m unittest -v
```

A virtual environment with NumPy is already prepared in this workspace. You can
start with `source .venv/bin/activate`. NumPy 1.26.4 is pinned for the provided
Python 3.9 environment; use Python 3.9–3.12 with this pin.

NumPy is the only external dependency. There is no TensorFlow, PyTorch,
automatic differentiation, or external dataset. Training uses your CPU.

## Read in this order

| Step | File / command | What to look for |
| --- | --- | --- |
| 1 | `python -m lessons.lesson01_neuron` | One weighted sum plus a bias; output `2.3` |
| 2 | `python -m lessons.lesson02_layer` | Several neurons share inputs; output `[4.8, 1.21, 2.385]` |
| 3 | `python -m lessons.lesson03_batches` | Replace loops with a matrix multiplication |
| 4 | `python -m lessons.lesson04_activations_and_loss` | ReLU, probabilities, and measuring mistakes |
| 5 | `python -m lessons.lesson05_backpropagation` | Compute gradients and take one SGD step |
| 6 | [network.py](network.py) | All the reusable math, including Adam |
| 7 | [train.py](train.py) | Repeat forward, backward, and update; evaluate unseen points |
| 8 | [test_network.py](test_network.py) | Hand calculations, numerical gradients, learning checks |

Read [GUIDE.md](GUIDE.md) alongside the code for the math, array shapes, and experiments.
Run lesson commands from the project root so imports resolve correctly.

## What training does

Each point has two input features: its horizontal and vertical coordinate.
Its label is the spiral arm it belongs to: 0, 1, or 2.
The network is `2 inputs -> 64 ReLU neurons -> 3 class scores -> softmax`.
Training adjusts weights and biases to make the correct class more probable.

```sh
python train.py --epochs 3000
```

The script prints loss and accuracy every 500 updates, evaluates a separately
generated set of points, and shows example predictions. Epoch 0 is the untrained
baseline. Lower loss and higher held-out accuracy show that learning occurred.
Fixed seeds make runs repeatable, with possible small differences across platforms.
The held-out seed is fixed for this demo; if you tune against it repeatedly,
reserve another untouched seed for a final evaluation.

This is a complete small classifier, not a general machine learning framework.
It does not save model files; each invocation trains from the fixed seed again.

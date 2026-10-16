# Reading the network

## 1. A neuron is a calculation

Start with `lessons/lesson01_neuron.py`. A feature is one number describing a
sample. A weight controls how that feature contributes to a neuron's output.
A bias shifts the result independently of the features.

```text
z = x1*w1 + x2*w2 + x3*w3 + b
  = 1*0.2 + 2*0.8 + 3*(-0.5) + 2
  = 2.3
```

`z` is a raw score, not yet a probability. The neuron does not learn just by
calculating this result. Learning comes later, when we change its parameters
(the weights and bias) using feedback from examples.

Experiment: increase the third input by 1. Its weight is -0.5, so the output
falls by 0.5. Increase the bias by 1 and the output increases by 1.

## 2. A layer is several neurons

In lesson 2 every neuron receives the same input values. Each has its own
weights and bias, so it can respond to a different combination of features.
The result is one output per neuron. Those outputs can become the inputs of
another layer.

Lesson 3 handles a batch: several samples at once. A matrix multiplication
performs all the weighted sums. NumPy's `@` is matrix multiplication;
`*` multiplies corresponding elements instead.

Use rows for samples and columns for features throughout this project:

| Array | Shape in full training | Meaning |
| --- | --- | --- |
| `x_train` | `(300, 2)` | 300 points, two coordinates each |
| hidden weights | `(2, 64)` | Two weights for each of 64 neurons |
| hidden biases | `(1, 64)` | One bias for each hidden neuron |
| hidden output | `(300, 64)` | 64 values for every point |
| output weights | `(64, 3)` | Hidden values combined into three scores |
| logits / probabilities | `(300, 3)` | Three scores / probabilities per point |
| labels | `(300,)` | One correct class index per point |

`(N, D) @ (D, H)` produces `(N, H)`. NumPy broadcasts the `(1, H)` bias
over all N rows. The batch dimension does not change the number of parameters.
Our classifier has `2*64 + 64 + 64*3 + 3 = 387` trainable numbers.

## 3. Why activations matter

Stacking weighted sums alone still gives a weighted sum. For example,
`(X @ W1 + b1) @ W2 + b2` can be rearranged into `X @ W + b`.
It cannot model the bends needed to separate spiral arms.

ReLU is `max(0, z)`: negative values become zero and positive values pass through.
That bend lets the network represent nonlinear boundaries. The last dense layer
outputs raw class scores, often called logits. We do not apply ReLU to those scores.

Softmax turns each row of scores into nonnegative probabilities that sum to 1:

```text
p(class j) = exp(score j) / sum(exp(all scores in this row))
```

Subtracting the largest score before exponentiating preserves the result and
avoids overflow. The predicted class is the column with the largest probability.
These probabilities are model estimates; a confident prediction can still be wrong.

## 4. Loss provides feedback

Accuracy asks whether the largest score belongs to the right class. It does not
measure how close the prediction was or provide a useful derivative for training.
Cross-entropy uses the probability assigned to the correct class instead:

```text
one sample's loss = -log(probability of the correct class)
batch loss = average of all sample losses
```

A correct-class probability of 0.9 gives about 0.105 loss; 0.1 gives about 2.303.
For three equally likely classes the loss is `log(3)`, about 1.099.
An untrained random network need not give exactly equal probabilities.

`softmax_cross_entropy` calculates log probabilities directly using shifted scores
and log-sum-exp. This keeps a confidently wrong prediction's loss finite even
when its probability rounds down to zero. Read the extreme-score test to see why.

## 5. Backpropagation assigns responsibility

A derivative tells us how a small change affects something else. A gradient
collects those derivatives for an array. Backpropagation applies the chain rule
from loss back through each layer, telling us how each parameter affects loss.

For softmax combined with cross-entropy, the derivative with respect to scores is:

```text
dlogits = (probabilities - one_hot_correct_labels) / number_of_samples
```

We avoid constructing a full one-hot array: copy probabilities and subtract 1
at the correct column of each row, then divide by batch size. We average here
once; the later layers must not divide again.

For a dense layer `Z = X @ W + b`, given the upstream gradient `dZ`:

```text
dW = X.T @ dZ                  # effect on each weight
db = sum(dZ, over sample rows)  # shared bias collects all sample contributions
dX = dZ @ W.T                  # send responsibility to the preceding layer
```

ReLU passes gradients where its input was positive and blocks them elsewhere.
At zero we use a derivative of zero. Forward methods remember their inputs
because backward methods need them. Call backward after the corresponding
forward pass, before another forward pass overwrites that stored information.

The test suite compares an analytic loss gradient against finite differences:
perturb a score by a tiny amount in either direction and measure loss changes.
It also checks a dense layer against hand-computed values.

## 6. Updating parameters creates learning

Lesson 5 uses stochastic gradient descent (SGD):

```text
parameter = parameter - learning_rate * gradient
```

The minus sign moves downhill. Too small a learning rate makes progress slow;
too large a rate can jump past good solutions or make training diverge.
Our examples use the full training batch each update, so this is the full-batch
form of gradient descent even though the update is commonly called SGD.

The full training script uses Adam to learn the spiral more quickly. Adam keeps
a moving average of gradients (momentum) and of squared gradients (velocity).
It corrects their initial bias toward zero, then scales each parameter's step:

```text
m = 0.9*m + 0.1*gradient
v = 0.999*v + 0.001*gradient**2
m_hat = m / (1 - 0.9**step)
v_hat = v / (1 - 0.999**step)
parameter -= learning_rate * m_hat / (sqrt(v_hat) + epsilon)
```

Epsilon prevents division by zero. All parameters share the same step count,
which increases once per batch update. This is written explicitly in `Adam`;
NumPy does not calculate gradients or optimize parameters for us.

## 7. Read the complete training loop

Follow `train.py` in this order:

1. Generate labeled training points and separately generated held-out points.
2. Initialize weights randomly; zeros would make hidden neurons too alike.
3. Forward the training points through dense, ReLU, and dense layers.
4. Calculate loss, probabilities, and the initial gradient.
5. Backpropagate that gradient through the output layer, ReLU, and hidden layer.
6. Update weights and biases with Adam; repeat.
7. Evaluate held-out points without updating parameters.

The generator assigns labels while creating the arms. The model receives only
coordinates as inputs, never labels. Labels enter the loss during training.
Both sets come from the same synthetic process but have different random noise.
Nearby spiral arms and overlapping points mean perfection is not guaranteed.

## 8. Experiments after the first run

Change one thing at a time and compare both training and held-out results:

- Run `python train.py --epochs 10`, then `--epochs 3000`: how does learning change?
- In `Classifier`, change hidden width from 64 to 8 in both dense constructors:
  can the smaller network represent the same boundary?
- Lower Adam's learning rate from 0.02 to 0.002: does it need more updates?
- Replace ReLU's forward result with `inputs` and backward result with `dvalues`:
  why does a stack of linear layers struggle with the spiral?
- Increase the generator's noise from 0.15 to 0.3: why might loss remain higher?

Restore changes before running the test suite. Its training check uses fixed
settings and expects held-out accuracy above 70%; it is a regression check for
this example, not a promise about other datasets.

## Boundaries of this project

The reusable classes expect finite NumPy matrices with the shapes shown above,
and integer class indices within range. This is teaching code, without a public
data ingestion API. The training entry point validates the epoch count, and the
data generator rejects empty class sizes. The classifier is deliberately fixed
to two input features and three classes.

There are no checkpoints, GPU support, dropout, regularization, mini-batching,
or hyperparameter search. Those are follow-up topics after this forward/backward
loop makes sense. For the original material, see
[Sentdex's first video](https://www.youtube.com/watch?v=Wo5dMEP_BbI),
[NNFS](https://nnfs.io/), and the
[official book example repository](https://github.com/Sentdex/nnfs_book).

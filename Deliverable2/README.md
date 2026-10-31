# MLP_ReSoft: A Neural Network from Scratch with NumPy

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![NumPy](https://img.shields.io/badge/NumPy-from%20scratch-013243)
![Udacity](https://img.shields.io/badge/Udacity-Intro%20to%20Deep%20Learning%20with%20PyTorch-02b3e4)

`MLP_ReSoft` is a multilayer perceptron for **multiclass classification**, built entirely from scratch with NumPy. It combines a **ReLU** hidden layer with a **Softmax** output layer, and implements the forward pass, cross-entropy loss, backpropagation and gradient descent by hand. No deep learning framework is used.

I built this after completing Udacity's **Intro to Deep Learning with PyTorch** course, to make sure I understood what PyTorch automates under the hood.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Math Behind the Model](#math-behind-the-model)
- [Usage](#usage)
- [What I Learned](#what-i-learned)
- [Reference](#Reference)

---

## Features

- Single hidden layer network with configurable size
- **ReLU** activation (and its derivative) for the hidden layer
- Numerically stable **Softmax** for the output layer
- **Categorical cross-entropy** loss
- Manual **backpropagation** and gradient descent updates
- Scikit-learn style API: `fit`, `predict`, `predict_proba`, `accuracy_score`

## Architecture

```
Input (n_inputs) -> Linear (w_1, b_1) -> ReLU -> Linear (w_2, b_2) -> Softmax -> Class probabilities (n_classes)
```

| Layer | Parameters | Shape | Activation |
|-------|-----------|-------|------------|
| Hidden | `w_1`, `b_1` | `(n_inputs, n_hidden)`, `(1, n_hidden)` | ReLU |
| Output | `w_2`, `b_2` | `(n_hidden, n_classes)`, `(1, n_classes)` | Softmax |

Weights are initialized from a standard normal distribution and biases start at zero.

## Math Behind the Model

### Forward pass

```
z_1 = X . w_1 + b_1
a_1 = ReLU(z_1)
z_2 = a_1 . w_2 + b_2
output = Softmax(z_2)
```

### ReLU

```
ReLU(z) = max(0, z)
ReLU'(z) = 1 if z > 0 else 0
```

### Softmax (numerically stable)

```python
def Softmax(self, z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)
```

Subtracting the row maximum before exponentiating prevents overflow without changing the result.

### Categorical cross-entropy loss

```
L = -mean( log( output[i, y_i] ) )
```

### Backpropagation

With Softmax and cross-entropy combined, the output gradient simplifies to the predicted probabilities minus the one-hot labels:

```
dz_2 = (output - one_hot(y)) / n
dw_2 = a_1^T . dz_2
db_2 = sum(dz_2)

da_1 = dz_2 . w_2^T
dz_1 = da_1 * ReLU'(z_1)
dw_1 = X^T . dz_1
db_1 = sum(dz_1)
```

### Gradient descent update

```
w = w - learning_rate * dw
b = b - learning_rate * db
```

# XOR data: each row is one sample, each column is one feature
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
y = pd.Series([0, 1, 1, 0])

# Create and train the model
model = MLP_ReSoft(n_inputs=2, n_hidden=8, n_classes=2,
                   learning_rate=0.5, epochs=10000)
model.fit(X, y)

# Predict
pred = model.predict(X)
print(pred)
print("Accuracy:", model.accuracy_score(y.to_numpy(), pred))
```

> `y` is passed as a pandas Series because the backpropagation step uses `y.nunique()` to build the one-hot labels.

During training, `fit` prints the epoch number and the cross-entropy loss, along with progress messages for each stage (forward pass, loss calculation, backpropagation).

## API Reference

### `MLP_ReSoft(n_inputs, n_hidden, n_classes, learning_rate=0.01, epochs=100)`

| Parameter | Description |
|-----------|-------------|
| `n_inputs` | Number of input features |
| `n_hidden` | Number of neurons in the hidden layer |
| `n_classes` | Number of output classes |
| `learning_rate` | Gradient descent step size (default `0.01`) |
| `epochs` | Number of training iterations (default `100`) |

### Methods

| Method | Description |
|--------|-------------|
| `fit(X, y)` | Trains the model with full-batch gradient descent |
| `forward_pass(X)` | Runs the forward pass and returns class probabilities |
| `predict_proba(X)` | Returns Softmax probabilities for each class |
| `predict(X)` | Returns the predicted class labels |
| `accuracy_score(y, y_pred)` | Returns the fraction of correct predictions |
| `ReLU(z)` / `ReLU_derivative(z)` | Hidden layer activation and its derivative |
| `Softmax(z)` | Output layer activation |
| `CategoryCrossEntropy(y, output)` | Computes the cross-entropy loss |
| `backprop(X, y)` | Computes gradients and updates weights and biases |


## What I Learned

- How a forward pass works using plain matrix operations
- Why non-linear activations like ReLU are essential (a linear model cannot solve XOR)
- How Softmax converts raw scores into probabilities, and why it needs numerical stability
- How to derive and implement backpropagation by hand
- Why the Softmax and cross-entropy gradient simplifies to `output - y`
- How learning rate, weight initialization and epochs affect training
- What frameworks like PyTorch automate behind the scenes


## Reference

- [Udacity](https://www.udacity.com/) for the *Intro to Deep Learning with PyTorch* course
- [NumPy](https://numpy.org/) documentation
- Flexisaf 

## License


*Built by `<Ayodele Ogunmakinde>` · [GitHub](https://github.com/<hexa-source>)*
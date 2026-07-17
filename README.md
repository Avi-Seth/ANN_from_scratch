# ANN From Scratch

This is my attempt at understanding what actually goes on under the hood of a neural network by building one completely from scratch in pure Python. No NumPy, no PyTorch, not even the `math` module. Even the sigmoid function is homemade: I approximate e^x with a 30-term Taylor series and evaluate it using synthetic division (yes, the thing from algebra class actually turned out to be useful).

The point of this project is learning, not writing production code. And honestly? Mission accomplished. Forward passes, backpropagation, gradient descent... I now get how all of it works because I had to derive and code every step myself.

## How it works

```python
net = CSN_NeuralNetwork("baseline", [5, 4, 3, 2], ["A", "B"])
net.train(batch, labels, 0.1)   # one gradient descent step
net.test([0, 0, 1, 1, 1])       # forward pass, returns activations + label
```

- Layers are defined by a simple list of sizes
- Weights and biases are plain nested Python lists
- Training uses backpropagation with mean squared error
- Run the demo with `python3 nn.py` (needs nothing but Python 3)

The backprop math is actually correct. The gradients match numerical differentiation, which was honestly the most satisfying part of the whole project.

## Known limitations (and yes, some bugs)

I am aware of a few things that are broken or limited here. I might not fix them, because the goal was to understand the math, not to ship polished software:

- `test()` has an indentation bug that makes it crash whenever the predicted class is not the first output label
- My Taylor series sigmoid falls apart for inputs below about -8 (it can even return negative "probabilities", which is pretty funny for a probability)
- The hidden layers have no activation function, so the network is secretly just a linear model with extra steps
- All weights start at 0.5, so hidden neurons are identical clones of each other and stay that way
- Pure Python loops, so it is slow. Do not throw MNIST at this thing

## License

GPL-3.0

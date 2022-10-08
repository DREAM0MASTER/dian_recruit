import numpy as np
from .modules import Module


class Sigmoid(Module):

    def __init__(self, a:np.array):
        self.a = a

    def forward(self, x):

        # TODO Implement forward propogation
        # of sigmoid function.

        ...
        return 1/(1+np.exp(-x))

        # End of todo

    def backward(self, dy):

        # TODO Implement backward propogation
        # of sigmoid function.

        ...
        dx = np.multiply(np.multiply(self.forward(self.a), 1-self.forward(self.a)), dy)
        return dx

        # End of todo


class Tanh(Module):

    def forward(self, x):

        # TODO Implement forward propogation
        # of tanh function.

        ...
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

        # End of todo

    def backward(self, dy):

        # TODO Implement backward propogation
        # of tanh function.

        ...

        # End of todo


class ReLU(Module):

    def forward(self, x):

        # TODO Implement forward propogation
        # of ReLU function.

        ...
        return np.maximum(x, 0)

        # End of todo

    def backward(self, dy):

        # TODO Implement backward propogation
        # of ReLU function.

        ...

        # End of todo


class Softmax(Module):

    def forward(self, x):

        # TODO Implement forward propogation
        # of Softmax function.

        ...
        exp_x = np.exp(x)
        sum_exp_x = np.sum(exp_x)
        y = exp_x / sum_exp_x
        return y
        # End of todo

    def backward(self, dy):

        # Omitted.
        ...


class Loss(object):
    """
    Usage:
        >>> criterion = Loss(n_classes)
        >>> ...
        >>> for epoch in n_epochs:
        ...     ...
        ...     probs = model(x)
        ...     loss = criterion(probs, target)
        ...     model.backward(loss.backward())
        ...     ...
    """
    def __init__(self, n_classes):
        self.n_classes = n_classes

    def __call__(self, probs, targets):
        self.probs = probs
        self.targets = targets
        ...
        return self

    def backward(self):
        ...


class SoftmaxLoss(Loss):

    def __call__(self, probs, targets):

        # TODO Calculate softmax loss.

        ...

        # End of todo

    def backward(self):

        # TODO Implement backward propogation
        # of softmax loss function.

        ...

        # End of todo


class CrossEntropyLoss(Loss):

    def __call__(self, probs, targets):

        # TODO Calculate cross-entropy loss.

        ...
        self.probs = probs
        n = len(probs)
        self.n = n
        y = np.zeros((n, 10))
        for i in range(0,n):
            y[i][int(targets[i])] = 1
        self.y = y
        return np.sum(np.multiply(probs-y,probs-y))/n

        # End of todo

    def backward(self):

        # TODO Implement backward propogation
        # of cross-entropy loss function.

        ...
        return (self.probs-self.y)/self.n


        # End of todo

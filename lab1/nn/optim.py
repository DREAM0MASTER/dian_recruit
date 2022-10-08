import numpy as np

import module_test
from .tensor import Tensor
from .modules import Module
import mnist


class Optim(object):

    def __init__(self, module, lr):
        self.module = module
        self.lr = lr

    def step(self):
        self._step_module(self.module)

    def _step_module(self, module):

        # TODO Traverse the attributes of `self.module`,
        # if is `Tensor`, call `self._update_weight()`,
        # else if is `Module` or `List` of `Module`,
        # call `self._step_module()` recursively.

        ...
        dw1 = np.zeros((2,1,3,3))
        for i in range(0,2):
            for j in range(0, module.n):
                dw1[i] += mnist.convolution(module.delta1[j][i], module.a0[j])
        dw2 = np.zeros((2,2,3,3))
        for i in range(0,2):
            for j in range(0, module.n):
                dw2[i] += mnist.convolution(module.delta3[j][i], module.a2[j])
        dw3 = np.zeros((73,10))
        a4 = module.a4
        a4 = a4.reshape((np.shape(a4)[0], np.shape(a4)[1] * np.shape(a4)[2] * np.shape(a4)[3]))
        dw3[1:] = np.dot(a4.T, module.delta5)*module.n
        dw3[0] = np.sum(module.delta5, axis=0)

        module.conv2d1.conv_chan -= self.lr*dw1
        module.conv2d2.conv_chan -= self.lr*dw2
        module.linear1.w -= self.lr*dw3


        # End of todo

    def _update_weight(self, tensor):
        tensor -= self.lr * tensor.grad


class SGD(Optim):

    def __init__(self, module:mnist.Model, lr, momentum: float=0):
        super(SGD, self).__init__(module, lr)
        self.momentum = momentum

    def _update_weight(self, tensor):

        # TODO Update the weight of tensor
        # in SGD manner.

        ...

        # End of todo


class Adam(Optim):

    def __init__(self, module, lr):
        super(Adam, self).__init__(module, lr)

        # TODO Initialize the attributes
        # of Adam optimizer.

        ...

        # End of todo

    def _update_weight(self, tensor):

        # TODO Update the weight of
        # tensor in Adam manner.

        ...

        # End of todo



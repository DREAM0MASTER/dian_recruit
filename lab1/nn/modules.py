import numpy as np
from itertools import product

import nn
from . import tensor


class Module(object):
    """Base class for all neural network modules.
    """
    def __init__(self) -> None:
        """If a module behaves different between training and testing,
        its init method should inherit from this one."""
        self.training = True

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Defines calling forward method at every call.
        Should not be overridden by subclasses.
        """
        return self.forward(x)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Defines the forward propagation of the module performed at every call.
        Should be overridden by all subclasses.
        """
        ...

    def backward(self, dy: np.ndarray) -> np.ndarray:
        """Defines the backward propagation of the module.
        """
        return dy

    def train(self):
        """Sets the mode of the module to training.
        Should not be overridden by subclasses.
        """
        if 'training' in vars(self):
            self.training = True
        for attr in vars(self).values():
            if isinstance(attr, Module):
                Module.train()

    def eval(self):
        """Sets the mode of the module to eval.
        Should not be overridden by subclasses.
        """
        if 'training' in vars(self):
            self.training = False
        for attr in vars(self).values():
            if isinstance(attr, Module):
                Module.eval()


class Linear(Module):

    def __init__(self, in_length: int, out_length: int):
        """Module which applies linear transformation to input.

        Args:
            in_length: L_in from expected input shape (N, L_in).
            out_length: L_out from output shape (N, L_out).
        """

        # w[0] for bias and w[1:] for weight
        self.w = tensor.tensor((in_length + 1, out_length))

    def forward(self, x):
        """Forward propagation of linear module.

        Args:
            x: input of shape (N, L_in).
        Returns:
            out: output of shape (N, L_out).
        """

        # TODO Implement forward propogation
        # of linear module.

        ...
        N = np.shape(x)[0]
        return np.dot(x, self.w[1:]) + np.tile(self.w[0],(N,1))

        # End of todo

    def backward(self, dy):
        """Backward propagation of linear module.

        Args:
            dy: output delta of shape (N, L_out).
        Returns:
            dx: input delta of shape (N, L_in).
        """

        # TODO Implement backward propogation
        # of linear module.

        ...
        return np.dot(self.w[1:], dy.T).T

        # End of todo


class BatchNorm1d(Module):

    def __init__(self, length: int, momentum: float=0.9):
        """Module which applies batch normalization to input.

        Args:
            length: L from expected input shape (N, L).
            momentum: default 0.9.
        """
        super(BatchNorm1d, self).__init__()

        # TODO Initialize the attributes
        # of 1d batchnorm module.

        ...
        self.length = length
        self.momentum = momentum
        self.epsi = 0.01
        self.beta = 0
        self.gama = 1


        # End of todo

    def forward(self, x):
        """Forward propagation of batch norm module.

        Args:
            x: input of shape (N, L).
        Returns:
            out: output of shape (N, L).
        """

        # TODO Implement forward propogation
        # of 1d batchnorm module.

        ...
        x_mean = x.mean()
        x_var = x.var()
        x_normalized = (x-x_mean)/np.sqrt(x_var+self.epsi)
        return x_normalized*self.gama + self.beta

        # End of todo

    def backward(self, dy):
        """Backward propagation of batch norm module.

        Args:
            dy: output delta of shape (N, L).
        Returns:
            dx: input delta of shape (N, L).
        """

        # TODO Implement backward propogation
        # of 1d batchnorm module.

        ...


        # End of todo


class Conv2d(Module):

    def __init__(self, in_channels: int, channels: int, kernel_size: int=3,
                 stride: int=1, padding: int=0, bias: bool=True):
        """Module which applies 2D convolution to input.

        Args:
            in_channels: C_in from expected input shape (B, C_in, H_in, W_in).
            channels: C_out from output shape (B, C_out, H_out, W_out).
            kernel_size: default 3.
            stride: default 1.
            padding: default 0.
        """

        # TODO Initialize the attributes
        # of 2d convolution module.

        ...
        self.in_channels=in_channels
        self.channels=channels
        self.kernel_size=kernel_size
        self.stride=stride
        self.padding=padding
        self.bias=bias
        self.conv_chan = tensor.tensor((2, in_channels, kernel_size, kernel_size))

        # End of todo

    def forward(self, x):
        """Forward propagation of convolution module.

        Args:
            x: input of shape (B, C_in, H_in, W_in).
        Returns:
            out: output of shape (B, C_out, H_out, W_out).
        """

        # TODO Implement forward propogation
        # of 2d convolution module.

        ...
        b, c_in, h_in, w_in = np.shape(x)[0], np.shape(x)[1], np.shape(x)[2], np.shape(x)[3]
        c_out, c_i, kh, kw = np.shape(self.conv_chan)[0], np.shape(self.conv_chan)[1], np.shape(self.conv_chan)[2], np.shape(self.conv_chan)[3]
        h_out=h_in-kh+1
        w_out=w_in-kw+1
        res = tensor.zeros((b, c_out, h_out, w_out))
        for i in range(0, b):
            for j in range(0, c_out):
                im2col = nn.Conv2d_im2col.forward([], x[i])
                w_conv = nn.Conv2d_im2col.forward([], self.conv_chan[j])
                res[i][j] += np.reshape(np.dot(im2col, w_conv.T), (h_out, w_out))
        return res

        # End of todo

    def backward(self, dy):
        """Backward propagation of convolution module.

        Args:
            dy: output delta of shape (B, C_out, H_out, W_out).
        Returns:
            dx: input delta of shape (B, C_in, H_in, W_in).
        """

        # TODO Implement backward propogation
        # of 2d convolution module.

        ...
        b, c_in, h_in, w_in = np.shape(dy)[0], np.shape(dy)[1], np.shape(dy)[2], np.shape(dy)[3]
        c_out, c_i, kh, kw = np.shape(self.conv_chan)[0], np.shape(self.conv_chan)[1], np.shape(self.conv_chan)[2], \
                             np.shape(self.conv_chan)[3]


        rot_conv_chan = self.conv_chan
        pad_dy = np.zeros((b, c_in, h_in+4, w_in+4))
        for i in range(0, c_out):
            for j in range(0, c_i):
                rot_conv_chan[i][j] = self.rot(self.conv_chan[i][j])
        for i in range(0, b):
            for j in range(0, c_in):
                pad_dy[i][j] = np.pad(dy[i][j], (2, 2), 'constant')
        pad_dy = np.array(pad_dy)
        rot_conv_chan = np.array(rot_conv_chan)

        h_in = h_in + 4
        w_in = w_in + 4
        h_out = h_in - kh + 1
        w_out = w_in - kw + 1
        res = tensor.zeros((b, c_out, h_out, w_out))
        for i in range(0, b):
            for j in range(0, c_out):
                im2col = nn.Conv2d_im2col.forward([], pad_dy[i], c_i)
                w_conv = nn.Conv2d_im2col.forward([], rot_conv_chan[j], c_i)
                res[i][j] += np.reshape(np.dot(im2col, w_conv.T), (h_out, w_out))
        return res

        # End of todo

    def rot(self, w):
        new_arr = np.reshape(w, len(w) * len(w[0]))
        new_arr = new_arr[::-1]
        new_arr = np.reshape(new_arr, np.shape(w))
        return new_arr


class Conv2d_im2col(Conv2d):

    def forward(self, x, c_i=0):

        # TODO Implement forward propogation of
        # 2d convolution module using im2col method.

        ...
        c_in, h_in, w_in = len(x), len(x[0]), len(x[0][0])
        kernel_size = 3
        stride = 1
        if c_i == 0:
            c_i = c_in
        image_col = []
        for k in range(0, c_in-c_i+1, stride):
            for i in range(0, h_in - kernel_size + 1, stride):
                for j in range(0, w_in - kernel_size + 1, stride):
                    col = x[k:k+c_i , i:i + kernel_size, j:j + kernel_size].reshape([-1])
                    image_col.append(col)
        return np.array(image_col)

        # End of todo


class AvgPool(Module):

    def __init__(self, a: np.array, kernel_size: int=2,
                 stride: int=2, padding: int=0):
        """Module which applies average pooling to input.

        Args:
            kernel_size: default 2.
            stride: default 2.
            padding: default 0.
        """

        # TODO Initialize the attributes
        # of average pooling module.

        ...
        self.kernel_size = 2
        self.stride = 2
        self.padding = 0
        self.a = a
        # End of todo

    def forward(self, x):
        """Forward propagation of average pooling module.

        Args:
            x: input of shape (B, C, H_in, W_in).
        Returns:
            out: output of shape (B, C, H_out, W_out).
        """

        # TODO Implement forward propogation
        # of average pooling module.

        ...
        b, c, h_in, w_in = len(x), len(x[0]), len(x[0][0]), len(x[0][0][0])
        h_out, w_out = int((h_in+1)/self.kernel_size), int((w_in+1)/self.kernel_size)
        res = np.zeros((b, c, h_out, w_out))
        for i in range(0, b):
            for j in range(0, c):
                for k in range(0, h_out):
                    for l in range(0, w_out):
                        if l*2+1==w_in and k*2+1!=h_in:
                            res[i][j][k][l] = res[i][j][k][l] + (x[i][j][k*2][l*2] + x[i][j][k*2+1][l*2] + 0 + 0)/4
                        elif l*2+1!=w_in and k*2+1==h_in:
                            res[i][j][k][l] = res[i][j][k][l] + (x[i][j][k*2][l*2] + 0 + x[i][j][k*2][l*2+1] + 0)/4
                        elif l*2+1==w_in and k*2+1==h_in:
                            res[i][j][k][l] = res[i][j][k][l] + (x[i][j][k*2][l*2] + 0 + 0 + 0)/4
                        else:
                            res[i][j][k][l] = res[i][j][k][l] + (x[i][j][k*2][l*2] + x[i][j][k*2+1][l*2]
                                                                 + x[i][j][k*2][l*2+1] + x[i][j][k*2+1][l*2+1])/4

        return res
        # End of todo

    def backward(self, dy):
        """Backward propagation of average pooling module.

        Args:
            dy: output delta of shape (B, C, H_out, W_out).
        Returns:
            dx: input delta of shape (B, C, H_in, W_in).
        """

        # TODO Implement backward propogation
        # of average pooling module.

        ...
        b, c, h_out, w_out = len(self.a), len(self.a[0]), len(self.a[0][0]), len(self.a[0][0][0])
        res = np.zeros((b, c, h_out, w_out))
        for i in range(0, b):
            for j in range(0, c):
                for k in range(c, h_out):
                    for l in range(0, w_out):
                        res[i][j][k][l] = dy[i][j][int(k/2)][int(l/2)]/4
        return res

        # End of todo


class MaxPool(Module):

    def __init__(self, kernel_size: int=2,
                 stride: int=2, padding: int=0):
        """Module which applies max pooling to input.

        Args:
            kernel_size: default 2.
            stride: default 2.
            padding: default 0.
        """

        # TODO Initialize the attributes
        # of maximum pooling module.

        ...
        self.kernel_size = 2
        self.stride = 2
        self.padding = 0

        # End of todo

    def forward(self, x):
        """Forward propagation of max pooling module.

        Args:
            x: input of shape (B, C, H_in, W_in).
        Returns:
            out: output of shape (B, C, H_out, W_out).
        """

        # TODO Implement forward propogation
        # of maximum pooling module.

        ...
        b, c, h_in, w_in = len(x), len(x[0]), len(x[0][0]), len(x[0][0][0])
        res = tensor.zeros((b, c, h_in / self.kernel_size, w_in / self.kernel_size))
        for i in range(0, b):
            for j in range(0, c):
                for k in range(0, h_in, self.kernel_size):
                    for l in range(0, w_in, self.kernel_size):
                        res = max(x[i][j][k][l] ,x[i][j][k + 1][l] ,x[i][j][k][l + 1] ,x[i][j][k + 1][l + 1])
        return res

        # End of todo

    def backward(self, dy):
        """Backward propagation of max pooling module.

        Args:
            dy: output delta of shape (B, C, H_out, W_out).
        Returns:
            out: input delta of shape (B, C, H_in, W_in).
        """

        # TODO Implement backward propogation
        # of maximum pooling module.

        ...

        # End of todo


class Dropout(Module):

    def __init__(self, p: float=0.5):

        # TODO Initialize the attributes
        # of dropout module.

        ...
        self.p = p
        self.mask = tensor.zeros((1,1))

        # End of todo

    def forward(self, x):

        # TODO Implement forward propogation
        # of dropout module.

        ...
        self.mask = (tensor.tensor(x.shape) < 1-self.p).float()
        return np.dot(self.mask * x) / (1-self.p)
        # End of todo

    def backard(self, dy):

        # TODO Implement backward propogation
        # of dropout module.

        ...
        for i in dy:
            if i<0:
                i=0
        return dy


        # End of todo


if __name__ == '__main__':
    import pdb; pdb.set_trace()

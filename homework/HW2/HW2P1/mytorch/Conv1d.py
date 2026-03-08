# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
from resampling import *


class Conv1d_stride1():
    def __init__(self, in_channels, out_channels, kernel_size,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """

        self.A = A

        batch_size, _, input_size = A.shape
        output_size = (input_size - self.kernel_size) + 1

        Z = np.zeros(shape=(batch_size, self.out_channels, output_size))

        for n in range(batch_size):
            for c in range(self.out_channels):
                for w in range(output_size):
                    Z[n, c, w] = np.sum(A[n, :, w:w+self.kernel_size] * self.W[c, :, :]) + self.b[c]

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """

        batch_size, _, output_size = dLdZ.shape
        input_size = output_size - 1 + self.kernel_size

        for o in range(self.out_channels):
            for i in range(self.in_channels):
                for k in range(self.kernel_size):
                    self.dLdW[o, i, k] = np.sum(self.A[:, i, k:k+output_size] * dLdZ[:, o, :])



        self.dLdb = np.sum(dLdZ, axis=(0, 2))


        flipped_weight = np.flip(self.W, axis=2)
        dLdZ_padding = np.pad(dLdZ, ((0, 0), (0, 0), (self.kernel_size-1, self.kernel_size-1)), "constant")

        dLdA = np.zeros(shape=(batch_size, self.in_channels, input_size))

        for n in range(batch_size):
            for i in range(self.in_channels):
                for s in range(input_size):
                    dLdA[n, i, s] = np.sum(dLdZ_padding[n, :, s:s+self.kernel_size] * flipped_weight[:, i, :])


        return dLdA


class Conv1d():
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names

        self.stride = stride

        # Initialize Conv1d() and Downsample1d() isntance
        self.conv1d_stride1 = Conv1d_stride1(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            weight_init_fn=weight_init_fn, 
            bias_init_fn=bias_init_fn
        )
        self.downsample1d = Downsample1d(downsampling_factor=stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """

        # Call Conv1d_stride1
        Z = self.conv1d_stride1.forward(A)

        # downsample
        Z = self.downsample1d.forward(Z)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        # Call downsample1d backward
        dLdZ = self.downsample1d.backward(dLdZ)

        # Call Conv1d_stride1 backward
        dLdA = self.conv1d_stride1.backward(dLdZ)

        return dLdA

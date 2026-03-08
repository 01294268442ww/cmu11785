import numpy as np
from resampling import *


class Conv2d_stride1():
    def __init__(self, in_channels, out_channels,
                 kernel_size, weight_init_fn=None, bias_init_fn=None):

        # Do not modify this method

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(
                out_channels,
                in_channels,
                kernel_size,
                kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        self.A = A

        batch_size, in_channels, input_height, input_width = A.shape
        output_height = (input_height - self.kernel_size) + 1
        output_width = (input_width - self.kernel_size) + 1
        

        Z = np.zeros(shape=(batch_size, self.out_channels, output_height, output_width))

        for n in range(batch_size):
            for o in range(self.out_channels):
                for h in range(output_height):
                    for w in range(output_width):
                        Z[n, o, h, w] = np.sum(A[n, :, h:h+self.kernel_size, w:w+self.kernel_size] * self.W[o, :, :, :]) + self.b[o]

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        _, _, output_height, output_width = dLdZ.shape

        for o in range(self.out_channels):
            for i in range(self.in_channels):
                for h in range(self.kernel_size):
                    for w in range(self.kernel_size):
                        self.dLdW[o,i, h, w] = np.sum(self.A[:, i, h:h+output_height, w:w+output_width] * dLdZ[:, o, :, :])


        self.dLdb = np.sum(dLdZ, axis=(0, 2, 3))

        dLdZ_pad = np.pad(
            dLdZ,
            ((0, 0), (0, 0), (self.kernel_size-1, self.kernel_size-1), (self.kernel_size-1, self.kernel_size)),
            mode="constant"
        )

        flipped_weight = np.flip(self.W, axis=(2, 3))
        batch_size, _, output_height, output_width = dLdZ.shape
        input_height = output_height - 1 + self.kernel_size
        input_width = output_width - 1 + self.kernel_size

        dLdA = np.zeros(shape=(batch_size, self.in_channels, input_height, input_width))
        for n in range(batch_size):
            for i in range(self.in_channels):
                for h in range(input_height):
                    for w in range(input_width):
                        dLdA[n, i, h, w] = np.sum(dLdZ_pad[n, :, h:h+self.kernel_size, w:w+self.kernel_size] * flipped_weight[:, i, :, :])

        return dLdA


class Conv2d():
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names
        self.stride = stride

        # Initialize Conv2d() and Downsample2d() isntance
        self.conv2d_stride1 = Conv2d_stride1(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            weight_init_fn=weight_init_fn,
            bias_init_fn=bias_init_fn
        )
        self.downsample2d = Downsample2d(downsampling_factor=stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        # Call Conv2d_stride1
        Z = self.conv2d_stride1.forward(A)

        # downsample
        Z = self.downsample2d.forward(Z)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        # Call downsample1d backward
        dLdZ = self.downsample2d.backward(dLdZ)

        # Call Conv1d_stride1 backward
        dLdA = self.conv2d_stride1.backward(dLdZ)

        return dLdA

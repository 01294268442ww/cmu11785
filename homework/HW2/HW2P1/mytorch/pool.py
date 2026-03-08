import numpy as np
from resampling import *


class MaxPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """

        # only the max in kernel have gradient
        
        self.input_shape = A.shape
        batch_size, in_channels, input_width, input_height = A.shape
        
        out_channels = in_channels
        output_width = input_width - self.kernel + 1
        output_height = input_height - self.kernel + 1

        Z = np.zeros(shape=(batch_size, out_channels, output_width, output_height))
        self.max_pos = {}

        for n in range(batch_size):
            for o in range(out_channels):
                for w in range(output_width):
                    for h in range(output_height):
                        window = A[n, o, w:w+self.kernel, h:h+self.kernel]
                        idx = np.argmax(window)
                        dw, dh = np.unravel_index(idx, (self.kernel, self.kernel))
                        Z[n, o, w, h] = window[dw, dh]
                        self.max_pos[(n, o, w, h)] = (w+dw, h+dh)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        
        batch_size, in_channels, input_width, input_height = dLdZ.shape
        dLdA = np.zeros(shape=self.input_shape)

        for (n, o, w, h), (i, j) in self.max_pos.items():
            dLdA[n, o, i, j] += dLdZ[n, o, w, h]
        
        return dLdA


class MeanPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.input_size = A.shape
        batch_size, in_channels, input_width, input_height = A.shape

        out_channels = in_channels
        output_width = input_width - self.kernel + 1
        output_height = input_height - self.kernel + 1

        Z = np.zeros(shape=(batch_size, out_channels, output_width, output_height))
        for n in range(batch_size):
            for o in range(out_channels):
                for w in range(output_width):
                    for h in range(output_height):
                        Z[n, o, w, h] = np.mean(A[n, o, w:w+self.kernel, h:h+self.kernel])
        
        return Z


    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        batch_size, out_channels, output_width, output_height = dLdZ.shape
        dLdA = np.zeros(shape=self.input_size)
        for n in range(batch_size):
            for o in range(out_channels):
                for w in range(output_width):
                    for h in range(output_height):
                        dLdA[n, o, w:w+self.kernel, h:h+self.kernel] += 1 / self.kernel**2 * dLdZ[n, o, w, h]
        
        return dLdA


class MaxPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.maxpool2d_stride1 = MaxPool2d_stride1(kernel=kernel)
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """

        Z = self.maxpool2d_stride1.forward(A)
        Z = self.downsample2d.forward(Z)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdA = self.downsample2d.backward(dLdZ)
        dLdA = self.maxpool2d_stride1.backward(dLdA)

        return dLdA


class MeanPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.meanpool2d_stride1 = MeanPool2d_stride1(kernel=kernel)
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        Z = self.meanpool2d_stride1.forward(A)
        Z = self.downsample2d.forward(Z)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdA = self.downsample2d.backward(dLdZ)
        dLdA = self.meanpool2d_stride1.backward(dLdA)

        return dLdA

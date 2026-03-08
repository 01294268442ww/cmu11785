import numpy as np


class Linear:

    def __init__(self, in_features, out_features, weight_init_fn=None, bias_init_fn=None, debug=False):
        """
        Initialize the weights and biases with zeros
        Checkout np.zeros function.
        Read the writeup to identify the right shapes for all.
        """
        if weight_init_fn is None:
            self.W = np.zeros((out_features, in_features), dtype="f")
        else :
            self.W = weight_init_fn(out_features, in_features)

        if bias_init_fn is None:
            self.b = np.zeros((out_features, 1), dtype="f")
        else:
            self.b = bias_init_fn(out_features)

        self.dLdW = np.zeros((out_features, in_features), dtype="f")
        self.dLdb = np.zeros((out_features, 1), dtype="f")
        self.debug = debug

    def forward(self, A):
        """
        :param A: Input to the linear layer with shape (N, C0)
        :return: Output Z of linear layer with shape (N, C1)
        Read the writeup for implementation details
        """
        self.A = A
        self.N = A.shape[0]  # store the batch size of input
        # Think how will self.Ones helps in the calculations and uncomment below
        self.Ones = np.ones((self.N,1), dtype="f")
        Z = self.A @ self.W.T + self.Ones @ self.b.T

        return Z

    def backward(self, dLdZ):

        dZdA = self.W.T  
        dZdW = self.A
        dZdb = self.Ones
        dZdi = None

        dLdA = dLdZ @ dZdA.T
        dLdW = dLdZ.T @ dZdW
        dLdb = dLdZ.T @ dZdb
        dLdi = None

        self.dLdW = dLdW / self.N
        self.dLdb = dLdb / self.N

        if self.debug:

            self.dZdA = dZdA
            self.dZdW = dZdW
            self.dZdi = dZdi
            self.dZdb = dZdb
            self.dLdA = dLdA
            self.dLdi = dLdi

        return dLdA

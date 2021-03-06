from td.funcs.function import Function_Approximator
import numpy as np

import jax.numpy as jnp
from jax import random
from jax import grad, jit, jacrev, jacfwd, vmap


class MLP(Function_Approximator):

    def __init__(self, features, widths, biases = False, activation='ReLU'):

        self.Phi = features
        self.Phi_trans = jnp.array(np.transpose(self.Phi))

        self.input_dim = features.shape[1]
        self.depth = len(widths)
        self.widths = widths
        self.biases = biases
        self.activation = activation

        assert self.depth >= 1

        self.theta = [(np.random.normal(size=(widths[0], self.input_dim)) / np.sqrt(self.input_dim))]
        for i in range(self.depth - 1):
            self.theta.append( np.random.normal(size=(widths[i+1], widths[i])) / np.sqrt(widths[i]) )
        self.theta.append( np.random.normal(size=(1, widths[-1])) / np.sqrt(widths[-1]) )

        if self.biases:
            for i in range(self.depth):
                self.theta.append(np.zeros((widths[i], 1)))


        # jit all the necessary functions
        self.fast_jac = jit(jacfwd(self._full_forward))
        self.fast_full_evaluate = jit(self._full_forward)

        self.fast_grad = jit(grad(self._forward))
        self.fast_evaluate = jit(self._forward)


    def _forward(self, theta, s):

        x = jnp.dot(theta[0],s)
        for i in range(self.depth):
            if self.biases:
                x += theta[i + self.depth + 1]
            if self.activation == 'ReLU':
                x = jnp.maximum(x, 0.0)
            elif self.activation == 'tanh':
                x = jnp.tanh(x)
            x = jnp.dot(theta[i+1],x)

        return x[0]

    def _full_forward(self, theta):

        x = jnp.dot(theta[0], self.Phi_trans)
        for i in range(self.depth):
            if self.biases:
                x = x + theta[i + self.depth + 1]
            if self.activation == 'ReLU':
                x = jnp.maximum(x, 0.0)
            elif self.activation == 'tanh':
                x = jnp.tanh(x)
            x = jnp.dot(theta[i+1],x)
        return x[0,:]



    def evaluate(self, s):
        return self.fast_evaluate(self.theta, self.Phi[s])

    def full_evaluate(self):
        return self.fast_full_evaluate(self.theta)

    def gradient(self, s):
        return self.fast_grad(self.theta, self.Phi[s])
    
    def jacobian(self):
        return self.fast_jac(self.theta)
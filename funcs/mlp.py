from td.funcs.function import Function_Approximator
import numpy as np

import jax.numpy as jnp
from jax import random
from jax import grad, jit, jacrev, jacfwd, vmap


class MLP(Function_Approximator):

    def __init__(self, features, widths, biases = False):

        self.Phi = features

        self.input_dim = features.shape[1]
        self.depth = len(widths)
        self.widths = widths
        self.biases = biases

        assert self.depth >= 1

        self.theta = [(np.random.normal(size=(widths[0], self.input_dim)) / np.sqrt(self.input_dim))]
        for i in range(self.depth - 1):
            self.theta.append( np.random.normal(size=(widths[i+1], widths[i])) / np.sqrt(widths[i]) )
        self.theta.append( np.random.normal(size=(1, widths[-1])) / np.sqrt(widths[-1]) )

        if self.biases:
            for i in range(self.depth):
                self.theta.append(np.zeros((widths[i], 1)))

    def _forward(self, theta, s):

        x = jnp.dot(theta[0],s)
        for i in range(self.depth):
            if self.biases:
                x += theta[i + self.depth + 1]
            x = jnp.maximum(x, 0.0)
            x = jnp.dot(theta[i+1],x)

        return x[0]


    def _full_forward(self, theta, Phi):

        x = jnp.dot(theta[0], jnp.transpose(Phi))
        for i in range(self.depth):
            if self.biases:
                x = x + theta[i + self.depth + 1]
            x = jnp.maximum(x, 0.0)
            x = jnp.dot(theta[i+1],x)
        return x[0,:]



    def evaluate(self, s):
        return self._forward(self.theta, self.Phi[s])

    def full_evaluate(self):
        return self._full_forward(self.theta, self.Phi)

    def gradient(self, s):
        g = grad(self._forward)
        return g(self.theta, self.Phi[s])
    
    def jacobian(self):
        j = jacfwd(self._full_forward)
        jac = j(self.theta, self.Phi)
        return jac
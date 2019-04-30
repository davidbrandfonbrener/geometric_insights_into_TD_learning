from td.funcs.function import Function_Approximator
import numpy as np

import jax.numpy as jnp
from jax import random
from jax import grad, jit, jacrev, jacfwd, vmap


class TwoLayerNetNoBias(Function_Approximator):

    def __init__(self, features, width):

        self.Phi = features

        input_dim = features.shape[1]

        self.theta1 = (np.random.normal(size=(width, input_dim)) ) / np.sqrt(input_dim) 
        self.theta2 = (np.random.normal(size=(1,width)) ) / np.sqrt(width) 

        self.theta = [self.theta1, self.theta2]

    def _forward(self, theta, s):
        x = jnp.dot(theta[0], s) 
        x = jnp.maximum(x, 0.0)
        x = jnp.dot(theta[1], x)
        return x[0]


    def _full_forward(self, theta, Phi):
        x = jnp.dot(Phi, jnp.transpose(theta[0]))
        x = jnp.maximum(x, 0.0)
        x = jnp.dot(x, jnp.transpose(theta[1]))
        return x[:,0]

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


class TwoLayerNet(Function_Approximator):

    def __init__(self, features, width):

        self.Phi = features

        input_dim = features.shape[1]
        self.theta1 = np.random.random((width, input_dim)) / np.sqrt(width)
        self.theta2 = np.random.random((1,width)) / np.sqrt(width)
        self.b1 = np.zeros((width,1)) 
        self.b2 = np.zeros((1,1))

        self.theta = [self.theta1, self.b1, self.theta2, self.b2]

    def _forward(self, theta, s):
        x = jnp.dot(theta[0], s) + theta[1]
        x = jnp.maximum(x, 0.0)
        #x = jnp.tanh(x)
        x = jnp.dot(theta[2], x) + theta[3]
        return x[0,0]

    def _full_forward(self, theta, Phi):
        f = vmap(self._forward, (None, 0), 0)
        return f(theta, Phi)

    def evaluate(self, s):
        return self._forward(self.theta, self.Phi[s])

    def full_evaluate(self):
        return self._full_forward(self.theta, self.Phi)

    def gradient(self, s):
        g = jit(grad(self._forward))
        return g(self.theta, self.Phi[s])
    
    def jacobian(self):
        j = jacfwd(self._full_forward)
        return j(self.theta, self.Phi)
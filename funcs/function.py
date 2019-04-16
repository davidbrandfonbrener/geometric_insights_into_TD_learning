import numpy as np
from scipy.linalg import expm

import jax.numpy as jnp
from jax import random
from jax import grad, jit, jacrev, jacfwd, vmap

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F


class Function_Approximator(object):
    
    def __init__(self, theta):
        
        self.theta=[theta]

    def evaluate(self, s):
        pass

    def full_evaluate(self):
        pass

    def gradient(self, s):
        pass

    def jacobian(self):
        pass


class Tabular(Function_Approximator):
    
    def evaluate(self, s):
        return self.theta[0][s]

    def full_evaluate(self):
        return self.theta[0]

    def gradient(self, s):
        g = np.zeros_like(self.theta[0])
        g[s] = 1
        return [g]

    def jacobian(self):
        return [np.diag(np.ones_like(self.theta[0]))]


class Linear(Function_Approximator):

    def __init__(self, theta, features):
        
        self.theta = [theta]
        self.Phi = features
    
    def evaluate(self, s):
        return np.dot(self.Phi[s,:], self.theta[0]) 

    def full_evaluate(self):
        return np.dot(self.Phi, self.theta[0]) 

    def gradient(self, s):
        return [self.Phi[s, :]]

    def jacobian(self):
        return [self.Phi]


class TwoLayerNetNoBias(Function_Approximator):

    def __init__(self, features, width):

        self.Phi = features

        input_dim = features.shape[1]

        self.theta1 = np.random.normal(size=(width, input_dim)) / np.sqrt(width)
        self.theta2 = np.random.normal(size=(1,width)) / np.sqrt(width)

        self.theta = [self.theta1, self.theta2]

    def _forward(self, theta, s):
        x = jnp.dot(theta[0], s) 
        x = jnp.maximum(x, 0.0)
        #x = jnp.tanh(x)
        x = jnp.dot(theta[1], x)
        #print(x.shape, x)
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
        return j(self.theta, self.Phi)
    


class Spiral(Function_Approximator):

    def __init__(self, theta):
        
        self.theta = [theta]

        self.epsilon = 0.05
        self.Q = np.array([[1.0 + self.epsilon, 0.5, 1.5],
                            [1.5, 1.0 + self.epsilon, 0.5],
                            [0.5, 1.5, 1.0 + self.epsilon]])

        self.V0 = np.array([10.0, 10.0, -20.0])

    def evaluate(self, s):
        return self.full_evaluate()[s]

    def full_evaluate(self):
        return np.dot(expm(self.theta[0] * self.Q), self.V0)

    def jacobian(self):
        return [np.dot(self.Q, np.dot(expm(self.theta[0] * self.Q), self.V0))]

    def gradient(self, s):
        j = self.jacobian()
        return [j[0][s]]






class Affine(Function_Approximator):

    def __init__(self, theta, features, bias):
        
        self.theta = [theta]
        self.Phi = features
        self.b = bias
    
    def evaluate(self, s):
        return np.dot(self.Phi[s,:], self.theta[0]) + self.b[s]

    def full_evaluate(self):
        return np.dot(self.Phi, self.theta[0]) + self.b

    def gradient(self, s):
        return [self.Phi[s, :]]

    def jacobian(self):
        return [self.Phi]


class TwoLayerNet(Function_Approximator):

    def __init__(self, features, width):

        self.Phi = features

        input_dim = features.shape[1]
        self.theta1 = np.random.random((width, input_dim)) #/ np.sqrt(width)
        self.theta2 = np.random.random((1,width)) #/ np.sqrt(width)
        self.b1 = np.random.random((width,1)) #/ np.sqrt(width)
        self.b2 = 10 * np.random.random((1,1))

        self.theta = [self.theta1, self.b1, self.theta2, self.b2]

    def _forward(self, theta, s):
        x = jnp.dot(theta[0], s) + theta[1]
        x = jnp.maximum(x, 0.0)
        #x = jnp.tanh(x)
        x = jnp.dot(theta[2], x) + theta[3]
        return x[0,0]

    def _full_forward(self, theta):
        f = vmap(self._forward, (None, 0), 0)
        return f(theta, Phi)

    def evaluate(self, s):
        return self._forward(self.theta, s)

    def full_evaluate(self, Phi):
        return self._full_forward(self.theta, Phi)

    def gradient(self, s):
        g = jit(grad(self._forward))
        return g(self.theta, s)
    
    def jacobian(self):
        j = jacfwd(self._full_forward)
        return j(self.theta, self.Phi)
    










class BiasSpiral(Function_Approximator):

    def __init__(self, theta, bias):
        
        self.theta = [theta, bias]

        self.epsilon = 0.05
        self.Q = np.array([[1.0 + self.epsilon, 0.5, 1.5],
                            [1.5, 1.0 + self.epsilon, 0.5],
                            [0.5, 1.5, 1.0 + self.epsilon]])

        self.V0 = np.array([11.0, 10.0, -20.0])

    def evaluate(self, s):
        return self.full_evaluate()[s]

    def full_evaluate(self):
        return np.dot(expm(self.theta[0] * self.Q), self.V0) + 100.0 * np.ones_like(self.V0) * self.theta[1]

    def jacobian(self):
        return [np.dot(self.Q, np.dot(expm(self.theta[0] * self.Q), self.V0)), 100.0 * np.ones_like(self.V0)]

    def gradient(self, s):
        j = self.jacobian()
        return [j[0][s], j[1][s]]


import numpy as np
from scipy.linalg import expm

import jax.numpy as jnp
from jax import random
from jax import grad, jit, jacrev, jacfwd

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
        return [np.diag(np.ones_like(self.theta))]



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





class Spiral(Function_Approximator):

    def __init__(self, theta):
        
        self.theta = [theta]

        self.epsilon = 0.05
        self.Q = np.array([[1.0 + self.epsilon, 0.5, 1.5],
                            [1.5, 1.0 + self.epsilon, 0.5],
                            [0.5, 1.5, 1.0 + self.epsilon]])

        self.V0 = np.array([10.0, 10.0, -20.0])

    def evaluate(self, s):
        return self.evaluate()[s]

    def full_evaluate(self):
        return np.dot(expm(self.theta[0] * self.Q), self.V0)

    def jacobian(self):
        return [np.dot(self.Q, np.dot(expm(self.theta[0] * self.Q), self.V0))]

    def gradient(self, s):
        return [self.jacobian()[s]]




class TwoLayerNet(Function_Approximator):

    def __init__(self, input_dim, width):

        # key = random.PRNGKey(0)

        # self.theta1 = random.normal(key, (width, input_dim)) / np.sqrt(width)
        # self.theta2 = random.normal(key, (1,width)) / np.sqrt(width)
        # self.b1 = random.normal(key, (width, 1)) / np.sqrt(width)
        # self.b2 = random.normal(key, (1,1))

        self.theta1 = np.random.random((width, input_dim)) / np.sqrt(width)
        self.theta2 = np.random.random(width) / np.sqrt(width)
        self.b1 = np.random.random(width) / np.sqrt(width)
        self.b2 = np.random.random()

        self.theta = [self.theta1, self.b1, self.theta2, self.b2]

    def _forward(self, theta, s):
        x = jnp.dot(theta[0], s) + theta[1]
        x = jnp.max(x, 0)
        x = jnp.dot(theta[2], x) + theta[3]
        return x

    def _full_forward(self, theta, Phi):
        x = jnp.dot(theta[0], Phi) + theta[1]
        x = jnp.max(x, 0)
        x = jnp.dot(theta[2], x) + theta[3]
        return x

    def evaluate(self, s):
        return self._forward(self.theta, s)

    def gradient(self, s):
        g = jit(grad(self._forward))
        return g(self.theta, s)
    
    def jacobian(self, Phi):
        j = jit(jacrev(self._full_forward))
        return j(self.theta, Phi)
    








# class Spiral(Function_Approximator):

#     def __init__(self, theta):
        
#         self.theta = theta
#         self.a = np.array([100.0, -70.0, -30.0])
#         self.b = np.array([-23.094, -75.056, 98.15])

#         self.l = 0.866
#         self.epsilon = 0.1

#     def evaluate(self, s):
#         temp = self.a[s] * np.cos(self.l * self.theta) - self.b[s] * np.sin(self.l * self.theta)
#         return temp * np.exp(self.epsilon * self.theta)

#     def evaluate(self):
#         v = np.zeros(3)
#         for s in range(3):
#             temp = self.a[s] * np.cos(self.l * self.theta) - self.b[s] * np.sin(self.l * self.theta)
#             v[s] = temp * np.exp(self.epsilon * self.theta)
#         return v

#     def gradient(self, s):
#         temp1 = self.a[s] * np.cos(self.l * self.theta) - self.b[s] * np.sin(self.l * self.theta)
#         temp2 = -self.a[s] * np.sin(self.l * self.theta) - self.b[s] * np.cos(self.l * self.theta)
            
#         return np.exp(self.epsilon * self.theta) * (self.epsilon*temp1 + self.l*temp2) 

#     def jacobian(self):
#         v = np.zeros(3)
#         for s in range(3):
            
#             temp1 = self.a[s] * np.cos(self.l * self.theta) - self.b[s] * np.sin(self.l * self.theta)
#             temp2 = -self.a[s] * np.sin(self.l * self.theta) - self.b[s] * np.cos(self.l * self.theta)
            
#             v[s] = np.exp(self.epsilon * self.theta) * (self.epsilon*temp1 + self.l*temp2) 
        
#         return v

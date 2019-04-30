from td.funcs.function import Function_Approximator
import numpy as np

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
from td.funcs.function import Function_Approximator
import numpy as np
from scipy.linalg import expm

class Spiral(Function_Approximator):

    def __init__(self, theta, P, V0, epsilon):
        
        self.theta = [theta]

        self.epsilon = epsilon
        self.Q = np.ones_like(P) + P - np.transpose(P) + epsilon * np.diag(np.ones_like(V0))

        self.V0 = V0

    def evaluate(self, s):
        return self.full_evaluate()[s]

    def full_evaluate(self):
        return np.dot(expm(self.theta[0] * self.Q), self.V0)

    def jacobian(self):
        return [np.dot(self.Q, np.dot(expm(self.theta[0] * self.Q), self.V0))]

    def gradient(self, s):
        j = self.jacobian()
        return [j[0][s]]

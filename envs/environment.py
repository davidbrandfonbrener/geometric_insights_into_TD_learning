import dm_control
import gym
import numpy as np

class Environment(object):
    
    def __init__(self, gamma = 1.0, P = None, mu = None, R_mat = None):
        
        self.gamma = gamma

        self.P = P 
        self.mu = mu 
        self.R_mat = R_mat

        self.state = None

    def step(self):
        pass

    def reset(self):
        pass

    def compute_V_star(self):
        pass
        


class MRP(Environment):
    
    def step(self):
        s = self.state
        self.state = np.random.choice(self.P.shape[1], p = self.P[s,:])
        return self.state, self.R_mat[s, self.state]

    def reset(self):
        self.state = np.random.choice(self.mu.size, p = self.mu)
        return self.state

    def compute_V_star(self):
        self.R = np.sum(self.R_mat * self.P, axis=1)
        self.V_star = np.dot(np.linalg.inv(np.diag(np.ones_like(self.R)) - self.gamma * self.P), self.R)
        return self.V_star
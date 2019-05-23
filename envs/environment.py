import numpy as np

class Environment(object):

    def step(self):
        pass

    def reset(self):
        pass

        


class MRP(Environment):

    def __init__(self, gamma = 1.0, P = None, R_mat = None, V_star = None):
        
        self.gamma = gamma

        self.P = P 

        self.state = None

        vals, vecs = np.linalg.eig(np.transpose(P))
        self.mu = np.array(vecs[:, np.argmax(vals)], dtype=float)
        self.mu = self.mu / np.sum(self.mu)
        assert min(self.mu) > 0
        assert np.sum(self.mu) - 1 < 1e-8

        self.A = np.dot(np.diag(self.mu), np.diag(np.ones_like(self.mu)) - self.gamma * self.P)
        
        # determine reward and optimal value function
        if R_mat is not None:
            self.R_mat = R_mat
            self.R = np.sum(self.R_mat * self.P, axis=1)
            self.V_star = np.dot(np.linalg.inv( np.diag(np.ones_like(self.mu)) - self.gamma * self.P), self.R)
        else:
            self.V_star = V_star
            self.R = np.dot(np.diag(np.ones_like(self.mu)) - self.gamma * self.P, self.V_star)
            self.R_mat = np.zeros_like(P)
            for i in range(self.R_mat.shape[0]):
                for j in range(self.R_mat.shape[1]):
                    if self.P[i,j] != 0:
                        self.R_mat[i,j] = self.R[i] / self.P[i,j]
                        break

        assert np.linalg.norm(self.V_star - self.R - self.gamma * np.dot(self.P, self.V_star)) < 1e-8
        assert np.sum(abs(self.R - np.sum(self.R_mat * self.P, axis=1))) < 1e-8
    
    def step(self):
        s = self.state
        self.state = np.random.choice(self.P.shape[1], p = self.P[s,:])
        
        return self.state, self.R_mat[s, self.state]

    def reset(self):
        self.state = np.random.choice(self.mu.size, p = self.mu)
        return self.state




# class Gym(Environment):

#     def __init__(self, task, policy = None):

#         self.gym_env = gym.make(task)
#         self.policy = policy
#         self.state = None

#         self.obs_shape = self.gym_env.observation_space.shape
#         self.obs_count = self.gym_env.observation_space.high - self.gym_env.observation_space.low + 1


#     def obs_to_state(self, obs):
        
#         return disc_state


#     def state_to_obs(self, state):

#         return obs


#     def step(self):
#         if self.policy is None:
#             action = self.gym_env.action_space.sample()
#         else:
#             action = self.policy(self.state)

#         obs, r, done, info = self.gym_env.step(action)
#         self.state = self.obs_to_state(obs)

#         return self.state, r, done

#     def reset(self):
#         self.state = self.obs_to_state(self.gym_env.reset())
#         return self.state



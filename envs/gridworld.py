from td.envs.environment import Environment
import numpy as np
from matplotlib import pyplot as plt

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
NULL = 4
ACTIONS = [UP, DOWN, LEFT, RIGHT, NULL]

class Grid(object):
    def __init__(self, x_dim, y_dim, wall_locs, reward_locs):

        self.x_dim = x_dim
        self.y_dim = y_dim

        self.grid = np.stack([wall_locs, reward_locs], axis = -1)

        self.reset()

    def get_pos(self):
        return self.pos

    def get_grid(self):
        return self.grid
    
    def get_state(self):
        return self.pos_to_state(self.pos)

    def pos_to_state(self,loc):
        return self.y_dim * loc[0] + loc[1]
    
    def state_to_pos(self,state):
        return [state // self.y_dim, state % self.x_dim]
    
    def render(self):
        im = self.grid[:, :, 1] - self.grid[:, :, 0]
        im[self.pos[0], self.pos[1]] = -2
        plt.imshow(im)

    def reset(self):
        #self.orientation = np.random.choice([UP, DOWN, LEFT, RIGHT])
        locs = []
        for x in range(self.x_dim):
            for y in range(self.y_dim):
                if self.grid[x,y,0] == 0:
                    locs.append(self.pos_to_state([x,y]))
        self.pos = self.state_to_pos(np.random.choice(locs))

    def make_step(self, policy=None):

        if policy is None:
            action = np.random.choice(ACTIONS)
        else:
            action = policy(self.pos)
        
        if action == UP:
            if self.pos[1] + 1 < self.y_dim:
                if self.grid[self.pos[0], self.pos[1] + 1, 0] == 0:
                    self.pos[1] += 1
        elif action == DOWN:
            if self.pos[1] - 1 >= 0:
                if self.grid[self.pos[0], self.pos[1] - 1, 0] == 0:
                    self.pos[1] -= 1
        elif action == RIGHT:
            if self.pos[0] + 1 < self.x_dim:
                if self.grid[self.pos[0]+1, self.pos[1], 0] == 0:
                    self.pos[0] += 1
        elif action == LEFT:
            if self.pos[0] - 1 >= 0:
                if self.grid[self.pos[0] - 1, self.pos[1], 0] == 0:
                    self.pos[0] -= 1
        
        reward = self.grid[self.pos[0], self.pos[1], 1]

        return reward

class Gridworld(Environment):
    def __init__(self, grid, gamma = 1.0, policy = None):
        
        self.gamma = gamma
        self.grid = grid

        self.policy = policy

        self.state = self.grid.get_state()

    def step(self):
        r = self.grid.make_step(self.policy)
        self.state = self.grid.get_state()
        return r, self.state

    def reset(self):
        self.grid.reset()
        self.state = self.grid.get_state()
        return self.state
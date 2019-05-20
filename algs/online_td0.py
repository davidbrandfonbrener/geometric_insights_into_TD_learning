import numpy as np 
from copy import deepcopy
from td.utils import utils

def step(V, env, alpha):

    s = env.state
    s_prime, r = env.step()

    delta = r + env.gamma * V.evaluate(s_prime) - V.evaluate(s)

    grads = V.gradient(s)
    
    for i in range(len(V.theta)):
        V.theta[i] = V.theta[i] + alpha * grads[i] * delta

    return deepcopy(V.theta), r, s


def TD0(V, env, alpha, steps, log_idx, plot_step):

    Vs, thetas, rewards, states = [], [],[],[]

    env.reset()

    for i in range(steps):

        theta, r, s = step(V, env, alpha)
        
        thetas.append(theta)
        rewards.append(r)
        states.append(s)

        if i % plot_step == 0:
            Vs.append(V.full_evaluate())

        if i % log_idx == 0:
            print("step: ", i, ", error: ", utils.dist_mu(env.mu, env.V_star, np.expand_dims(Vs[-1], 0)))


    return Vs, thetas, rewards, states
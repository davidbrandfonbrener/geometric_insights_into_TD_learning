import numpy as np 
from copy import deepcopy

def step(V, env, alpha):

    s = env.state
    s_prime, r = env.step()

    delta = r + env.gamma * V.evaluate(s_prime) - V.evaluate(s)

    grads = V.gradient(s)
    
    for i in range(len(V.theta)):
        V.theta[i] = V.theta[i] + alpha * grads[i] * delta

    return deepcopy(V.theta), r, s


def TD0(V, env, alpha, steps):

    Vs, thetas, rewards, states = [], [],[],[]

    env.reset()

    for i in range(steps):

        theta, r, s = step(V, env, alpha)
        
        thetas.append(theta)
        rewards.append(r)
        states.append(s)

        Vs.append(V.full_evaluate())


    return Vs, thetas, rewards, states
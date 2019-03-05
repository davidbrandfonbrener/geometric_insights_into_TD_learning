import numpy as np 


def step(V, env, alpha):

    s = env.state
    s_prime, r = env.step()

    delta = r + env.gamma * V.evaluate(s_prime) - V.evaluate(s)

    grads = V.gradient(s)

    for i in range(len(V.theta)):
        V.theta[i]  = V.theta[i] + alpha * grads[i] * delta

    return V.theta, r, s


def TD0(V, env, alpha, steps):

    thetas, rewards, states = [],[],[]

    env.reset()

    for i in range(steps):

        theta, r, s = step(V, env, alpha)
        
        thetas.append(theta)
        rewards.append(r)
        states.append(s)


    return thetas, rewards, states
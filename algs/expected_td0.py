import numpy as np 
from copy import deepcopy
from td.utils import utils


def step(V, env, alpha):

    jac = V.jacobian()
    for i in range(len(V.theta)):
        V.theta[i] = V.theta[i] - alpha * np.dot(np.moveaxis(jac[i], 0, -1), np.dot(env.A, V.full_evaluate() - env.V_star))
            
    return deepcopy(V.theta)


def TD0(V, env, alpha, steps, log_idx):

    thetas, Vs = [], []

    for i in range(steps):

        theta = step(V, env, alpha)
        
        thetas.append(theta)

        Vs.append(V.full_evaluate())

        if i % log_idx == 0:
            print("step: ", i, ", error: ", utils.dist_mu(env, np.expand_dims(Vs[-1], 0)))

    return thetas, Vs
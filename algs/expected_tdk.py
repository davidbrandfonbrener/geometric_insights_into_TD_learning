import numpy as np 
from copy import deepcopy
from td.utils import utils


def step(V, Ak, V_star, alpha):

    jac = V.jacobian()
    for i in range(len(V.theta)):
        V.theta[i] = V.theta[i] - alpha * np.dot(np.moveaxis(jac[i], 0, -1), np.dot(Ak, V.full_evaluate() - V_star))
            
    return deepcopy(V.theta)


def TDk(k, V, env, alpha, steps, log_idx):

    thetas, Vs = [], []

    V_star = env.V_star
    Ak = np.dot(np.diag(env.mu), np.diag(np.ones_like(env.mu)) - env.gamma * np.linalg.matrix_power(env.P, k))

    for i in range(steps):

        theta = step(V, Ak, V_star, alpha)
        
        thetas.append(theta)

        Vs.append(V.full_evaluate())

        if i % log_idx == 0:
            print("step: ", i, ", error: ", utils.dist_mu(env, np.expand_dims(Vs[-1], 0)))

    return thetas, Vs
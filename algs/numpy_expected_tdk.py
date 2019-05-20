import numpy as np 
from copy import deepcopy
from td.utils import utils


def TDk(k, V, env, alpha, steps, log_idx, plot_step):

    thetas, Vs = [], []

    V_star = env.V_star
    Pk = env.P
    for i in range(k-1):
        Pk = np.dot(env.P, Pk)

    Ak = np.dot(np.diag(env.mu), np.diag(np.ones_like(env.mu)) - env.gamma * Pk)
    Ak = np.array(Ak)

    for i in range(steps):

        theta = step(V, Ak, V_star, alpha)
        
        thetas.append(theta)

        if i % plot_step == 0:
            Vs.append(V.full_evaluate())

        if i % log_idx == 0:
            print("step: ", i, ", error: ", utils.dist_mu(env.mu, env.V_star, np.expand_dims(Vs[-1], 0)))

    return thetas, Vs


def step(V, Ak, V_star, alpha):

    jac = V.jacobian()
    for i in range(len(V.theta)):
        J = np.array(np.moveaxis(jac[i], 0, -1))
        V.theta[i] = V.theta[i] - alpha * np.dot(J, np.dot(Ak, V.full_evaluate() - V_star))
        V_eval = V.full_evaluate()
        V.theta[i] -= alpha * np.dot(J, np.dot(Ak, V_eval - V_star))
            
    return deepcopy(V.theta)
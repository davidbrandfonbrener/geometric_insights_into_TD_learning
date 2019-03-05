import numpy as np 


def step(V, env, alpha):

    A = np.dot(np.diag(env.mu), np.diag(np.ones_like(env.mu)) - env.gamma * env.P)

    jac = V.jacobian()

    for i in range(len(V.theta)):
        V.theta[i] = V.theta[i] - alpha * np.dot(np.transpose(jac[i]), np.dot(A, V.full_evaluate() - env.V_star))

    return V.theta


def TD0(V, env, alpha, steps):

    thetas, Vs = [], []

    for i in range(steps):

        theta = step(V, env, alpha)
        
        thetas.append(theta)
        Vs.append(V.full_evaluate())


    return thetas, Vs
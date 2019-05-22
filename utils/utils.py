import numpy as np
import jax.numpy as jnp
from jax import jit

def jac_cond(jac):
    
    flat_jac = [np.reshape(a, (a.shape[0], -1)) for a in jac]
    vals = np.linalg.svd(np.concatenate(flat_jac, axis = 1))[1]

    return max(vals) / min(vals)

def dist_S(env, Vs):
    S = 0.5 * (env.A + np.transpose(env.A))
    out = np.zeros(Vs.shape[0])
    for i in range(Vs.shape[0]):
        v = Vs[i] - env.V_star
        out[i] = np.dot(v, np.dot(S, v))
    return out

def dist_mu(mu, V_star, Vs):
    out = []
    for i in range(Vs.shape[0]):
        v = Vs[i] - V_star
        out.append(jnp.dot(v, mu * v))
    return np.array(out)

def mu_norm(mu, Vs):
    out = []
    for i in range(Vs.shape[0]):
        v = Vs[i] 
        out.append(jnp.dot(v, mu * v))
    return np.array(out)

def theta_norm(thetas):
    out = np.ones(len(thetas))
    for j, t in enumerate(thetas):
        for i in range(len(t)):
            out[j] = out[j] * np.linalg.norm(t[i])
    
    return out


def overparam_cond_number_bound(P, mu, gamma, k):

    A = np.dot(np.diag(mu), np.diag(np.ones_like(mu)) - gamma * np.linalg.matrix_power(P, k))

    S = 0.5 * ( A + np.transpose(A))
    R = 0.5 * (A  - np.transpose(A))

    B = np.dot(np.linalg.inv(np.dot(S,S) + np.dot(np.transpose(A),A)), np.dot(np.transpose(R), R))


    b = 1 / max(np.linalg.eig(B)[0])

    return np.sqrt(b)


def overparam_mu_cond_number_bound(P, mu, gamma):

    Dmu = np.diag(mu)

    A = np.dot(Dmu, np.diag(np.ones_like(mu)) - gamma * P)
    B = np.dot(Dmu, np.diag(np.ones_like(mu)) + gamma * P)

    Cm = 0.25 * np.dot(np.linalg.inv(np.dot(Dmu,Dmu) + np.dot(np.transpose(A),A)), np.dot(np.transpose(B), B))

    d = 1 / max(np.linalg.eig(Cm)[0])

    return np.sqrt(d)


def compute_homogeneous_bound(P, mu, V_star, gamma):

    R = np.dot(np.diag(np.ones_like(mu)) - gamma * P, V_star)

    return np.sqrt(np.dot(R, np.dot(np.diag(mu), R))) / (1.0 - gamma)


def dynamics_norm(V, Ak, V_star):
    norm = 0
    jac = V.jacobian()
    for i in range(len(V.theta)):
        JT = np.array(np.moveaxis(jac[i], 0, -1))
        theta_dot = np.dot(JT, np.dot(Ak, V.full_evaluate() - V_star))
        norm += np.sum(np.square(theta_dot))
    return np.sqrt(norm)



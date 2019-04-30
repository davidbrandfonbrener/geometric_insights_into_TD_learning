import numpy as np
import jax.numpy as jnp

def jac_svd_max_min(jac):
    
    flat_jac = [np.reshape(a, (a.shape[0], -1)) for a in jac]
    vals = np.linalg.svd(np.concatenate(flat_jac, axis = 1))[1]

    return max(vals), min(vals)

def dist_S(env, Vs):
    S = 0.5 * (env.A + np.transpose(env.A))
    out = np.zeros(Vs.shape[0])
    for i in range(Vs.shape[0]):
        v = Vs[i] - env.V_star
        out[i] = np.dot(v, np.dot(S, v))
    return out


def dist_mu(env, Vs):
    out = np.zeros(Vs.shape[0])
    for i in range(Vs.shape[0]):
        v = Vs[i] - env.V_star
        out[i] = np.dot(v, env.mu * v)
    return out

def theta_norm(thetas):
    out = np.ones(len(thetas))
    for j, t in enumerate(thetas):
        for i in range(len(t)):
            out[j] = out[j] * np.linalg.norm(t[i])
    
    return out

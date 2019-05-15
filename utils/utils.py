import numpy as np
import jax.numpy as jnp

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


def overparam_cond_number_bound(P, mu, gamma, k):

    A = np.dot(np.diag(mu), np.diag(np.ones_like(mu)) - gamma * np.linalg.matrix_power(P, k))

    S = 0.5 * ( A + np.transpose(A))
    R = 0.5 * (A  - np.transpose(A))

    B = np.dot(np.linalg.inv(np.dot(S,S)), np.dot(np.transpose(R), R))
    Ba = np.dot(np.linalg.inv(np.dot(np.transpose(A),A)), np.dot(np.transpose(R), R))

    b = 1 / max(np.linalg.eig(B)[0])
    ba = 1 / max(np.linalg.eig(Ba)[0])

    return np.sqrt(b + ba)


def overparam_mu_cond_number_bound(P, mu, gamma):

    Dmu = np.diag(mu)

    A = np.dot(Dmu, np.diag(np.ones_like(mu)) - gamma * P)
    B = np.dot(Dmu, np.diag(np.ones_like(mu)) + gamma * P)

    Cm = 0.25 * np.dot(np.linalg.inv(np.dot(Dmu,Dmu)), np.dot(np.transpose(A), A))
    CB = np.dot(np.linalg.inv(np.dot(np.transpose(A),A)), np.dot(np.transpose(A), A))

    d = 1 / max(np.linalg.eig(Cm)[0])
    db = 1 / max(np.linalg.eig(CB)[0])

    return np.sqrt(d + db)


def compute_homogeneous_bound(P, mu, V_star, gamma):

    R = np.dot(np.diag(np.ones_like(mu)) - gamma * P, V_star)

    return np.sqrt(np.dot(R, np.dot(np.diag(mu), R))) / (1.0 - gamma)




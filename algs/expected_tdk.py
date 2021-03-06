import numpy as np 
import jax.numpy as jnp 
from copy import deepcopy
from td.utils import utils
from jax import jit


def TDk(k, V, env, alpha, steps, log_idx, plot_step):

    thetas, Vs = [], []

    V_star = env.V_star
    Pk = env.P
    for i in range(k-1):
        Pk = np.dot(env.P, Pk)

    Ak = np.dot(np.diag(env.mu), np.diag(np.ones_like(env.mu)) - env.gamma * Pk)
    Ak = jnp.array(Ak)

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
        J = jnp.array(np.moveaxis(jac[i], 0, -1))
        # V.theta[i] = V.theta[i] - alpha * jnp.dot(jnp.moveaxis(J, 0, -1), jnp.dot(Ak, V.full_evaluate() - V_star))
        # J = jnp.moveaxis(J, 0, -1)
        V_eval = V.full_evaluate()
        V.theta[i] -= quick_step(alpha, J, Ak, V_eval, V_star)
            
    return deepcopy(V.theta)

@jit
def quick_step(alpha, J, A, V, V_star):
    return alpha * jnp.dot(J, jnp.dot(A, V- V_star))
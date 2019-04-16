from td.funcs import function
from td.envs import environment
from td.algs import expected_td0, online_td0

import numpy as np
from matplotlib import pyplot as plt


def reversing_loop(n, k, delta):
    # initialize a deterministic cycle
    P = np.zeros((n,n))
    for i in range(n):
        P[i, i-1] = 1

    # add reverse connections with prob delta to the first k states
    for i in range(k):
        P[i, i-1] = 1 - delta
        P[i, (i+1) % n] = delta

    # find mu
    vals, vecs = np.linalg.eig(P)
    mu= np.zeros(n)
    for i in range(n):
        print(vals[i])
        if np.abs(vals[i] - 1) < 1e-6:
            mu = vecs[:,i]
            break
    assert np.sum(mu) != 0
    mu = np.array( mu / np.sum(mu), dtype=float)
    print(mu)

    return P, mu


def dist_S(env, Vs):
    S = 0.5 * (env.A + np.transpose(env.A))
    out = np.zeros(Vs.shape[0])
    for i in range(Vs.shape[0]):
        v = Vs[i] - env.V_star
        out[i] = np.dot(v, np.dot(S, v))
    return out


def dist_mu(env, Vs):
    D = np.diag(env.mu)
    out = np.zeros(Vs.shape[0])
    for i in range(Vs.shape[0]):
        v = Vs[i] - env.V_star
        out[i] = np.dot(v, np.dot(D, v))
    return out



n = 30
k = 10
delta = 0.5
features = 2
width = 5

offline = True

# integration
steps = 10000
stepsize = 0.01
#plotting
plot_start = 1000
plot_step = 100

P, mu = reversing_loop(n, k, delta)
gamma = 0.9

R_mat = np.zeros_like(P)
R_mat[-1, -2] = 1

#Phi = np.concatenate([np.random.rand(n,features) - 0.5, np.ones((n,1))], axis = 1)
#Phi = np.concatenate([-np.expand_dims(np.arange(0,n), 1)/(1.0*n) + 0.5, np.expand_dims(np.arange(0,n), 1)/(1.0*n) - 0.5, np.ones((n,1))], axis = 1)  
angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
Phi = np.concatenate([np.expand_dims(np.sin(angles), 1), np.expand_dims(np.cos(angles), 1), np.ones((n,1))], axis = 1) 
print(Phi)


env = environment.MRP(gamma, P, mu, R_mat)
print(env.V_star)

np.random.seed(1)



if offline: 

    V = function.Tabular(np.zeros(n))
    thetas, Vs = expected_td0.TD0(V, env, stepsize, steps)

    Vs = np.array(Vs)
    plt.plot(dist_mu(env, Vs[plot_start::plot_step]), color = 'b')


    V = function.Linear(np.zeros(features + 1), Phi)
    thetas, Vs = expected_td0.TD0(V, env, stepsize, steps)
    print(Vs[-1])
    print(thetas[-1])

    Vs = np.array(Vs)
    plt.plot(dist_mu(env, Vs[plot_start::plot_step]), color = 'r')


    V = function.TwoLayerNetNoBias(Phi, width)
    thetas, Vs = expected_td0.TD0(V, env, stepsize, steps)
    print(Vs[-1])
    print(np.dot(thetas[-1][1], thetas[-1][0]))

    Vs = np.array(Vs)
    plt.plot(dist_mu(env, Vs[plot_start::plot_step]), color = 'orange')

    plt.show()

else:

    V = function.Tabular(np.zeros(n))
    Vs, thetas, rs, ss = online_td0.TD0(V, env, stepsize, steps)

    Vs = np.array(Vs)
    plt.plot(dist_mu(env, Vs[plot_start::plot_step]), color = 'b')


    Phi = np.concatenate([np.random.rand(n,features) - 0.5, np.ones((n,1))], axis = 1) 
    print(Phi)

    V = function.Linear(np.zeros(features + 1), Phi)
    Vs, thetas, rs, ss = online_td0.TD0(V, env, stepsize, steps)
    print(Vs[-1])
    print(thetas[-1])

    Vs = np.array(Vs)
    plt.plot(dist_mu(env, Vs[plot_start::plot_step]), color = 'r')


    V = function.TwoLayerNetNoBias(Phi, width)
    Vs, thetas, rs, ss = online_td0.TD0(V, env, stepsize, steps)
    print(Vs[-1])
    print(np.dot(thetas[-1][1], thetas[-1][0]))


    Vs = np.array(Vs)
    plt.plot(dist_mu(env, Vs[plot_start::plot_step]), color = 'orange')

    plt.show()


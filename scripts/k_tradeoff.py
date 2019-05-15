from td.funcs import function
from td.envs import environment
from td.algs import expected_td0, expected_tdk
from td.funcs import mlp, simple
from td.utils import utils

import numpy as np
from matplotlib import pyplot as plt

import argparse

import jax.numpy as jnp

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', default=10, type=int)
    parser.add_argument('--delta', default=0.5)
    parser.add_argument('--width', default=10, type=int)
    parser.add_argument('--depth', default = 2, type=int)
    parser.add_argument('--log_idx', default=1000, type=int)
    parser.add_argument('--steps', default=4000, type=int)
    parser.add_argument('--stepsize', default=0.1)
    parser.add_argument('--plot_start', default=0, type=int) 
    parser.add_argument('--plot_step', default=100, type=int)
    parser.add_argument('--gamma', default=0.9)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--save_path', default="../outputs/data/k_tradeoff/")

    args = parser.parse_args()
    return args


def build_P(n, delta):
    # initialize a sticky cycle
    P = np.zeros((n,n))
    # add symmetric connections with prob delta
    for i in range(n):
        P[i, i-1] = 0.5 - delta
        P[i, (i+1) % n] = delta
        P[i,i] = 0.5

    return P



def run_experiment(args):

    np.random.seed(args.seed)

    P = build_P(args.n, args.delta)

    # R_mat = np.zeros_like(P)
    # R_mat[1, 0] = 1
    # env = environment.MRP(args.gamma, P, R_mat)

    V_star = np.zeros(args.n) 
    for i in range(0, args.n, 2):
        V_star[i] = 1
    env = environment.MRP(args.gamma, P, V_star=V_star)

    bound = utils.overparam_cond_number_bound(env.P, env.mu, env.gamma, args.k)
    
    angles = np.linspace(0, 2 * np.pi, args.n, endpoint=False)
    Phi = np.concatenate([np.expand_dims(np.sin(angles), 1), np.expand_dims(np.cos(angles), 1), np.ones((args.n,1))], axis = 1) 


    V = simple.Tabular(np.zeros(args.n))
    thetas, Vs = expected_tdk.TDk(args.k, V, env, args.stepsize, args.steps, args.log_idx)
    tabular_Vs = utils.dist_mu(env, np.array(Vs)[args.plot_start::args.plot_step])


    V = mlp.MLP(Phi, [args.width]*args.depth, biases = False)
    thetas, Vs = expected_tdk.TDk(args.k, V, env, args.stepsize, args.steps, args.log_idx)
    mlp_Vs = utils.dist_mu(env, np.array(Vs)[args.plot_start::args.plot_step])

    ts = thetas[args.plot_start::args.plot_step]
    condition_numbers = []
    for i in range(len(ts)):
        V.theta = ts[i]
        condition_numbers.append(utils.jac_cond(V.jacobian()))
    

    V = simple.Linear(np.zeros(Phi.shape[1]), Phi)
    thetas, Vs = expected_tdk.TDk(args.k, V, env, args.stepsize, args.steps, args.log_idx)
    linear_Vs = utils.dist_mu(env, np.array(Vs)[args.plot_start::args.plot_step])

    smoothness = max([abs(env.V_star[i] - env.V_star[i-1]) for i in range(args.n)])

    return tabular_Vs, linear_Vs, mlp_Vs, condition_numbers, bound, smoothness


    




def main():
    args = parse_args()

    ks = [1, 2, 3, 5]
    
    t, l, m, c, b, s = [], [], [], [], [], []
    for k in ks:

        args.k = k

        tabular_Vs, linear_Vs, mlp_Vs, condition_numbers, bound, smoothness = run_experiment(args)
        t.append(tabular_Vs)
        l.append(linear_Vs)
        m.append(mlp_Vs)
        c.append(condition_numbers)
        b.append(bound)
        s.append(smoothness)
    
    t = np.array(t)
    l = np.array(l)
    m = np.array(m)
    c = np.array(c)
    b = np.array(b)
    s = np.array(s)

    np.savez(args.save_path  + "n_" + str(args.n) + "_mlp_depth_" + str(args.depth) 
            + "_width_" + str(args.width) + "_seed_" + str(args.seed) + ".npz", 
            tabular_Vs = t, linear_Vs = l, mlp_Vs = m, condition_numbers = c, bound = b, ks = ks, smoothness= s)


if __name__ == "__main__":
    main()
from td.funcs import function
from td.envs import environment, P_matrices
from td.algs import expected_td0, expected_tdk, online_td0
from td.funcs import mlp, simple
from td.utils import utils

import numpy as np
from matplotlib import pyplot as plt

import argparse

import jax.numpy as jnp

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', default=50, type=int)
    parser.add_argument('--k', default=1, type=int)
    parser.add_argument('--delta', default=0.1, type=float)
    parser.add_argument('--width', default=6, type=int)
    parser.add_argument('--depth', default = 1, type=int)
    parser.add_argument('--log_idx', default=1000, type=int)
    parser.add_argument('--steps', default=10000, type=int)
    parser.add_argument('--stepsize', default=0.1, type=float)
    parser.add_argument('--plot_start', default=0, type=int)
    parser.add_argument('--plot_step', default=100, type=int)
    parser.add_argument('--gamma', default=0.9)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--online', default=False, type=bool)
    parser.add_argument('--save_path', default="../outputs/data/nn_init_hard_V/")

    args = parser.parse_args()
    return args

def run_experiment(args):

    np.random.seed(args.seed)

    P = P_matrices.build_cyclic_P(args.n, args.delta)

    # R_mat = np.zeros_like(P)
    # R_mat[0, 1] = 1
    # env = environment.MRP(args.gamma, P, R_mat)

    V_star = np.zeros(args.n)
    for i in range(0, args.n, 2):
        V_star[i] = 1
    env = environment.MRP(args.gamma, P, V_star=V_star)
    

    # build features
    angles = np.linspace(0, 2 * np.pi, args.n, endpoint=False)
    Phi = np.concatenate([np.expand_dims(np.sin(angles), 1), np.expand_dims(np.cos(angles), 1), np.ones((args.n,1))], axis = 1) 


    V = simple.Tabular(np.zeros(args.n))
    if args.online:
        Vs, thetas, _, _ = online_td0.TD0(V, env, args.stepsize, args.steps, args.log_idx, args.plot_step)
    else:
        thetas, Vs = expected_tdk.TDk(args.k, V, env, args.stepsize, args.steps, args.log_idx, args.plot_step)
    tabular_Vs = utils.dist_mu(env.mu, env.V_star, jnp.array(Vs))


    V = mlp.MLP(Phi, [args.width]*args.depth, biases = False)
    if args.online:
        Vs, thetas, _, _ = online_td0.TD0(V, env, args.stepsize, args.steps, args.log_idx, args.plot_step)
    else:
        thetas, Vs = expected_tdk.TDk(args.k, V, env, args.stepsize, args.steps, args.log_idx, args.plot_step)
    mlp_V_to_V_star = utils.dist_mu(env.mu, env.V_star, jnp.array(Vs))
    mlp_V_to_origin = utils.mu_norm(env.mu, jnp.array(Vs))

    ts = thetas[args.plot_start::args.plot_step]
    condition_numbers = []
    for i in range(len(ts)):
        V.theta = ts[i]
        condition_numbers.append(utils.jac_cond(V.jacobian()))
    
    bound = utils.overparam_cond_number_bound(env.P, env.mu, env.gamma, args.k)
    smoothness = max([abs(env.V_star[i] - env.V_star[i-1]) for i in range(args.n)])

    return tabular_Vs, mlp_V_to_V_star, mlp_V_to_origin, condition_numbers, bound, smoothness


    




def main():
    args = parse_args()
    print(args.online)

    seeds = range(1,11)
    
    t, Vs, Vo, c, b, s = [], [], [], [], [], []
    for seed in seeds:

        args.seed = seed
        tabular_Vs, mlp_Vs, mlp_Vo, condition_numbers, bound, smoothness = run_experiment(args)
        t.append(tabular_Vs)
        Vs.append(mlp_Vs)
        Vo.append(mlp_Vo)
        c.append(condition_numbers)
        b.append(bound)
        s.append(smoothness)
    
    t = np.array(t)
    Vs = np.array(Vs)
    Vo = np.array(Vo)
    c = np.array(c)
    b = np.array(b)
    s = np.array(s)

    np.savez(args.save_path  + "k_" + str(args.k) + "_n_" + str(args.n) + "_mlp_depth_" + str(args.depth) 
            + "_width_" + str(args.width) + "_online_" + str(args.online) + ".npz", 
            tabular_Vs = t, mlp_Vs = Vs, mlp_Vo = Vo, condition_numbers = c, bound = b, seeds= np.array(seeds), smoothness= s)


if __name__ == "__main__":
    main()
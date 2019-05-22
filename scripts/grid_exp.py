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
    parser.add_argument('--n', default=25, type=int)
    parser.add_argument('--k', default=1, type=int)
    parser.add_argument('--delta', default=0.1, type=float)
    parser.add_argument('--width', default=20, type=int)
    parser.add_argument('--depth', default = 1, type=int)
    parser.add_argument('--log_idx', default=1000, type=int)
    parser.add_argument('--steps', default=10000, type=int)
    parser.add_argument('--stepsize', default=0.1, type=float)
    parser.add_argument('--plot_start', default=0, type=int)
    parser.add_argument('--plot_step', default=100, type=int)
    parser.add_argument('--gamma', default=0.9)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--online', default=False, type=bool)
    parser.add_argument('--save_path', default="../outputs/data/grid_exp/")

    args = parser.parse_args()
    return args

def run_experiment(args):

    grid_n = int(np.sqrt(args.n))

    P = P_matrices.constant_gridworld(grid_n, 0.3, 0.1, 0.1, 0.3, 0.2)

    R_mat = np.zeros_like(P)
    R_mat[:, -1] = 1
    env = environment.MRP(args.gamma, P, R_mat)
    
    # build features
    Phi = np.stack([np.array(sum([[i]*grid_n for i in range(grid_n)], [])), np.array(list(range(grid_n))*grid_n)], axis = 1)
    Phi = np.concatenate([Phi, np.ones((args.n,1))], axis = 1) 

    np.random.seed(args.seed)
    V = mlp.MLP(Phi, [args.width]*args.depth, biases = False, activation='ReLU')
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

    return mlp_V_to_V_star, mlp_V_to_origin, condition_numbers, bound, smoothness


    




def main():
    args = parse_args()
    print(args.online)

    seeds = range(1,11)
    
    tVs, tVo, Vs, Vo, b, s, c = [], [], [], [], [], [], []
    for seed in seeds:

        args.seed = seed
        mlp_Vs, mlp_Vo, cond, bound, smoothness = run_experiment(args)
        Vs.append(mlp_Vs)
        Vo.append(mlp_Vo)
        b.append(bound)
        s.append(smoothness)
        c.append(cond)
    
    Vs = np.array(Vs)
    Vo = np.array(Vo)
    b = np.array(b)
    s = np.array(s)
    c = np.array(c)

    np.savez(args.save_path  + "grid_k_" + str(args.k) + "_n_" + str(args.n) + "_mlp_depth_" + str(args.depth) 
            + "_width_" + str(args.width) + "_online_" + str(args.online) + ".npz", 
            mlp_Vs = Vs, mlp_Vo = Vo, bound = b, seeds= np.array(seeds), 
            smoothness= s, conditions_numbers = c)


if __name__ == "__main__":
    main()
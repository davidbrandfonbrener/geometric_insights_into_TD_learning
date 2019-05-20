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
    parser.add_argument('--n', default=20, type=int)
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
    parser.add_argument('--hard', default=False, type=bool)
    parser.add_argument('--save_path', default="../outputs/data/homog_exp/")

    args = parser.parse_args()
    return args

def run_experiment(args):


    P = P_matrices.build_cyclic_P(args.n, args.delta)

    if args.hard:
        V_star = np.zeros(args.n)
        for i in range(0, args.n, 2):
            V_star[i] = 1
        env = environment.MRP(args.gamma, P, V_star=V_star)
    else:
        R_mat = np.zeros_like(P)
        R_mat[0, 1] = 1
        env = environment.MRP(args.gamma, P, R_mat)
    

    # build features
    angles = np.linspace(0, 2 * np.pi, args.n, endpoint=False)
    Phi = np.concatenate([np.expand_dims(np.sin(angles), 1), np.expand_dims(np.cos(angles), 1), np.ones((args.n,1))], axis = 1) 

    np.random.seed(args.seed)
    V = mlp.MLP(Phi, [args.width]*args.depth, biases = False, activation='tanh')
    if args.online:
        Vs, thetas, _, _ = online_td0.TD0(V, env, args.stepsize, args.steps, args.log_idx, args.plot_step)
    else:
        thetas, Vs = expected_tdk.TDk(args.k, V, env, args.stepsize, args.steps, args.log_idx, args.plot_step)
    tanh_mlp_V_to_V_star = utils.dist_mu(env.mu, env.V_star, jnp.array(Vs))
    tanh_mlp_V_to_origin = utils.mu_norm(env.mu, jnp.array(Vs))
    
    ts = thetas[args.plot_start::args.plot_step]
    tanh_condition_numbers = []
    for i in range(len(ts)):
        V.theta = ts[i]
        tanh_condition_numbers.append(utils.jac_cond(V.jacobian()))

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

    return tanh_mlp_V_to_V_star, tanh_mlp_V_to_origin, tanh_condition_numbers, mlp_V_to_V_star, mlp_V_to_origin, condition_numbers, bound, smoothness


    




def main():
    args = parse_args()
    print("online: ", args.online)
    print("hard: ", args.hard)

    seeds = range(1,11)
    
    tVs, tVo, Vs, tc, Vo, b, s, c = [], [], [], [], [], [], [], []
    for seed in seeds:

        args.seed = seed
        tanh_mlp_Vs, tanh_mlp_Vo, tanh_cond, mlp_Vs, mlp_Vo, cond, bound, smoothness = run_experiment(args)
        tVs.append(tanh_mlp_Vs)
        tVo.append(tanh_mlp_Vo)
        tc.append(tanh_cond)
        Vs.append(mlp_Vs)
        Vo.append(mlp_Vo)
        b.append(bound)
        s.append(smoothness)
        c.append(cond)
    
    tVs = np.array(tVs)
    tVo = np.array(tVo)
    tc = np.array(tc)
    Vs = np.array(Vs)
    Vo = np.array(Vo)
    b = np.array(b)
    s = np.array(s)
    c = np.array(c)

    np.savez(args.save_path  + "hard_" + str(args.hard) + "_k_" + str(args.k) + "_n_" + str(args.n) + "_mlp_depth_" + str(args.depth) 
            + "_width_" + str(args.width) + "_online_" + str(args.online) + ".npz", 
            tanh_mlp_Vs = tVs, tanh_mlp_Vo = tVo, tanh_condition_numbers = tc, mlp_Vs = Vs, mlp_Vo = Vo, bound = b, seeds= np.array(seeds), 
            smoothness= s, condition_numbers = c)


if __name__ == "__main__":
    main()
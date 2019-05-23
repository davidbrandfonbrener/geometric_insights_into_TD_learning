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
    parser.add_argument('--width', default=5, type=int)
    parser.add_argument('--depth', default = 1, type=int)
    parser.add_argument('--log_idx', default=1000, type=int)
    parser.add_argument('--steps', default=10000, type=int)
    parser.add_argument('--stepsize', default=0.1, type=float)
    parser.add_argument('--plot_start', default=0, type=int)
    parser.add_argument('--plot_step', default=100, type=int)
    parser.add_argument('--gamma', default=0.9)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--online', default=False, type=bool)
    parser.add_argument('--save_path', default="../outputs/data/under_homog_exp/")

    args = parser.parse_args()
    return args

def run_experiment(args):


    P = P_matrices.build_cyclic_P(args.n, args.delta)

    R_mat = np.zeros_like(P)
    R_mat[0, 1] = 1
    env = environment.MRP(args.gamma, P, R_mat)

    # build features
    angles = np.linspace(0, 2 * np.pi, args.n, endpoint=False)
    Phi = np.concatenate([np.expand_dims(np.sin(angles), 1), np.expand_dims(np.cos(angles), 1), np.ones((args.n,1))], axis = 1) 


    # np.random.seed(args.seed)
    # V = mlp.MLP(Phi, [args.width]*args.depth, biases = False, activation='tanh')
    # if args.online:
    #     Vs, thetas, _, _ = online_td0.TD0(V, env, args.stepsize, args.steps, args.log_idx, args.plot_step)
    # else:
    #     thetas, Vs = expected_tdk.TDk(args.k, V, env, args.stepsize, args.steps, args.log_idx, args.plot_step)
    # tanh_mlp_V_to_V_star = utils.dist_mu(env.mu, env.V_star, jnp.array(Vs))
    
    # ts = thetas[args.plot_start::args.plot_step]
    # tanh_dynamics = []
    # for i in range(len(ts)):
    #     V.theta = ts[i]
    #     tanh_dynamics.append(utils.dynamics_norm(V, env.A, env.V_star))
    # tanh_params = []
    # for i in range(len(ts)):
    #     theta = np.concatenate([x.flatten() for x in ts[i]]).ravel()
    #     tanh_params.append(np.linalg.norm(theta))
    tanh_params, tanh_dynamics, tanh_mlp_V_to_V_star = [],[],[]

    np.random.seed(args.seed)
    V = mlp.MLP(Phi, [args.width]*args.depth, biases = False, activation='ReLU')
    if args.online:
        Vs, thetas, _, _ = online_td0.TD0(V, env, args.stepsize, args.steps, args.log_idx, args.plot_step)
    else:
        thetas, Vs = expected_tdk.TDk(args.k, V, env, args.stepsize, args.steps, args.log_idx, args.plot_step)
    mlp_V_to_V_star = utils.dist_mu(env.mu, env.V_star, jnp.array(Vs))

    ts = thetas[args.plot_start::args.plot_step]
    dynamics = []
    for i in range(len(ts)):
        V.theta = ts[i]
        dynamics.append(utils.dynamics_norm(V, env.A, env.V_star))
    params = []
    for i in range(len(ts)):
        theta = np.concatenate([x.flatten() for x in ts[i]]).ravel()
        params.append(np.linalg.norm(theta))


    bound = utils.overparam_cond_number_bound(env.P, env.mu, env.gamma, args.k)

    return tanh_mlp_V_to_V_star, tanh_dynamics, tanh_params, mlp_V_to_V_star, dynamics, params, bound


    




def main():
    args = parse_args()
    print("online: ", args.online)

    seeds = range(7,10)
    
    tVs, td, tp, Vs, d, p, b= [], [], [], [], [], [], []
    for seed in seeds:
        print("Seed: ", seed)
        print("--------------")
        args.seed = seed
        tanh_mlp_Vs, tanh_dyn, tanh_par, mlp_Vs, dyn , par, bound = run_experiment(args)
        tVs.append(tanh_mlp_Vs)
        td.append(tanh_dyn)
        tp.append(tanh_par)
        Vs.append(mlp_Vs)
        d.append(dyn)
        p.append(par)
        b.append(bound)
    
    tVs = np.array(tVs)
    td = np.array(td)
    tp = np.array(tp)
    Vs = np.array(Vs)
    d = np.array(d)
    p = np.array(p)
    b = np.array(b)

    np.savez(args.save_path + "long2_k_" + str(args.k) + "_n_" + str(args.n) + "_mlp_depth_" + str(args.depth) 
            + "_width_" + str(args.width) + "_online_" + str(args.online) + ".npz", 
            tanh_mlp_Vs = tVs, tanh_dynamics = td, mlp_Vs = Vs, dynamics = d, bound = b, seeds= np.array(seeds),
            tanh_params = tp, params = p)


if __name__ == "__main__":
    main()
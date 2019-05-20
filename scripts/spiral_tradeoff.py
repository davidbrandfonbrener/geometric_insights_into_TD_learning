from td.funcs import function
from td.envs import environment, P_matrices
from td.algs import expected_td0, expected_tdk, online_td0, numpy_expected_tdk
from td.funcs import mlp, simple, spiral
from td.utils import utils

import numpy as np
from matplotlib import pyplot as plt

import argparse

#import jax.numpy as jnp

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', default=3, type=int)
    parser.add_argument('--k', default=1, type=int)
    #parser.add_argument('--width', default=10)
    #parser.add_argument('--depth', default = 2)
    parser.add_argument('--log_idx', default=1000, type=int)
    parser.add_argument('--steps', default=100000, type=int)
    parser.add_argument('--stepsize', default=0.000005)
    parser.add_argument('--plot_start', default=0, type=int) 
    parser.add_argument('--plot_step', default=100, type=int)
    parser.add_argument('--gamma', default=0.9)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--online', default=False, type=bool)
    parser.add_argument('--save_path', default="../outputs/data/spiral/")

    args = parser.parse_args()
    return args



def run_experiment(args):

    np.random.seed(args.seed)

    P = P_matrices.build_cyclic_P(args.n, args.delta)

    R_mat = np.zeros_like(P)
    env = environment.MRP(args.gamma, P, R_mat)

    bound = utils.overparam_cond_number_bound(env.P, env.mu, env.gamma, args.k)
    

    orientation = -1 * np.ones(args.n)
    orientation[-1] = args.n - 1
    orientation = 10.0 * orientation
    init_conditions = -20.0 
    print(orientation)
    epsilon = 0.05
    P_spir= P_matrices.build_cyclic_P(args.n, 0.5)
    V = spiral.Spiral(init_conditions, P_spir, orientation, epsilon)
    if args.online:
        Vs, thetas, _, _ = online_td0.TD0(V, env, args.stepsize, args.steps, args.log_idx, args.plot_step)
    else:
        thetas, Vs = numpy_expected_tdk.TDk(args.k, V, env, args.stepsize, args.steps, args.log_idx, args.plot_step)
    spiral_Vs = utils.dist_mu(env.mu, env.V_star, np.array(Vs))

    ts = thetas[args.plot_start::args.plot_step]
    condition_numbers = []
    for i in range(len(ts)):
        V.theta = ts[i]
        condition_numbers.append(utils.jac_cond(V.jacobian()))

    V = simple.Tabular(Vs[0])
    if args.online:
        Vs, thetas, _, _ = online_td0.TD0(V, env, args.stepsize, args.steps, args.log_idx, args.plot_step)
    else:
        thetas, Vs = numpy_expected_tdk.TDk(args.k, V, env, args.stepsize, args.steps, args.log_idx, args.plot_step)
    tabular_Vs = utils.dist_mu(env.mu, env.V_star, np.array(Vs))
    

    smoothness = max([abs(env.V_star[i] - env.V_star[i-1]) for i in range(args.n)])

    return tabular_Vs, spiral_Vs, condition_numbers, bound, smoothness


    




def main():
    args = parse_args()
    print(args.online)

    deltas = [.5, .4, .3, .27]
    
    t, sp, c, b, s = [], [], [], [], []
    for delta in deltas:

        args.delta = delta
        tabular_Vs, spiral_Vs, condition_numbers, bound, smoothness = run_experiment(args)
        t.append(tabular_Vs)
        sp.append(spiral_Vs)
        c.append(condition_numbers)
        b.append(bound)
        s.append(smoothness)
    
    t = np.array(t)
    sp = np.array(sp)
    c = np.array(c)
    b = np.array(b)
    s = np.array(s)

    np.savez(args.save_path  + "k_" + str(args.k) + "_n_" + str(args.n) + "_seed_" + str(args.seed) + ".npz", 
            tabular_Vs = t, spiral_Vs = sp, condition_numbers = c, bound = b, deltas = deltas, smoothness= s)


if __name__ == "__main__":
    main()
from td.funcs import function
from td.envs import environment
from td.algs import expected_td0, online_td0
from td.funcs import mlp, simple
from td.utils import utils

import numpy as np
from matplotlib import pyplot as plt

import argparse

import jax.numpy as jnp

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', default=20)
    parser.add_argument('--k', default=10)
    parser.add_argument('--delta', default=0.5)
    parser.add_argument('--width', default=100)
    parser.add_argument('--log_idx', default=1000)
    parser.add_argument('--steps', default=10000)
    parser.add_argument('--stepsize', default=0.1)
    parser.add_argument('--plot_start', default=10) 
    parser.add_argument('--plot_step', default=100)
    parser.add_argument('--gamma', default=0.9)
    parser.add_argument('--seed', default=1)
    parser.add_argument('--save_path', default="../outputs/plots/loop_30/")

    args = parser.parse_args()
    return args


def build_P(n, k, delta):
    # initialize a sticky cycle
    P = np.zeros((n,n))
    for i in range(n):
        P[i, i-1] = 0.5
        P[i,i] = 0.5

    # add symmetric connections with prob delta to the first k states
    for i in range(k):
        P[i, i-1] = 0.25
        P[i, (i+1) % n] = 0.25
        P[i,i] = 0.5

    return P



def run_experiment(args):

    P = build_P(args.n, args.k, args.delta)

    R_mat = np.zeros_like(P)
    R_mat[-1, -2] = 1
    
    angles = np.linspace(0, 2 * np.pi, args.n, endpoint=False)
    Phi = np.concatenate([np.expand_dims(np.sin(angles), 1), np.expand_dims(np.cos(angles), 1), np.ones((args.n,1))], axis = 1) 

    env = environment.MRP(args.gamma, P, R_mat)

    np.random.seed(args.seed)


    V = simple.Tabular(np.zeros(args.n))
    thetas, Vs = expected_td0.TD0(V, env, args.stepsize, args.steps, args.log_idx)
    Vs = np.array(Vs)
    plt.plot(utils.dist_mu(env, Vs[args.plot_start::args.plot_step]), color = 'b')


    V = mlp.MLP(Phi, [10, 10, 10], biases = False)
    thetas, Vs = expected_td0.TD0(V, env, args.stepsize, args.steps, args.log_idx)
    Vs = np.array(Vs)
    plt.plot(utils.dist_mu(env, Vs[args.plot_start::args.plot_step]), color = 'g')


    V = simple.Linear(np.zeros(Phi.shape[1]), Phi)
    thetas, Vs = expected_td0.TD0(V, env, args.stepsize, args.steps, args.log_idx)
    Vs = np.array(Vs)
    plt.plot(utils.dist_mu(env, Vs[args.plot_start::args.plot_step]), color = 'r')

    plt.show()




def main():
    args = parse_args()

    for k in [15]:
        args.k = k

        run_experiment(args)


if __name__ == "__main__":
    main()
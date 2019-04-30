from td.funcs import function
from td.envs import environment
from td.algs import expected_td0, online_td0
from td.funcs import mlp, simple

import numpy as np
from matplotlib import pyplot as plt

import argparse

import jax.numpy as jnp


def reversing_loop(n, k, delta):
    # initialize a sticky cycle
    P = np.zeros((n,n))
    for i in range(n):
        P[i, i-1] = 0.1
        P[i,i] = 0.9

    # add symmetric connections with prob delta to the first k states
    for i in range(k):
        P[i, i-1] = delta
        P[i, (i+1) % n] = delta 
        P[i,i] = 1 - delta * 2

    # find mu
    vals, vecs = np.linalg.eig(np.transpose(P))
    mu= np.zeros(n)
    for i in range(n):
        #print(vals[i])
        if np.abs(vals[i] - 1) < 1e-6:
            mu = vecs[:,i]
            break
    assert np.sum(mu) != 0
    mu = np.array( mu / np.sum(mu), dtype=float)
    #print(mu)

    return P, mu


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
        out[i] = jnp.dot(v, env.mu * v)
    return out

def theta_norm(thetas):
    out = np.ones(len(thetas))
    for j, t in enumerate(thetas):
        for i in range(len(t)):
            out[j] = out[j] * np.linalg.norm(t[i])
    
    return out


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', default=20)
    parser.add_argument('--k', default=10)
    parser.add_argument('--delta', default=0.5)
    parser.add_argument('--width', default=100)
    parser.add_argument('--offline', default=True)
    parser.add_argument('--steps', default=10000)
    parser.add_argument('--stepsize', default=0.05)
    parser.add_argument('--plot_start', default=100) 
    parser.add_argument('--plot_step', default=100)
    parser.add_argument('--gamma', default=0.9)
    parser.add_argument('--seed', default=1)
    parser.add_argument('--save_path', default="../outputs/plots/loop_30/")

    args = parser.parse_args()
    return args



def main():
    args = parse_args()

    for k in [0, 5, 10, 15, 20, 25, 30]:
        args.k = k

        run_experiment(args)



        # P, mu = reversing_loop(args.n, args.k, 0.25)
        # env = environment.MRP(args.gamma, P, mu, np.zeros_like(P))
        # S = 0.5 * (env.A + np.transpose(env.A))
        # R = 0.5 * (env.A - np.transpose(env.A))
        # print ("k: ", k)
        # print ("S max: ", np.max(np.linalg.eig(S)[0]))
        # print ("S min: ", np.min(np.linalg.eig(S)[0]))
        # print ("R max: ", np.max(np.linalg.eig(R)[0]) * (0 - 1j) )
        # print ("R min: ", np.min(np.linalg.eig(R)[0]) * (0 - 1j) )


def run_experiment(args):

    P, mu = reversing_loop(args.n, args.k, args.delta)

    R_mat = np.zeros_like(P)
    R_mat[-1, -2] = 1
    
    angles = np.linspace(0, 2 * np.pi, args.n, endpoint=False)
    Phi = np.concatenate([np.expand_dims(np.sin(angles), 1), np.expand_dims(np.cos(angles), 1), np.ones((args.n,1))], axis = 1) 
    #print(Phi)


    env = environment.MRP(args.gamma, P, R_mat)
    #print(env.V_star)

    np.random.seed(args.seed)

    if args.offline: 

        #TESTING
        V = mlp.MLP(Phi, [10, 10, 10], biases = True)
        thetas, Vs = expected_td0.TD0(V, env, args.stepsize, args.steps, 100)


        V = function.Tabular(np.zeros(args.n))
        thetas, Vs = expected_td0.TD0(V, env, args.stepsize, args.steps)

        Vs = np.array(Vs)
        plt.plot(dist_mu(env, Vs[args.plot_start::args.plot_step]), color = 'b')


        V = function.Linear(np.zeros(Phi.shape[1]), Phi)
        thetas, Vs = expected_td0.TD0(V, env, args.stepsize, args.steps)
        #print(Vs[-1])

        Vs = np.array(Vs)
        plt.plot(dist_mu(env, Vs[args.plot_start::args.plot_step]), color = 'g')


        V = function.TwoLayerNetNoBias(Phi, args.width)
        #V = function.TwoLayerNet(Phi, args.width)
        thetas, Vs = expected_td0.TD0(V, env, args.stepsize, args.steps)
        #print(Vs[-1])
        
        dts = theta_norm(thetas[args.plot_start::args.plot_step])
        

        Vs = np.array(Vs)
        dVs = dist_mu(env, Vs[args.plot_start::args.plot_step])

        plt.plot(dVs / dts, color = 'r')
        plt.plot(dVs, color = 'orange')
        plt.show()
        plt.plot(dts)
        plt.show()
        


    else:

        V = function.Tabular(np.zeros(args.n))
        Vs, thetas, rs, ss = online_td0.TD0(V, env, args.stepsize, args.steps)

        Vs = np.array(Vs)
        plt.plot(dist_mu(env, Vs[args.plot_start::args.plot_step]), color = 'b')

        V = function.Linear(Phi.shape[1], Phi)
        Vs, thetas, rs, ss = online_td0.TD0(V, env, args.stepsize, args.steps)
        print(Vs[-1])

        Vs = np.array(Vs)
        plt.plot(dist_mu(env, Vs[args.plot_start::args.plot_step]), color = 'r')


        V = function.TwoLayerNetNoBias(Phi, args.width)
        Vs, thetas, rs, ss = online_td0.TD0(V, env, args.stepsize, args.steps)
        print(Vs[-1])

        Vs = np.array(Vs)
        plt.plot(dist_mu(env, Vs[args.plot_start::args.plot_step]), color = 'orange')

    
    plt.xlabel("Training time")
    plt.ylabel("Distance to V*")
    plt.title("n = "+str(args.n)+", k = "+str(args.k))

    plt.savefig(args.save_path + "k_" + str(args.k))
    
    plt.close()

    #plt.show()



if __name__ == "__main__":
    main()
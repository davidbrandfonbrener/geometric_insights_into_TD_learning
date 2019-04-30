from td.funcs import spiral, simple, two_layers
from td.envs import environment
from td.algs import expected_td0, online_td0

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



P = np.array([[0.5,0, 0.5],[0.5,0.5,0],[0,0.5,0.5]])
gamma = 0.9
R_mat = np.zeros_like(P)

env = environment.MRP(gamma, P, R_mat)

# Spiral
V = spiral.Spiral(-7.0, P, np.array([-10, -10, 20]), 0.05)

thetas, Vs = expected_td0.TD0(V, env, 0.00001, 25000)

Vs = np.array(Vs)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot(Vs[:,0], Vs[:,1], Vs[:,2])


#Tabular expected
V = simple.Tabular(Vs[0,:])

thetas, Vs = expected_td0.TD0(V, env, 0.1, 1000)

Vs = np.array(Vs)

ax.plot(Vs[:,0], Vs[:,1], Vs[:,2])


# Affine expected
V = simple.Affine(np.zeros(2), np.random.random((3,2)), Vs[0,:])

thetas, Vs = expected_td0.TD0(V, env, 0.1, 1000)

Vs = np.array(Vs)

ax.plot(Vs[:,0], Vs[:,1], Vs[:,2])


# # Neural net expected
# V = two_layers.TwoLayerNet(1, 2)
# thetas, Vs = expected_td0.TD0(V, env, 0.1, 1000, np.array([0.0,1.0,2.0]))

# Vs = np.array(Vs)

# ax.plot(Vs[:,0], Vs[:,1], Vs[:,2])


# show plot
plt.show()





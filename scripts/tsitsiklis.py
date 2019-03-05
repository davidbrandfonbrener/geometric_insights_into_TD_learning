from td.funcs import function
from td.envs import environment
from td.algs import expected_td0, online_td0

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


P = np.array([[0.5,0, 0.5],[0.5,0.5,0],[0,0.5,0.5]])
mu = np.array([1.0/3.0, 1.0/3.0, 1.0/3.0])
gamma = 0.9
R_mat = np.zeros_like(P)

env = environment.MRP(gamma, P, mu, R_mat)

V_star = env.compute_V_star()


# Spiral
V = function.Spiral(-7.0)

thetas, Vs = expected_td0.TD0(V, env, 0.00001, 25000)

Vs = np.array(Vs)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot(Vs[:,0], Vs[:,1], Vs[:,2])
plt.show()





# Tabular expected
# V = function.Tabular(np.array([-3.0, 1.0, 2.0]))

# thetas, Vs = expected_td0.TD0(V, env, 0.1, 1000)

# Vs = np.array(Vs)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# ax.plot(Vs[:,0], Vs[:,1], Vs[:,2])
# plt.show()



# Tabular online 
# V = function.Tabular(np.array([-3.0, 1.0, 2.0]))

# thetas, Rs, Ss = online_td0.TD0(V, env, 0.1, 1000)

# thetas = np.array(thetas)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# ax.plot(thetas[:,0], thetas[:,1], thetas[:,2])
# plt.show()


# Two-layer online
# V = function.TwoLayerNet(1, 3)

# thetas, Rs, Ss = online_td0.TD0(V, env, 0.1, 1000)

# Vs = []
# for i in range(len(thetas)):
#     vtheta = [V._forward(thetas[i], 0), V._forward(thetas[i], 1), V._forward(thetas[i], 2)] 
#     Vs.append(vtheta)

# Vs = np.array(Vs)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# ax.plot(Vs[0,:], Vs[1,:], Vs[2,:])
# plt.show()


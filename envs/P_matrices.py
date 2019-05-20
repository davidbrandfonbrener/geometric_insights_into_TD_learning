import numpy as np

def build_cyclic_P(n, delta):

    assert 0 <= delta
    assert delta <= 0.5

    P = np.zeros((n,n))
    # add reverse connections with prob delta
    for i in range(n):
        P[i, i-1] = 0.5 - delta
        P[i, (i+1) % n] = delta
        P[i,i] = 0.5

    return P

def constant_gridworld(n, up, down, left, right, self_loop):

    assert up + down + left + right +  self_loop == 1.0

    P = np.zeros((n*n, n*n))
    for i in range(n):
        for j in range(n):
            
            P[i*n + j, i*n + j] = self_loop
            P[i*n + j, i*n + min(j + 1, n-1)] += right
            P[i*n + j, i*n + max(j - 1, 0)] += left
            P[i*n + j, min(i+1, n-1) * n + j] += up
            P[i*n + j, max(i-1, 0) * n + j] += down

    assert abs(np.sum(P) - n*n) < 1e-8

    # bottom corner sends you back to the start
    P[-1, :] = np.zeros(n*n)
    P[-1, 0] = 1

    return P
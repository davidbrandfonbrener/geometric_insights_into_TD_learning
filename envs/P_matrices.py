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
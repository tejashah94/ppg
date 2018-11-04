""" lasso.py
    
    Solutions to LASSO for recovery from compressed samples.

    Currently have the following algorithms:
        1) Coordinate descent
"""

import numpy as np

# Coordinate descent-based LASSO solution stolen from sklearn
def cd_lasso(y, A, x0=None, l1_lambda=0.01, tol=1.0e-2):
    N, D = A.shape

    A_norm2 = np.square(A).sum(axis=0)

    if x0 is None:
        # x0 = np.random.random(size=(D,))
        x0 = np.ones((D,)) * 0.5

    x = np.copy(x0)
    r = y - np.dot(A, x)
    max_xi = np.max(np.abs(x))

    while True:
        max_dxi = 0
        for i in range(D):
            if A_norm2[i] == 0.0:
                continue
            x_i0 = x[i]

            A_i = A[:,i]
            r += A_i * x[i]
            rho_i = (r * A_i).sum()
            sign = 1 if rho_i > 0 else -1
            x[i] = (sign * max(abs(rho_i) - l1_lambda, 0)) / A_norm2[i]

            dxi = abs(x[i] - x_i0)
            max_dxi = dxi if dxi > max_dxi else max_dxi

            r -= x[i] * A_i
            max_xi = max(max_xi, x[i])
        if (max_dxi / max_xi) < tol:
            break
    return x

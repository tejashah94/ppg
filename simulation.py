#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model

# Simulation to make sure we can run the compressive sampling algorithm on some
# data at least.

T = 600 # s
T_window = 60 # s
assert T % T_window == 0, 'No incomplete windows!'

f0 = 4 # Hz
T0 = 1.0 / f0 # s

N0 = int(T / T0)
num_windows = 2 * (T // T_window) - 1
L_window = N0 // (T // T_window)
assert L_window % 4 == 0, 'Window length should be a multiple of 4!'

# deltas = np.random.exponential(scale=Ts, size=(N,))
deltas = np.ones((N0,)) * T0
t0 = np.cumsum(deltas)

A0 = 1.0 + np.random.random((N0,)) * 0.4
b0 = np.random.random() * 2 * np.pi
sigma0 = 0.1
x0 = np.sin(2 * np.pi * t0 - b0) + np.random.randn(N0) * sigma0

X0 = A0 * x0
# plt.plot(t0, X0); plt.show()

psi = np.empty((L_window, L_window))
n, k = np.meshgrid(np.arange(L_window), np.arange(L_window))
psi = np.cos((np.pi / L_window) * (n + 0.5) * k)
psi[0,:] /= (1.0 / np.sqrt(2))
psi *= np.sqrt(2.0 / L_window)

# Randomly permute rows
phi = np.random.permutation(np.eye(L_window))
M = L_window // 12
phi = phi[:M,:]

A = np.dot(phi, psi.T)
Xr = np.empty(X0.shape)

for i in range(0, N0 - L_window + 1, L_window // 2):
    X_window = X0[i:i+L_window]
    Y = np.dot(phi, X_window)

    lasso = linear_model.Lasso(alpha=0.01, fit_intercept=False)
    lasso.fit(A, Y)
    s = lasso.coef_
    xr = np.dot(psi.T, s)
    Xr[i+L_window//4:i+3*L_window//4] = xr[L_window//4:3*L_window//4]

plt.plot(t0, X0, label='original')
plt.plot(t0, Xr, label='reconstructed')
plt.legend()
plt.show()

print('Corrcoef: ', np.corrcoef(X0[L_window//4:-L_window//4],
                                Xr[L_window//4:-L_window//4]))

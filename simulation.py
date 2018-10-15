#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model

# Simulation to make sure we can run the compressive sampling algorithm on some
# data at least.

T = 60 # s
f0 = 4 # Hz
T0 = 1.0 / f0 # s

N0 = int(T / T0)
# deltas = np.random.exponential(scale=Ts, size=(N,))
deltas = np.ones((N0,)) * T0
t0 = np.cumsum(deltas)

A0 = 1.0 + np.random.random((N0,)) * 0.4
b0 = np.random.random() * 2 * np.pi
sigma0 = 0.1
x0 = np.sin(2 * np.pi * t0 - b0) + np.random.randn(N0) * sigma0

X0 = A0 * x0
# plt.plot(t0, X0); plt.show()

psi = np.empty((N0, N0))
n, k = np.meshgrid(np.arange(N0), np.arange(N0))
psi = np.cos((np.pi / N0) * (n + 0.5) * k)
psi[0,:] /= (1.0 / np.sqrt(2))
psi *= np.sqrt(2.0 / N0)

# Randomly permute rows
phi = np.random.permutation(np.eye(N0))
M = N0 // 12
phi = phi[:M,:]

A = np.dot(phi, psi.T)
Y = np.dot(phi, X0)

lasso = linear_model.Lasso(alpha=0.01, fit_intercept=False)
lasso.fit(A, Y)
s = lasso.coef_
print(s)
Xr = np.dot(psi.T, s)

plt.plot(t0, X0, label='original')
plt.plot(t0, Xr, label='reconstructed')
plt.legend()
plt.show()

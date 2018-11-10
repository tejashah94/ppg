#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

# from sklearn import linear_model
from lasso import cd_lasso

# Simulation to make sure we can run the compressive sampling algorithm on some
# data at least.
datafile_name = 'data/samples.csv'
reconstruction_filename = 'reconstructed_py.csv'

data = np.loadtxt(datafile_name, delimiter=',')

t0 = data[:,0]
X0 = data[:,1]
N0 = t0.shape[0]

f0 = 4          # Hz
T0 = 1.0 / f0   # s

T_window = 60   # s
N_window = int(T_window // T0)
if N_window % 4 != 0:
    N_window = N_window - (N_window % 4)

# Incomplete windows are too much of a hassle to deal with right now.
N0 = N0 - (N0 % (N_window // 2))
t0 = t0[:N0]
X0 = X0[:N0]

psi = np.empty((N_window, N_window))
n, k = np.meshgrid(np.arange(N_window), np.arange(N_window))
psi = np.cos((np.pi / N_window) * (n + 0.5) * k)
psi[0,:] *= (1.0 / np.sqrt(2))
psi *= np.sqrt(2.0 / N_window)

# Random permutation of identity matrix, chopped off to take fewer samples.
phi_mat_filename = 'data/phi_mat.csv'
M = N_window // 12
try:
    phi = np.loadtxt(phi_mat_filename, delimiter=',')
except OSError:
    phi = np.random.permutation(np.eye(N_window))
    phi = phi[:M,:]

A = np.dot(phi, psi.T)
Xr = np.empty(X0.shape)

for i in range(0, N0 - N_window + 1, N_window // 2):
    X_window = X0[i:i+N_window]
    Y = np.dot(phi, X_window)

    # lasso = linear_model.Lasso(alpha=0.01, fit_intercept=False)
    # lasso.fit(A, Y)
    # s = lasso.coef_
    s = cd_lasso(Y, A)
    xr = np.dot(psi.T, s)
    Xr[i+N_window//4:i+3*N_window//4] = xr[N_window//4:3*N_window//4]

plt.plot(t0, X0, label='original')
plt.plot(t0, Xr, label='reconstructed')
plt.legend()
plt.show()

print('Corrcoef: ', np.corrcoef(X0[N_window//4:-N_window//4],
                                Xr[N_window//4:-N_window//4]))

out_data = np.hstack((t0.reshape(-1, 1), Xr.reshape(-1, 1)))
np.savetxt(reconstruction_filename, out_data, delimiter=',')

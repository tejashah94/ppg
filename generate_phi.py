#!/usr/bin/env python

import numpy as np

phi_mat_file = 'phi_mat.csv'
phi_flags_file = 'phi_flags.csv'

f0 = 4 # Hz
T0 = 1.0 / f0 # s
T_window = 60 # s
N_window = int(T_window // T0)
if N_window % 4 != 0:
    N_window -= N_window % 4
M = N_window // 12

chosen = np.random.choice(N_window, size=M, replace=False)

phi_flags = np.zeros((N_window,), dtype=np.int)
phi_mat = np.zeros((M, N_window), dtype=np.int)

phi_flags[chosen] = 1
for i, c in enumerate(chosen):
    phi_mat[i,c] = 1

np.savetxt(phi_mat_file, phi_mat, fmt='%d', delimiter=',')
np.savetxt(phi_flags_file, phi_flags, fmt='%d', delimiter='\n')

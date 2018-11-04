#!/usr/bin/env python

""" generate_sim_data.py
    --------------------
    
    Generates a noisy signal with an underlying 1 Hz cosine wave, to simulate
    what we may expect when dealing with the PPG signal.

    The code fixes some of the signal parameters as constants, and outputs the
    generated signal as a text file where each sample is stored in a new line.
    This should hopefully be simple enough that we can easily get the real PPG
    sensor measurements into the same format. Then the processing stays the same
    when we switch from simulation to reality.

    File format:
    (t, s)  # not part of the file because I don't want to have to strip it out
            # later anyway

    0, 123.45
    0.442, 121.31
    0.513, 123.41
    1.112, 122.84
    ...
    ...
    ...
"""

import numpy as np

output_datafile_name = 'samples.csv'

T = 600             # seconds
f0 = 4              # Hz, the rate at which this signal is being generated
T0 = 1.0 / f0       # seconds

N0 = int(T // T0)   # how many samples

deltas = np.ones((N0,)) * T0    # uniform inter-sample interval, can be changed
t0 = np.cumsum(deltas)          # actual samples

amp_A = 1.0                                         # AM variation amplitude
A = 1.0 + (np.random.random((N0,)) - 0.5) * amp_A   # actual AM

# A can be made more smooth...
# So let's smooth it!
smoothing_window_length = 2.0   # seconds
L0 = int(smoothing_window_length // T0)
smoothing_window = np.ones((L0,)) / L0
A0 = np.convolve(A, smoothing_window, mode='same')

# Random initial phase
b0 = np.random.random() * 2 * np.pi

# Additive noise
sigma0 = 0.15
u0 = np.random.randn(N0) * sigma0

# Final signal
x0 = A0 * (np.cos(2 * np.pi * t0 - b0) + u0)

# Collect sampling instants and values together
out_data = np.hstack((t0.reshape(-1, 1), x0.reshape(-1, 1)))

# Dump
np.savetxt(output_datafile_name, out_data, delimiter=',')

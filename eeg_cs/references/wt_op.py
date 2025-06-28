"""
.. _gallery:lop:wt_ops:

Wavelet Transform Operators
==============================

.. contents::
    :depth: 2
    :local:

This example demonstrates following features:

- ``cr.sparse.lop.convolve2D`` A 2D convolution linear operator
- ``cr.sparse.sls.lsqr`` LSQR algorithm for solving a least square problem on 2D images


"""

# %% 
# Let's import necessary libraries 
import jax.numpy as jnp
# For plotting diagrams
import matplotlib.pyplot as plt
## CR-Sparse modules
import cr.nimble as cnb
# Linear operators
from cr.sparse import lop
# Utilities
from cr.nimble.dsp import time_values
# Configure JAX for 64-bit computing
# from jax.config import config
# config.update("jax_enable_x64", True)

# %%
# 1D Wavelet Transform Operator
# ---------------------------------------------------


# %%
# A signal consisting of multiple sinusoids 
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# Individual sinusoids have different frequencies and amplitudes.
# Sampling frequency
fs = 1000.
# Time duration
T = 2
# time values
t = time_values(fs, T)
# Number of samples
n = t.size
x = jnp.zeros(n)
freqs = [25, 7, 9, 15, 30, 50]
amps = [1, -3, .8, 2, -1.5, 0.5]
for  (f, amp) in zip(freqs, amps):
    sinusoid = amp * jnp.sin(2 * jnp.pi * f * t)
    x = x + sinusoid
# Plot the signal
plt.figure(figsize=(8,2))
plt.plot(t, x, 'k', label='Composite signal')

# %%
# 1D wavelet transform operator
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
DWT_op = lop.dwt(n, wavelet='dmey', level=5)

# %%
# Wavelet coefficients
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
alpha = DWT_op.times(x)
plt.figure(figsize=(8,2))
plt.plot(alpha, label='Wavelet coefficients')

# %%
# Compression
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# Let's keep only 10 percent of the coefficients
cutoff = n // 2
alpha2 = alpha.at[cutoff:].set(0)
plt.figure(figsize=(8,2))
plt.plot(alpha2, label='Wavelet coefficients after compression')

# %%
# Reconstruction
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
x_rec = DWT_op.trans(alpha2)
# RMSE 
rmse = cnb.root_mse(x, x_rec)
print(rmse)
# SNR 
snr = cnb.signal_noise_ratio(x, x_rec)
print(snr)
plt.figure(figsize=(8,2))
plt.plot(x, 'k', label='Original')
plt.plot(x_rec, 'r', label='Reconstructed')
plt.title('Reconstructed signal')
plt.legend()

# %%

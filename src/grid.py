import numpy as np


def make_grid(N=128, L=2 * np.pi):
    """
    Uniform Cartesian grid on [0, L) x [0, L), fully periodic (bounded domain).

    Wavenumbers are integer-valued for L = 2pi: k = 0, 1, ..., N/2-1, -N/2, ..., -1.
    K2_inv sets the k=0 mode to zero (enforces zero-mean stream functions).
    """
    x      = np.linspace(0, L, N, endpoint=False)
    X, Y   = np.meshgrid(x, x, indexing='ij')
    k      = np.fft.fftfreq(N) * N
    KX, KY = np.meshgrid(k, k, indexing='ij')
    K2     = KX**2 + KY**2
    K2_inv = np.where(K2 == 0, 0.0, 1.0 / K2)
    return X, Y, KX, KY, K2, K2_inv
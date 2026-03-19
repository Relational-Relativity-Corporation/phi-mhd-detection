import numpy as np


def harris_sheet(X, Y, eps=0.1):
    """
    Periodic sinusoidal current sheet (Harris-like approximation).

    Biskamp (2003) convention:
        B = (-dpsi/dy, dpsi/dx)    Bx = -dpsi/dy, By = dpsi/dx
        j = nabla^2 psi,   omega = nabla^2 phi

    psi = cos(y)  ->  Bx = sin(y), By = 0.
    Field reverses at y = 0, pi, 2pi. |j| peaks at y = pi/2, 3pi/2.
    Perturbation eps*cos(x)*cos(y) breaks x-symmetry and seeds the tearing mode.
    omega = 0 (no initial velocity field).

    Domain: [0, 2pi] x [0, 2pi], fully periodic. nabla.B = 0 by construction.
    """
    psi   = np.cos(Y) + eps * np.cos(X) * np.cos(Y)
    omega = np.zeros_like(psi)
    return psi, omega


def orszag_tang(X, Y, eps=0.05):
    """
    Orszag-Tang vortex -- standard 2D MHD turbulent reconnection benchmark.

    psi = -(cos y + 0.5 cos 2x)   ->   Bx = sin y,  By = -sin 2x
    phi = -(cos x + cos y)         ->   vx = sin y,  vy = -sin x
    omega = nabla^2 phi = cos x + cos y

    Multiple current sheets, fast turbulent reconnection, well-characterised
    energy cascade. Small symmetry-breaking perturbation added to psi.

    Domain: [0, 2pi] x [0, 2pi], fully periodic. nabla.B = 0 by construction.
    """
    psi   = -(np.cos(Y) + 0.5 * np.cos(2 * X)) + eps * np.cos(X + Y)
    omega =   np.cos(X) + np.cos(Y)
    return psi, omega
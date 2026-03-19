"""
2D pseudo-spectral incompressible resistive MHD solver.

Governing equations (Biskamp 2003, mu0 = rho = 1):
    dpsi/dt   = -(v.grad)psi  + eta * nabla^2 psi
    domega/dt = -(v.grad)omega + (B.grad)j + nu * nabla^2 omega

State   : psi (magnetic flux function), omega (vorticity)
Derived : j  = nabla^2 psi
          phi = -nabla^{-2} omega
          Bx = -dpsi/dy,  By =  dpsi/dx
          vx = -dphi/dy,  vy =  dphi/dx

nabla.B = 0 exactly: B = curl(psi z-hat) is divergence-free by construction.
Time integration: explicit RK2 (midpoint method).
Spatial derivatives: FFT (exact to machine precision).
"""
import numpy as np

_f  = np.fft.fft2
_if = lambda h: np.real(np.fft.ifft2(h))


def _rhs(psi, omega, eta, nu, KX, KY, K2, K2_inv):
    ph  = _f(psi);   oh = _f(omega)

    j   = _if(-K2 * ph)
    vsh = -K2_inv * oh
    Bx  = _if(-1j * KY * ph)
    By  = _if( 1j * KX * ph)
    vx  = _if(-1j * KY * vsh)
    vy  = _if( 1j * KX * vsh)

    px, py = _if(1j*KX*ph),  _if(1j*KY*ph)
    ox, oy = _if(1j*KX*oh),  _if(1j*KY*oh)
    jh     = _f(j)
    jx, jy = _if(1j*KX*jh), _if(1j*KY*jh)

    dpsi   = -(vx*px + vy*py) + eta * j
    domega = -(vx*ox + vy*oy) + (Bx*jx + By*jy) + nu * _if(-K2 * oh)

    return dpsi, domega


def get_fields(psi, omega, KX, KY, K2, K2_inv):
    """Compute derived fields (j, B, v) from state (psi, omega)."""
    ph  = _f(psi);  oh = _f(omega)
    j   = _if(-K2 * ph)
    vsh = -K2_inv * oh
    Bx  = _if(-1j * KY * ph);  By = _if( 1j * KX * ph)
    vx  = _if(-1j * KY * vsh); vy = _if( 1j * KX * vsh)
    return j, Bx, By, vx, vy


def step_rk2(psi, omega, dt, eta, nu, KX, KY, K2, K2_inv):
    """Explicit RK2 (midpoint) step. Returns updated (psi, omega)."""
    d1, w1 = _rhs(psi,             omega,             eta, nu, KX, KY, K2, K2_inv)
    d2, w2 = _rhs(psi + 0.5*dt*d1, omega + 0.5*dt*w1, eta, nu, KX, KY, K2, K2_inv)
    return psi + dt*d2, omega + dt*w2
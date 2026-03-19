"""
ABRCE-layer diagnostics for 2D resistive MHD.

Layer map
---------
A  (Domain)    div_B    -- admissibility check. Zero by construction; verified numerically.
B  (Binding)   delta_j  -- ||J(t) - J(t-dt)||_L2. Rises as current sheet forms.
R  (Invariant) dj2_dt   -- d/dt integral(j^2 dV). Current enstrophy rate; 2D proxy for
                           structural change rate (3D helicity undefined in 2D MHD).
E  (Emission)  E_B, E_K -- magnetic and kinetic energy. Conventional monitoring layer.

Phi(t) = delta_j + dj2_dt
  Phi is a composite detection field, not a conserved physical scalar.
  Its terms are not required to share units -- they share operator-layer
  membership and directional sensitivity to pre-critical structural change.
  Normalise each term to its baseline variance for comparative benchmarking.
"""
import numpy as np


def compute(j, Bx, By, vx, vy, KX, KY, j_prev, j2_prev, dt_diag):
    dV = (2.0 * np.pi)**2 / j.size

    # Operator E
    E_B = 0.5 * float(np.sum(Bx**2 + By**2) * dV)
    E_K = 0.5 * float(np.sum(vx**2 + vy**2) * dV)
    j2  = float(np.sum(j**2) * dV)

    # Operator A
    divB_f = np.real(np.fft.ifft2(
        1j * KX * np.fft.fft2(Bx) + 1j * KY * np.fft.fft2(By)))
    div_B = float(np.max(np.abs(divB_f)))   # expect ~1e-14

    # Operator B
    delta_j = float(np.sqrt(np.sum((j - j_prev)**2) * dV)) if j_prev is not None else 0.0

    # Operator R
    dj2_dt = float(abs(j2 - j2_prev) / dt_diag) if (j2_prev is not None and dt_diag > 0) else 0.0

    phi = delta_j + dj2_dt

    return dict(E_B=E_B, E_K=E_K, j2=j2, div_B=div_B,
                delta_j=delta_j, dj2_dt=dj2_dt, phi=phi)
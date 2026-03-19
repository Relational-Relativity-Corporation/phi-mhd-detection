# phi-mhd-detection

> Smallest possible simulation testing the ABRCE precursor field Phi(t) in 2D resistive MHD.
> **Metatron Dynamics, 2026**

---

## What this tests

The working paper *Relational Structure in Magnetohydrodynamic Systems* proposes that
monitoring at the ABRCE B-layer (current density difference field) and R-layer (current
enstrophy rate) provides structural warning of magnetic reconnection before E-layer
(kinetic energy) signals become anomalous.

**Central hypothesis:**
> Phi(t) = delta_J + dj2/dt  crosses a detection threshold before  Delta_E_K
> shows anomalous reconnection signal.

---

## Physics

2D pseudo-spectral incompressible resistive MHD (Biskamp 2003).
Domain: [0, 2pi] x [0, 2pi], fully periodic (bounded). Units: mu_0 = rho = 1.

    dpsi/dt   = -(v.grad)psi  + eta * nabla^2 psi
    domega/dt = -(v.grad)omega + (B.grad)j + nu * nabla^2 omega

nabla.B = 0 exactly by construction (B = curl(psi z-hat)).
Verified numerically to machine precision at every diagnostic step.
Time integration: RK2. Spatial derivatives: FFT.

---

## Models

| Model | IC | Physics |
|---|---|---|
| `harris` | psi = cos(y) + eps*cos(x)*cos(y), omega = 0 | Sinusoidal current sheet, tearing-unstable, single-mode dominated |
| `orszag-tang` | Standard Orszag-Tang vortex + perturbation | Multiple current sheets, turbulent reconnection, standard benchmark |

Both satisfy the ABRCE domain admissibility condition (bounded, periodic, nabla.B = 0).

---

## ABRCE layer map

| Layer | Field | Physical meaning |
|---|---|---|
| A -- Domain | nabla.B | Admissibility check. Zero by construction. |
| B -- Binding | delta_J = \|\|J(t)-J(t-dt)\|\|_L2 | Current density difference field. Proposed precursor. |
| R -- Invariant | dj2/dt = d/dt integral(j^2 dV) | Current enstrophy rate. 2D structural proxy. |
| E -- Emission | E_B, E_K | Magnetic and kinetic energy. Conventional diagnostics. |

Phi(t) = delta_J + dj2/dt

Note: 3D magnetic helicity is not defined in 2D incompressible MHD.
Current enstrophy rate is the appropriate 2D structural proxy.

---

## On the dimensional composition of Phi(t)

The terms composing Phi(t) -- ||delta_J||_L2 and |dj2/dt| -- are not dimensionally
identical. This is intentional.

Phi(t) is a composite detection field, not a conserved physical scalar. Its terms are
not required to share units -- they are required to share operator-layer membership and
directional sensitivity to pre-critical structural change at the binding and invariant
layers.

In practice, each term should be normalised to its own baseline variance before
summation, making the composite dimensionless and directly comparable across runs and
parameter regimes. The simulation implements unit weights as a first-pass choice.
Normalisation is the correct form for any comparative benchmark against existing
diagnostics.

If challenged on dimensional grounds, the response is:
"Phi is a composite detection field, not a conserved physical scalar."

---

## Setup

```powershell
pip install numpy matplotlib
python run.py
```

Runs both models at 128x128. Completes in under 2 minutes on any modern CPU.

---

## Output

All output written to `results/`:

| File | Contents |
|---|---|
| `{model}_timeseries.png` | 4-panel: E_B, Delta_E_K, delta_J, Phi(t) with event markers |
| `{model}_J_final.png` | J(x,y) heatmap at final timestep |
| `{model}_timeseries.csv` | t, E_B, E_K, j2, delta_j, phi, div_B |
| `{model}_J_final.csv` | Final J(x,y) field (NxN) |

Event markers:
- Gold dotted: Phi(t) first crosses 5% of its maximum
- Red dashed:  Delta_E_K first crosses 5% of its maximum (reconnection onset)
- Lead time = t(onset) - t(Phi crossing)

---

## Parameters

| Flag | Default | Notes |
|---|---|---|
| `--model` | `both` | `harris`, `orszag-tang`, or `both` |
| `--N` | 128 | Grid size NxN |
| `--eta` | 0.01 | Resistivity |
| `--nu` | 0.005 | Viscosity |
| `--dt` | 0.02 | Timestep |
| `--T` | 40.0 | End time (Alfven units) |
| `--eps` | 0.1 | Perturbation amplitude |
| `--outdir` | `results` | Output directory |

Stability: dt * eta * (N/2)^2 < 2. Default parameters give 0.41. Safe.
If reconnection does not complete within T=40, try --eta 0.02 or --T 80.

Dealiasing: not applied at default parameters. If high-k instabilities appear
(grid-scale noise in J_final.png), increase --eta and --nu or reduce --eps.

---

## Numerical notes

- nabla.B = 0 to machine precision by construction. Confirmed in div_B column of CSV.
- Stability check dt*eta*K_max^2 < 2 is a useful heuristic, not a full CFL condition.
- RK2 midpoint method. NaN detection halts run rather than saving corrupted output.

---

## Framework

White paper: [Relational Structure in Magnetohydrodynamic Systems](https://relationalrelativity.dev)
White paper PDF: [relational-structure-in-mhd.pdf](https://github.com/Relational-Relativity-Corporation/metatron-dynamics-framework/blob/main/docs/relational-structure-in-mhd.pdf)
Framework: [Invariant_Relational_Kernel_ABRCE](https://github.com/Relational-Relativity-Corporation/Invariant_Relational_Kernel_ABRCE)
Org: [Relational-Relativity-Corporation](https://github.com/Relational-Relativity-Corporation)
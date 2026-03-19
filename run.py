#!/usr/bin/env python3
"""
phi-mhd-detection  --  smallest possible simulation testing Phi(t)
Metatron Dynamics, 2026

Usage
-----
    python run.py                         # both models, default params
    python run.py --model harris
    python run.py --model orszag-tang
    python run.py --eta 0.02 --T 80       # lower resistivity, longer run
    python run.py --N 256 --dt 0.01       # finer grid
"""
import argparse
import sys
import numpy as np
from pathlib import Path

from src.grid        import make_grid
from src.initial     import harris_sheet, orszag_tang
from src.solver      import step_rk2, get_fields
from src.diagnostics import compute
from src.output      import save


def run(model, N, eta, nu, dt, T, eps, outdir):
    print(f'\n[phi-mhd] model={model}  N={N}  eta={eta}  nu={nu}  '
          f'dt={dt}  T={T}  eps={eps}')

    k_max = N // 2
    stab  = dt * eta * k_max**2
    if stab >= 2.0:
        print(f'[phi-mhd] WARNING: dt*eta*K_max^2 = {stab:.2f} >= 2.0  '
              f'-- consider reducing dt or eta.')

    Path(outdir).mkdir(exist_ok=True)
    X, Y, KX, KY, K2, K2_inv = make_grid(N)

    psi, omega = (harris_sheet(X, Y, eps) if model == 'harris'
                  else orszag_tang(X, Y, eps))

    steps     = int(T / dt)
    rec_every = max(1, steps // 500)

    t_arr, E_B_arr, E_K_arr = [], [], []
    j2_arr, dj_arr, phi_arr, divB_arr = [], [], [], []
    j_prev, j2_prev = None, None
    diverged = False
    j = None

    for step in range(steps):
        psi, omega = step_rk2(psi, omega, dt, eta, nu, KX, KY, K2, K2_inv)

        if np.any(np.isnan(psi)):
            print(f'[phi-mhd] DIVERGED at step {step}  t={step*dt:.3f}',
                  file=sys.stderr)
            diverged = True
            break

        if step % rec_every == 0:
            t = step * dt
            j, Bx, By, vx, vy = get_fields(psi, omega, KX, KY, K2, K2_inv)
            diag = compute(j, Bx, By, vx, vy, KX, KY,
                           j_prev, j2_prev, dt * rec_every)

            t_arr.append(t);             E_B_arr.append(diag['E_B'])
            E_K_arr.append(diag['E_K']); j2_arr.append(diag['j2'])
            dj_arr.append(diag['delta_j']); phi_arr.append(diag['phi'])
            divB_arr.append(diag['div_B'])

            j_prev, j2_prev = j.copy(), diag['j2']

            if step % (rec_every * 25) == 0:
                print(f'  t={t:6.2f}  E_B={diag["E_B"]:.4f}  '
                      f'E_K={diag["E_K"]:.5f}  dJ={diag["delta_j"]:.5f}  '
                      f'Phi={diag["phi"]:.5f}  divB={diag["div_B"]:.1e}')

    if diverged or len(t_arr) < 10 or j is None:
        print('[phi-mhd] Insufficient data -- results not saved.')
        return

    save(model   = model,
         t       = np.array(t_arr),
         E_B     = np.array(E_B_arr),
         E_K     = np.array(E_K_arr),
         j2      = np.array(j2_arr),
         delta_j = np.array(dj_arr),
         phi     = np.array(phi_arr),
         div_B   = np.array(divB_arr),
         j_final = j,
         outdir  = outdir)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Phi(t) MHD precursor detection')
    p.add_argument('--model',  choices=['harris', 'orszag-tang', 'both'], default='both')
    p.add_argument('--N',      type=int,   default=128)
    p.add_argument('--eta',    type=float, default=0.01)
    p.add_argument('--nu',     type=float, default=0.005)
    p.add_argument('--dt',     type=float, default=0.02)
    p.add_argument('--T',      type=float, default=40.0)
    p.add_argument('--eps',    type=float, default=0.1)
    p.add_argument('--outdir', default='results')
    a = p.parse_args()

    models = ['harris', 'orszag-tang'] if a.model == 'both' else [a.model]
    for m in models:
        run(model=m, N=a.N, eta=a.eta, nu=a.nu,
            dt=a.dt, T=a.T, eps=a.eps, outdir=a.outdir)
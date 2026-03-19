import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

_BG   = '#0a0a0a'
_PAN  = '#111111'
_TEXT = '#f0ede6'
_MUT  = '#888783'
_GOLD = '#c9b97a'
_BLUE = '#7a9ec9'
_RED  = '#cc5555'


def _style():
    plt.rcParams.update({
        'figure.facecolor': _BG,  'axes.facecolor': _PAN,
        'axes.edgecolor':  '#2a2a2a',
        'axes.labelcolor': _MUT,  'xtick.color': _MUT, 'ytick.color': _MUT,
        'text.color': _TEXT,      'grid.color': '#1e1e1e',
        'grid.linewidth': 0.5,    'font.size': 10,
        'axes.titlecolor': _TEXT,
    })


def _first_crossing(arr, frac=0.05, start=1):
    thresh = frac * float(np.max(arr))
    hits   = np.where(arr[start:] > thresh)[0]
    return int(hits[0]) + start if len(hits) else len(arr) - 1


def save(model, t, E_B, E_K, j2, delta_j, phi, div_B, j_final, outdir):
    outdir = Path(outdir)
    _style()

    ek_delta  = E_K - E_K[0]
    phi_idx   = _first_crossing(phi,              frac=0.05)
    onset_idx = _first_crossing(ek_delta.clip(0), frac=0.05)
    lead      = max(0.0, t[onset_idx] - t[phi_idx])

    if ek_delta.max() < 1e-6:
        print(f'[out] WARNING: reconnection did not complete within T={t[-1]:.1f}. '
              f'Try --eta 0.02 or --T 80.')

    # time-series panel
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    fig.suptitle(f'ABRCE Precursor Field  Phi(t)   model: {model}',
                 color=_TEXT, fontsize=12, y=0.99)

    rows = [
        (E_B,      r'$E_B$ - magnetic energy',         _MUT),
        (ek_delta, r'$\Delta E_K$ - reconnection KE',  _BLUE),
        (delta_j,  r'$\|\delta J\|_{L^2}$ - B-layer',  _GOLD),
        (phi,      r'$\Phi(t)$ - ABRCE precursor',      _TEXT),
    ]

    for ax, (arr, lbl, col) in zip(axes, rows):
        ax.plot(t, arr, color=col, lw=1.2)
        ax.set_ylabel(lbl, fontsize=9)
        ax.grid(True)
        ax.axvline(t[phi_idx],   color=_GOLD, lw=0.9, ls=':',  alpha=0.85)
        ax.axvline(t[onset_idx], color=_RED,  lw=0.9, ls='--', alpha=0.85)

    axes[-1].set_xlabel('t  (Alfven units)', color=_MUT)
    note = (f'Gold dotted: Phi(t) threshold  |  Red dashed: reconnection onset  |'
            f'  Lead time: {lead:.2f}')
    fig.text(0.5, 0.005, note, ha='center', color=_MUT, fontsize=8)
    plt.tight_layout(rect=[0, 0.02, 1, 0.98])

    p = outdir / f'{model}_timeseries.png'
    plt.savefig(p, dpi=150, bbox_inches='tight', facecolor=_BG)
    plt.close()
    print(f'[out] {p}')

    # J(x,y) snapshot
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor(_BG);  ax.set_facecolor(_BG)
    lim = max(abs(j_final.min()), abs(j_final.max()))
    im  = ax.imshow(j_final.T, origin='lower', cmap='RdBu_r',
                    vmin=-lim, vmax=lim,
                    extent=[0, 2*np.pi, 0, 2*np.pi], aspect='equal')
    cb  = fig.colorbar(im, ax=ax)
    cb.ax.yaxis.set_tick_params(color=_MUT)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=_MUT)
    ax.set_title(f'J(x,y) - final state - {model}', color=_TEXT)
    ax.set_xlabel('x', color=_MUT);  ax.set_ylabel('y', color=_MUT)
    ax.tick_params(colors=_MUT)
    for sp in ax.spines.values(): sp.set_edgecolor('#2a2a2a')

    p = outdir / f'{model}_J_final.png'
    plt.savefig(p, dpi=150, bbox_inches='tight', facecolor=_BG)
    plt.close()
    print(f'[out] {p}')

    # time-series CSV
    p = outdir / f'{model}_timeseries.csv'
    with open(p, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['t', 'E_B', 'E_K', 'j2', 'delta_j', 'phi', 'div_B'])
        for row in zip(t, E_B, E_K, j2, delta_j, phi, div_B):
            w.writerow([f'{v:.8e}' for v in row])
    print(f'[out] {p}')

    # J(x,y) CSV
    p = outdir / f'{model}_J_final.csv'
    np.savetxt(p, j_final, delimiter=',', fmt='%.8e')
    print(f'[out] {p}')

    print(f'[phi-mhd] {model}: Phi crossing t={t[phi_idx]:.2f} | '
          f'onset t={t[onset_idx]:.2f} | lead={lead:.2f}')
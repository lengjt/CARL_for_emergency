#!/usr/bin/env python3
"""Generate a shareable physics-check figure from saved measurements."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/".cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory",type=Path)
    directory=p.parse_args().directory
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    for scale in (1.0,.5):
        a=np.genfromtxt(directory/f"brake_{scale}.csv",delimiter=",",names=True)
        axes[0,0].plot(a['x_m'],a['v_x_mps'],label=f"scale={scale}")
    axes[0,0].set(xlabel="distance [m]",ylabel="vx [m/s]",title="20 m/s braking, -8000 N request")
    calibration=json.loads((directory/"tire_calibration.json").read_text())
    for c in calibration['curves']:
        if c['fz_N']!=2500:continue
        axes[0,1].plot(np.linspace(-.3,.3,301),np.array(c['fx_N'])/2500,label=f"scale={c['scale']}")
    axes[0,1].set(xlabel="longitudinal slip",ylabel="Fx / Fz",title="MF52 pure longitudinal slip, Fz=2500 N")
    for name in ('turn_left','turn_right'):
        a=np.genfromtxt(directory/f'{name}.csv',delimiter=',',names=True)
        axes[1,0].plot(a['x_m'],a['y_m'],label=name)
    axes[1,0].set(xlabel="x [m]",ylabel="y [m]",title="Mirrored steering commands")
    a=np.genfromtxt(directory/'patch_entry.csv',delimiter=',',names=True)
    for w in ('fl','rl'):
        axes[1,1].plot(a['t'],a['scale_'+w],label=w)
    axes[1,1].set(xlabel="time [s]",ylabel="tire-force scale",title="Front/rear entry into road patch")
    for ax in axes.flat:ax.grid(alpha=.2);ax.legend()
    fig.suptitle("Open Car Dynamics | synthetic research vehicle | engineering sanity checks")
    fig.tight_layout()
    fig.savefig(directory/'physics_checks.png',dpi=160)
    fig.savefig(directory/'physics_checks.pdf')
    plt.close(fig)

if __name__=='__main__':main()

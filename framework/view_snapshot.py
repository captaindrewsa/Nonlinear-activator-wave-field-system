import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) != 2:
    raise SystemExit("Usage: python view_snapshot.py path/to/snapshot.npz")

path = Path(sys.argv[1])

with np.load(path) as z:
    phi = z["phi"]
    psi = z["psi"] if "psi" in z else None
    v = z["v"] if "v" in z else None
    t = float(np.ravel(z["t"])[0]) if "t" in z else float("nan")

fields = [("phi", phi), ("psi", psi), ("v", v)]
fields = [(name, arr) for name, arr in fields if arr is not None]

fig, axes = plt.subplots(1, len(fields), figsize=(5 * len(fields), 4))
if len(fields) == 1:
    axes = [axes]

for ax, (name, arr) in zip(axes, fields):
    im = ax.imshow(arr.T, origin="lower", cmap="coolwarm")
    ax.set_title(f"{name}, t={t:.6g}")
    ax.set_xlabel("grid y")
    ax.set_ylabel("grid x")
    fig.colorbar(im, ax=ax, shrink=0.85)

fig.tight_layout()
out = path.with_suffix(".png")
fig.savefig(out, dpi=180)
print(out)
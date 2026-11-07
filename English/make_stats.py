import json
import numpy as np
from pathlib import Path

root=Path("English/preprocessed")
pitch=[]; energy=[]

for p in (root/"pitch").glob("*.npy"):
    x=np.load(p)
    x=x[np.isfinite(x)]
    if len(x): pitch.extend(x.tolist())

for p in (root/"energy").glob("*.npy"):
    x=np.load(p)
    x=x[np.isfinite(x)]
    if len(x): energy.extend(x.tolist())

stats={
    "pitch": [float(np.min(pitch)), float(np.max(pitch))],
    "energy": [float(np.min(energy)), float(np.max(energy))]
}

out=Path("English/preprocessed_data/stats.json")
out.write_text(json.dumps(stats,indent=2),encoding="utf-8")
print(json.dumps(stats,indent=2))

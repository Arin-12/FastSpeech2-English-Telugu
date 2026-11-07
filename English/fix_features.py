from pathlib import Path

p = Path("English/prepare_features.py")
s = p.read_text(encoding="utf-8")

s = s.replace(
'''        f0 = fit_length(np.log(np.maximum(f0, 1.0)), frames)
        voiced = (fit_length((f0 > 0).astype(np.float32), frames) > 0)''',
'''        voiced = fit_length((f0 > 0).astype(np.float32), frames) > 0.5
        f0 = fit_length(np.log(np.maximum(f0, 1.0)), frames)'''
)

s = s.replace(
'''voiced_values = np.concatenate([
    p[v] for p, v in pitch_arrays.values() if np.any(v)
])''',
'''voiced_parts = [p[v] for p, v in pitch_arrays.values() if np.any(v)]
if not records:
    raise RuntimeError("No samples prepared. Check mel/audio paths and alignment IDs.")
if not voiced_parts:
    voiced_parts = [p for p, v in pitch_arrays.values()]
voiced_values = np.concatenate(voiced_parts)'''
)

p.write_text(s, encoding="utf-8")
print("Fixed pitch mask and empty-voiced fallback.")

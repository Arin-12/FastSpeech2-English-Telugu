import json, time
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import pyworld as pw
from pathlib import Path

ROOT = Path.cwd()
ENG = ROOT / "English"
OUT = ENG / "preprocessed"
PITCH = OUT / "pitch"
ENERGY = OUT / "energy"
PITCH.mkdir(exist_ok=True)
ENERGY.mkdir(exist_ok=True)

raw = pd.read_csv(ENG / "data" / "train.csv")
prep = pd.read_csv(OUT / "train.csv")
df = raw.merge(prep[["id", "normalized_text"]], on="id")
align = {}

with open(ENG / "alignments.jsonl", encoding="utf-8") as f:
    for line in f:
        a = json.loads(line)
        align[a["id"]] = a

def allocate(weights, n):
    weights = np.maximum(np.asarray(weights, dtype=float), 1)
    if n < len(weights):
        raise ValueError("Not enough mel frames")
    extra = n - len(weights)
    exact = weights / weights.sum() * extra
    result = np.floor(exact).astype(int) + 1
    left = n - result.sum()
    if left:
        result[np.argsort(exact - np.floor(exact))[::-1][:left]] += 1
    return result.tolist()

records, pitches, energies, voiced_parts = [], {}, {}, []
start = time.time()

for _, r in df.iterrows():
    sid = str(r["id"])
    if sid not in align:
        continue
    try:
        mel_path = OUT / "mel" / f"{sid}.npy"
        wav_path = ENG / "data" / str(r["audio_path"])
        mel = np.load(mel_path)
        frames = mel.shape[1] if mel.shape[0] == 80 else mel.shape[0]

        a = align[sid]
        durations = allocate(a["char_durations"], frames)

        audio, sr = sf.read(str(wav_path), dtype="float64")
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if sr != 22050:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=22050)

        f0, t = pw.dio(audio, 22050, frame_period=1000*256/22050)
        f0 = pw.stonemask(audio, f0, t, 22050)
        v = f0 > 0
        logf0 = np.log(np.maximum(f0, 1.0))
        logf0 = np.interp(np.linspace(0, max(0, len(logf0)-1), frames),
                          np.arange(len(logf0)), logf0)
        v = np.interp(np.linspace(0, max(0, len(v)-1), frames),
                      np.arange(len(v)), v.astype(float)) > 0.5

        rms = librosa.feature.rms(y=audio.astype(np.float32),
                                  frame_length=1024, hop_length=256)[0]
        energy = np.log(np.maximum(np.interp(
            np.linspace(0, max(0, len(rms)-1), frames),
            np.arange(len(rms)), rms), 1e-6))

        pitches[sid] = (logf0.astype(np.float32), v)
        energies[sid] = energy.astype(np.float32)
        if v.any():
            voiced_parts.append(logf0[v])

        records.append({
            "id": sid, "text": a["text"], "durations": durations,
            "mel_path": f"English/preprocessed/mel/{sid}.npy",
            "frames": frames
        })

    except Exception as e:
        print("Skip", sid, repr(e)[:120])

    if len(records) % 50 == 0 and records:
        print("Prepared", len(records), "samples")

if not records:
    raise RuntimeError("No features created; check exact file names.")

pv = np.concatenate(voiced_parts) if voiced_parts else np.concatenate(
    [x[0] for x in pitches.values()])
ev = np.concatenate(list(energies.values()))
pm, ps = float(pv.mean()), float(pv.std()+1e-6)
em, es = float(ev.mean()), float(ev.std()+1e-6)

for r in records:
    sid = r["id"]
    p, v = pitches[sid]
    p = np.where(v, (p-pm)/ps, 0).astype(np.float32)
    e = ((energies[sid]-em)/es).astype(np.float32)
    np.save(PITCH / f"{sid}.npy", p)
    np.save(ENERGY / f"{sid}.npy", e)
    r["pitch_path"] = f"English/preprocessed/pitch/{sid}.npy"
    r["energy_path"] = f"English/preprocessed/energy/{sid}.npy"

with open(OUT / "english_manifest.jsonl", "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r) + "\n")

with open(OUT / "english_feature_stats.json", "w") as f:
    json.dump({"pitch_mean":pm,"pitch_std":ps,
               "energy_mean":em,"energy_std":es}, f, indent=2)

print("COMPLETE | Prepared:",len(records),
      "| Seconds:",round(time.time()-start,1))

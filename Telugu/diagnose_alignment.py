import json
import numpy as np
import soundfile as sf
import onnxruntime as ort
from pathlib import Path

root = Path("Telugu/aligner")
vocab = json.loads((root / "vocab.json").read_text(encoding="utf-8"))
config = json.loads((root / "config.json").read_text(encoding="utf-8"))
reverse = {v:k for k,v in vocab.items()}

audio, sr = sf.read("Telugu/data/wavs/telugu_000000.wav", dtype="float32")
if audio.ndim == 2:
    audio = audio.mean(axis=1)

session = ort.InferenceSession(
    str(root / "onnx/model_int8.onnx"),
    providers=["CPUExecutionProvider"]
)
logits = session.run(["logits"], {"input_values": audio[None,:]})[0][0]
ids = logits.argmax(axis=-1)

# Compress consecutive CTC predictions.
segments = []
start = 0
for i in range(1, len(ids)+1):
    if i == len(ids) or ids[i] != ids[start]:
        token = int(ids[start])
        segments.append((
            reverse.get(token, "[UNK]"),
            start, i, round((i-start)*0.02, 3)
        ))
        start = i

print("Audio duration:", round(len(audio)/sr, 3))
print("CTC prediction segments:")
for token, start, end, duration in segments:
    print(f"{start:3d}-{end:3d} | {token!r:8s} | {duration:.3f}s")

print("\nWaveform RMS in 0.2-second windows:")
for start in np.arange(0, len(audio)/sr, 0.2):
    a = int(start*sr)
    b = min(len(audio), int((start+0.2)*sr))
    rms = float(np.sqrt(np.mean(audio[a:b]**2))) if b>a else 0
    print(f"{start:.1f}-{start+0.2:.1f}s: {rms:.5f}")

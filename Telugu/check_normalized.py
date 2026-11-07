import json
import numpy as np
import soundfile as sf
import onnxruntime as ort
from pathlib import Path

root = Path("Telugu/aligner")
vocab = json.loads((root / "vocab.json").read_text(encoding="utf-8"))
reverse = {v:k for k,v in vocab.items()}
config = json.loads((root / "config.json").read_text(encoding="utf-8"))

print("Feature extraction normalization:", config.get("feat_extract_norm"))

audio, sr = sf.read("Telugu/data/wavs/telugu_000000.wav", dtype="float32")
if audio.ndim == 2:
    audio = audio.mean(axis=1)

audio = (audio - audio.mean()) / max(audio.std(), 1e-7)

session = ort.InferenceSession(
    str(root / "onnx/model_int8.onnx"),
    providers=["CPUExecutionProvider"]
)

logits = session.run(
    ["logits"],
    {"input_values": audio[None, :].astype(np.float32)}
)[0][0]

ids = logits.argmax(axis=-1)
decoded = []
previous = None

for idx in ids:
    idx = int(idx)
    if idx != previous and idx != 65:
        decoded.append(reverse.get(idx, "[UNK]"))
    previous = idx

print("Expected:", "నరకవాసి")
print("Normalized prediction:", "".join(decoded).replace("|", " "))
print("Acoustic frames:", len(ids))

segments = []
start = 0
for i in range(1, len(ids)+1):
    if i == len(ids) or ids[i] != ids[start]:
        token = reverse.get(int(ids[start]), "[UNK]")
        segments.append((token, start, i))
        start = i

print("\nNormalized CTC segments:")
for token, a, b in segments:
    print(f"{a:3d}-{b:3d} | {token!r} | {(b-a)*0.02:.2f}s")

import json
import numpy as np
import soundfile as sf
import onnxruntime as ort
from pathlib import Path

root = Path("Telugu/aligner")

with open(root / "vocab.json", encoding="utf-8") as f:
    vocab = json.load(f)

reverse = {v: k for k, v in vocab.items()}

audio, sr = sf.read(
    "Telugu/data/wavs/telugu_000000.wav",
    dtype="float32"
)

if audio.ndim == 2:
    audio = audio.mean(axis=1)

session = ort.InferenceSession(
    str(root / "onnx/model_int8.onnx"),
    providers=["CPUExecutionProvider"]
)

logits = session.run(
    ["logits"],
    {"input_values": audio[None, :].astype(np.float32)}
)[0][0]

ids = np.argmax(logits, axis=-1)

# CTC greedy decoding.
decoded = []
previous = None

for idx in ids:
    idx = int(idx)
    if idx != previous and idx != 65:
        decoded.append(reverse.get(idx, "[UNK]"))
    previous = idx

print("Expected:", "నరకవాసి")
print("Model prediction:", "".join(decoded).replace("|", " "))
print("Acoustic frames:", len(ids))
print("Audio duration:", round(len(audio) / sr, 3), "seconds")

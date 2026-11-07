import csv
import json
import numpy as np
import soundfile as sf
import onnxruntime as ort
from pathlib import Path

ROOT = Path("Telugu")
with open(ROOT / "aligner/vocab.json", encoding="utf-8") as f:
    vocab = json.load(f)
with open(ROOT / "aligner/config.json", encoding="utf-8") as f:
    config = json.load(f)

reverse = {v: k for k, v in vocab.items()}
blank = config["pad_token_id"]

with open(ROOT / "data/metadata.csv", encoding="utf-8-sig") as f:
    row = next(csv.DictReader(f))

text = row["text"].strip()
clean = "".join(c for c in text if c not in ".,!?;:।")
clean = "|".join(clean.split())
tokens = list(clean)

if any(c not in vocab for c in tokens):
    raise ValueError("Unsupported character in transcript")

target = [vocab[c] for c in tokens]

audio, sr = sf.read(row["audio_filepath"], dtype="float32")
if audio.ndim == 2:
    audio = audio.mean(axis=1)
if sr != 16000:
    raise ValueError("Audio must be 16 kHz")

session = ort.InferenceSession(
    str(ROOT / "aligner/onnx/model_int8.onnx"),
    providers=["CPUExecutionProvider"]
)
logits = session.run(
    ["logits"],
    {"input_values": audio[None, :].astype(np.float32)}
)[0][0]

logits = logits.astype(np.float64)
logp = logits - np.logaddexp.reduce(logits, axis=-1, keepdims=True)

T = len(logp)
extended = [blank]
for token in target:
    extended.extend([token, blank])

S = len(extended)
dp = np.full((T, S), -np.inf)
back = np.full((T, S), -1, dtype=np.int32)

dp[0, 0] = logp[0, blank]
if S > 1:
    dp[0, 1] = logp[0, extended[1]]

for t in range(1, T):
    for s in range(S):
        candidates = [(dp[t-1, s], s)]
        if s >= 1:
            candidates.append((dp[t-1, s-1], s-1))
        if s >= 2 and extended[s] != blank and extended[s] != extended[s-2]:
            candidates.append((dp[t-1, s-2], s-2))

        score, prev = max(candidates, key=lambda x: x[0])
        dp[t, s] = score + logp[t, extended[s]]
        back[t, s] = prev

state = S - 1
if S > 1 and dp[-1, S-2] > dp[-1, S-1]:
    state = S - 2

if not np.isfinite(dp[-1, state]):
    raise RuntimeError("No valid alignment")

path = np.zeros(T, dtype=np.int32)
for t in range(T-1, -1, -1):
    path[t] = state
    if t:
        state = int(back[t, state])

# Assign every acoustic frame to a character.
assigned = np.zeros(T, dtype=np.int32)
i = 0

while i < T:
    s = int(path[i])
    j = i + 1
    while j < T and path[j] == s:
        j += 1

    if s % 2 == 1:
        # Character state.
        char_index = s // 2
        assigned[i:j] = char_index
    else:
        # Blank state: distribute between neighboring characters.
        left = s // 2 - 1
        right = s // 2

        if left < 0:
            assigned[i:j] = 0
        elif right >= len(tokens):
            assigned[i:j] = len(tokens) - 1
        else:
            midpoint = (i + j) // 2
            assigned[i:midpoint] = left
            assigned[midpoint:j] = right

    i = j

counts = np.bincount(assigned, minlength=len(tokens))

if counts.sum() != T or np.any(counts == 0):
    raise RuntimeError(f"Invalid durations: {counts.tolist()}")

stride = int(np.prod(config.get("conv_stride", [5,2,2,2,2,2,2])))
total_seconds = T * stride / sr

print("Transcript:", text)
print("Alignment text:", clean)
print("Audio duration:", round(len(audio)/sr, 3), "seconds")
print("Acoustic frames:", T)
print("Wav2Vec2 frame stride:", stride, "samples")
print("\nCharacter durations:")

for char, count in zip(tokens, counts):
    print(f"{char} : {count} acoustic frames")

print("\nDuration sum:", int(counts.sum()))
print("Aligned acoustic duration:", round(total_seconds, 3), "seconds")
print("Duration coverage:", round(100*counts.sum()/T, 2), "%")

result = {
    "text": text,
    "alignment_text": clean,
    "tokens": tokens,
    "acoustic_durations": counts.tolist(),
    "acoustic_frames": int(T),
    "frame_stride_samples": stride,
    "sample_rate": sr
}

out = ROOT / "alignment_test" / "telugu_000000_alignment_v2.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print("\nSaved:", out)

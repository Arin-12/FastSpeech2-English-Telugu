import json
import numpy as np
import soundfile as sf
import onnxruntime as ort

root = "Telugu/aligner"
vocab = json.load(open(root + "/vocab.json", encoding="utf-8"))
blank = 65

text = "నరకవాసి"
tokens = list(text)
target = [vocab[c] for c in tokens]

audio, sr = sf.read(
    "Telugu/data/wavs/telugu_000000.wav",
    dtype="float32"
)
if audio.ndim == 2:
    audio = audio.mean(axis=1)

audio = (audio - audio.mean()) / max(audio.std(), 1e-7)

session = ort.InferenceSession(
    root + "/onnx/model_int8.onnx",
    providers=["CPUExecutionProvider"]
)
logits = session.run(
    ["logits"],
    {"input_values": audio[None, :]}
)[0][0]

# Numerically stable log-softmax
x = logits - np.max(logits, axis=-1, keepdims=True)
logp = x - np.log(np.exp(x).sum(axis=-1, keepdims=True))

# Interleave blank states and target characters
extended = [blank]
for token in target:
    extended.extend([token, blank])

T = len(logp)
S = len(extended)

dp = np.full((T, S), -np.inf, dtype=np.float64)
back = np.full((T, S), -1, dtype=np.int32)

dp[0, 0] = logp[0, blank]
if S > 1:
    dp[0, 1] = logp[0, extended[1]]

for t in range(1, T):
    for s in range(S):
        candidates = [(dp[t-1, s], s)]

        if s > 0:
            candidates.append((dp[t-1, s-1], s-1))

        if (
            s > 1
            and extended[s] != blank
            and extended[s] != extended[s-2]
        ):
            candidates.append((dp[t-1, s-2], s-2))

        best_score, previous = max(candidates, key=lambda z: z[0])
        dp[t, s] = best_score + logp[t, extended[s]]
        back[t, s] = previous

# End in the final character or final blank
state = S - 1 if dp[-1, S-1] > dp[-1, S-2] else S - 2

path = []
for t in range(T-1, -1, -1):
    path.append(state)
    if t > 0:
        state = back[t, state]
path = path[::-1]

# Character state boundaries; split internal blank gaps at midpoints
spans = []
for i, char in enumerate(tokens):
    state_id = 2*i + 1
    frames = [t for t, s in enumerate(path) if s == state_id]
    if not frames:
        raise RuntimeError(f"No alignment found for character: {char}")
    spans.append([frames[0], frames[-1] + 1])

print("Forced alignment:")
total = 0
for i, (char, (start, end)) in enumerate(zip(tokens, spans)):
    if i > 0:
        previous_end = spans[i-1][1]
        boundary = (previous_end + start) // 2
        spans[i-1][1] = boundary
        start = boundary
        spans[i][0] = boundary

for char, (start, end) in zip(tokens, spans):
    duration = (end-start) * 0.02
    total += duration
    print(f"{char} : {start*0.02:.3f}s - {end*0.02:.3f}s = {duration:.3f}s")

print(f"\nAligned character duration: {total:.3f}s")
print(f"Audio duration: {len(audio)/sr:.3f}s")
print(f"Viterbi score: {max(dp[-1, -1], dp[-1, -2]):.2f}")

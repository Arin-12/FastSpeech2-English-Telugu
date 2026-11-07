import csv, json, unicodedata
import numpy as np
import soundfile as sf
import onnxruntime as ort
from pathlib import Path

root = Path("Telugu/aligner")
vocab = json.loads((root / "vocab.json").read_text(encoding="utf-8"))
blank = 65

session = ort.InferenceSession(
    str(root / "onnx/model_int8.onnx"),
    providers=["CPUExecutionProvider"]
)

def clean_text(text):
    text = unicodedata.normalize("NFC", text)
    text = text.replace(",", "").replace(".", "")
    return " ".join(text.split()).strip()

def align(audio, text):
    text = clean_text(text)
    chars = list(text)
    tokens = [vocab["|" if c.isspace() else c] for c in chars]

    audio = (audio - audio.mean()) / max(audio.std(), 1e-7)
    logits = session.run(
        ["logits"],
        {"input_values": audio[None, :].astype(np.float32)}
    )[0][0]

    x = logits - logits.max(axis=-1, keepdims=True)
    logp = x - np.log(np.exp(x).sum(axis=-1, keepdims=True))

    extended = [blank]
    for token in tokens:
        extended.extend([token, blank])

    T, S = len(logp), len(extended)
    dp = np.full((T, S), -np.inf)
    back = np.full((T, S), -1, dtype=np.int32)

    dp[0, 0] = logp[0, blank]
    if S > 1:
        dp[0, 1] = logp[0, extended[1]]

    for t in range(1, T):
        for s in range(S):
            choices = [(dp[t-1, s], s)]
            if s > 0:
                choices.append((dp[t-1, s-1], s-1))
            if s > 1 and extended[s] != blank and extended[s] != extended[s-2]:
                choices.append((dp[t-1, s-2], s-2))
            score, prev = max(choices, key=lambda z: z[0])
            dp[t, s] = score + logp[t, extended[s]]
            back[t, s] = prev

    state = S-1 if dp[-1, S-1] > dp[-1, S-2] else S-2
    score = dp[-1, state]
    path = []

    for t in range(T-1, -1, -1):
        path.append(state)
        if t:
            state = back[t, state]
    path.reverse()

    spans = []
    for i, char in enumerate(chars):
        frames = [t for t, s in enumerate(path) if s == 2*i+1]
        if not frames:
            raise ValueError(f"Missing character: {char!r}")
        spans.append([frames[0], frames[-1]+1])

    for i in range(1, len(spans)):
        boundary = (spans[i-1][1] + spans[i][0]) // 2
        spans[i-1][1] = boundary
        spans[i][0] = boundary

    durations = [(b-a)*0.02 for a, b in spans]
    return score, durations

rows = list(csv.DictReader(
    open("Telugu/data/metadata.csv", encoding="utf-8-sig")
))

print("Testing first 10 recordings\n")
for row in rows[:10]:
    name = Path(row["audio_filepath"]).name
    text = clean_text(row["text"])

    unsupported = [
        c for c in text
        if (("|" if c.isspace() else c) not in vocab)
    ]
    if unsupported:
        print(name, "SKIPPED:", repr(sorted(set(unsupported))))
        continue

    audio, sr = sf.read(row["audio_filepath"], dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    try:
        score, durations = align(audio, text)
        print(
            f"{name} | chars={len(durations)} | "
            f"aligned={sum(durations):.2f}s | "
            f"audio={len(audio)/sr:.2f}s | "
            f"score={score:.1f} | "
            f"min={min(durations):.2f}s | max={max(durations):.2f}s"
        )
    except Exception as e:
        print(name, "FAILED:", repr(e))

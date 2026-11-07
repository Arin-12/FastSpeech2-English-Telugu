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

def normalize(text):
    text = unicodedata.normalize("NFC", text)
    text = text.replace(",", "").replace(".", "")
    return " ".join(text.split()).strip()

def align(audio, text):
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
    score = float(dp[-1, state])
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
            raise ValueError(f"Missing character {char!r}")
        spans.append([frames[0], frames[-1]+1])

    for i in range(1, len(spans)):
        boundary = (spans[i-1][1] + spans[i][0]) // 2
        spans[i-1][1] = boundary
        spans[i][0] = boundary

    return [
        {
            "character": char,
            "start_frame": a,
            "end_frame": b,
            "duration_frames": b-a,
            "duration_seconds": round((b-a)*0.02, 3)
        }
        for char, (a, b) in zip(chars, spans)
    ], score

rows = list(csv.DictReader(
    open("Telugu/data/metadata.csv", encoding="utf-8-sig")
))

output = Path("Telugu/data/alignments.jsonl")
skipped = []
success = 0

with output.open("w", encoding="utf-8") as f:
    for i, row in enumerate(rows):
        name = Path(row["audio_filepath"]).name
        text = normalize(row["text"])

        unsupported = [
            c for c in text
            if ("|" if c.isspace() else c) not in vocab
        ]

        if unsupported:
            skipped.append({
                "file": name,
                "unsupported": sorted(set(unsupported))
            })
            print(f"SKIP {name}: {unsupported}")
            continue

        try:
            audio, sr = sf.read(row["audio_filepath"], dtype="float32")
            if audio.ndim == 2:
                audio = audio.mean(axis=1)

            aligned, score = align(audio, text)

            record = {
                "audio_filepath": row["audio_filepath"],
                "text": text,
                "sample_rate": sr,
                "audio_duration": round(len(audio)/sr, 3),
                "alignment_score": round(score, 3),
                "characters": aligned
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            success += 1
            print(f"[{i+1}/{len(rows)}] OK {name}")

        except Exception as e:
            skipped.append({"file": name, "error": str(e)})
            print(f"FAILED {name}: {e}")

Path("Telugu/data/alignment_skipped.json").write_text(
    json.dumps(skipped, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

print("\nALIGNMENT SUMMARY")
print("Total recordings:", len(rows))
print("Successful:", success)
print("Skipped/failed:", len(skipped))
print("Output:", output)

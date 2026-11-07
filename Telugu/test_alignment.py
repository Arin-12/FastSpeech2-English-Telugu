import csv
import json
import numpy as np
import soundfile as sf
import onnxruntime as ort
import unicodedata
from pathlib import Path

ROOT = Path("Telugu")
META = ROOT / "data" / "metadata.csv"
MODEL = ROOT / "aligner" / "onnx" / "model_int8.onnx"
VOCAB_FILE = ROOT / "aligner" / "vocab.json"
CONFIG_FILE = ROOT / "aligner" / "config.json"

with open(VOCAB_FILE, encoding="utf-8") as f:
    vocab = json.load(f)

with open(CONFIG_FILE, encoding="utf-8") as f:
    config = json.load(f)

BLANK = config["pad_token_id"]
SR = 16000
HOP_LENGTH = 256

with open(META, encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

# First recording is used for this initial test.
row = rows[0]
audio_path = Path(row["audio_filepath"])
transcript = unicodedata.normalize("NFC", row["text"])

# Remove punctuation and convert spaces to the CTC word separator.
punctuation = ".,!?;:।"
clean_text = "".join(c for c in transcript if c not in punctuation)
clean_text = "|".join(clean_text.split())

tokens = list(clean_text)

missing = [c for c in tokens if c not in vocab]
if missing:
    raise ValueError(f"Unsupported transcript characters: {missing}")

target = [vocab[c] for c in tokens]

print("Audio:", audio_path)
print("Original transcript:", transcript)
print("Alignment transcript:", clean_text)
print("Characters:", len(tokens))

audio, sample_rate = sf.read(audio_path, dtype="float32")

if audio.ndim == 2:
    audio = audio.mean(axis=1)

if sample_rate != SR:
    raise ValueError(f"Expected {SR} Hz audio, got {sample_rate} Hz")

audio = np.ascontiguousarray(audio[None, :], dtype=np.float32)

print("\nLoading ONNX model...", flush=True)
session = ort.InferenceSession(
    str(MODEL),
    providers=["CPUExecutionProvider"]
)

print("Running inference...", flush=True)
logits = session.run(
    ["logits"],
    {"input_values": audio}
)[0][0]

# Stable log-softmax.
logits = logits.astype(np.float64)
log_probs = logits - np.logaddexp.reduce(logits, axis=-1, keepdims=True)

T, V = log_probs.shape
print("Acoustic output frames:", T)

# CTC expanded target: blank, char, blank, char, ..., blank.
extended = [BLANK]
for token in target:
    extended.extend([token, BLANK])

S = len(extended)

dp = np.full((T, S), -np.inf, dtype=np.float64)
back = np.full((T, S), -1, dtype=np.int8)

dp[0, 0] = log_probs[0, BLANK]
if S > 1:
    dp[0, 1] = log_probs[0, extended[1]]

# Viterbi dynamic programming.
for t in range(1, T):
    for s in range(S):
        best_score = dp[t - 1, s]
        best_prev = s

        if s >= 1 and dp[t - 1, s - 1] > best_score:
            best_score = dp[t - 1, s - 1]
            best_prev = s - 1

        if (
            s >= 2
            and extended[s] != BLANK
            and extended[s] != extended[s - 2]
            and dp[t - 1, s - 2] > best_score
        ):
            best_score = dp[t - 1, s - 2]
            best_prev = s - 2

        dp[t, s] = best_score + log_probs[t, extended[s]]
        back[t, s] = best_prev

# Select the best final CTC state.
final_state = S - 1
if S > 1 and dp[T - 1, S - 2] > dp[T - 1, S - 1]:
    final_state = S - 2

if not np.isfinite(dp[T - 1, final_state]):
    raise RuntimeError("No valid CTC alignment found.")

# Backtrack through the best path.
path = np.zeros(T, dtype=np.int32)
state = final_state

for t in range(T - 1, -1, -1):
    path[t] = state
    if t > 0:
        state = int(back[t, state])
        if state < 0:
            raise RuntimeError("Alignment backtracking failed.")

# Convert character spans into time and FastSpeech mel-frame durations.
# Wav2Vec2 frame stride is the product of the convolution strides.
conv_stride = config.get("conv_stride", [5, 2, 2, 2, 2, 2, 2])
frame_stride_samples = int(np.prod(conv_stride))

alignment = []
durations = []

for i, char in enumerate(tokens):
    state_id = 2 * i + 1
    positions = np.where(path == state_id)[0]

    if len(positions) == 0:
        raise RuntimeError(
            f"Character {char!r} received zero alignment frames."
        )

    start_frame = int(positions[0])
    end_frame = int(positions[-1]) + 1

    start_seconds = start_frame * frame_stride_samples / SR
    end_seconds = end_frame * frame_stride_samples / SR

    duration_seconds = end_seconds - start_seconds
    mel_duration = max(
        1,
        round(duration_seconds * SR / HOP_LENGTH)
    )

    durations.append(mel_duration)
    alignment.append({
        "character": char,
        "start_seconds": round(start_seconds, 4),
        "end_seconds": round(end_seconds, 4),
        "duration_seconds": round(duration_seconds, 4),
        "mel_frames": mel_duration
    })

output = {
    "audio_filepath": str(audio_path),
    "original_text": transcript,
    "alignment_text": clean_text,
    "sample_rate": SR,
    "wav2vec_frame_stride_samples": frame_stride_samples,
    "mel_hop_length": HOP_LENGTH,
    "characters": alignment,
    "durations": durations
}

out_dir = ROOT / "alignment_test"
out_dir.mkdir(exist_ok=True)

out_file = out_dir / "telugu_000000_alignment.json"
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print("\nALIGNMENT RESULTS")
for item in alignment:
    print(
        f"{item['character']} | "
        f"{item['start_seconds']:.3f}s - "
        f"{item['end_seconds']:.3f}s | "
        f"{item['mel_frames']} mel frames"
    )

print("\nTotal aligned characters:", len(alignment))
print("Total mel-frame durations:", sum(durations))
print("Saved:", out_file)
print("\nALIGNMENT TEST COMPLETED")

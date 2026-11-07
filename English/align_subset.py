import json
import time
import numpy as np
import pandas as pd
import torch
import torchaudio
import soundfile as sf
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "English" / "alignments.jsonl"
LIMIT = 300

torch.set_num_threads(4)
bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
labels = bundle.get_labels()
char_to_id = {c: i for i, c in enumerate(labels)}

def clean_text(text):
    text = str(text).upper()
    text = " ".join(text.split())
    return "".join(c if c in char_to_id else "" for c in text.replace(" ", "|"))

def viterbi(logp, target):
    T, C = logp.shape
    if len(target) == 0 or len(target) > T:
        raise ValueError("Invalid target length")

    # Blank-interleaved CTC state sequence
    states = np.zeros(2 * len(target) + 1, dtype=np.int32)
    states[1::2] = target
    S = len(states)

    dp = np.full(S, -1e30, dtype=np.float32)
    dp[0] = logp[0, 0]
    dp[1] = logp[0, states[1]]
    back = np.zeros((T, S), dtype=np.int8)

    skip_ok = np.zeros(S, dtype=bool)
    for s in range(2, S):
        skip_ok[s] = states[s] != 0 and states[s] != states[s - 2]

    for t in range(1, T):
        stay = dp
        advance = np.full(S, -1e30, dtype=np.float32)
        advance[1:] = dp[:-1]
        skip = np.full(S, -1e30, dtype=np.float32)
        skip[2:] = dp[:-2]
        skip[~skip_ok] = -1e30

        candidates = np.stack([stay, advance, skip], axis=0)
        choice = np.argmax(candidates, axis=0)
        best = candidates[choice, np.arange(S)]
        dp = best + logp[t, states]
        back[t] = choice

    end = S - 1 if dp[S - 1] >= dp[S - 2] else S - 2
    path = np.zeros(T, dtype=np.int32)
    path[-1] = end
    for t in range(T - 1, 0, -1):
        move = back[t, path[t]]
        path[t - 1] = path[t] - (1 if move == 1 else 2 if move == 2 else 0)

    durations = [int(np.sum(path == (2 * i + 1))) for i in range(len(target))]
    return durations

print("Loading cached wav2vec2...")
model = bundle.get_model().eval().to("cpu")

raw = pd.read_csv(ROOT / "English" / "data" / "train.csv")
prep = pd.read_csv(ROOT / "English" / "preprocessed" / "train.csv")
df = raw.merge(prep[["id", "normalized_text"]], on="id", how="inner")
df = df.head(LIMIT)

success, failed = 0, 0
start = time.time()

with OUT.open("w", encoding="utf-8") as f:
    for i, row in df.iterrows():
        try:
            text = clean_text(row["normalized_text"])
            target = [char_to_id[c] for c in text]
            if not target:
                raise ValueError("Empty normalized transcript")

            wav_path = ROOT / "English" / "data" / row["audio_path"]
            audio, sr = sf.read(str(wav_path), dtype="float32")
            if audio.ndim == 2:
                audio = audio.mean(axis=1)

            waveform = torch.from_numpy(audio).unsqueeze(0)
            if sr != bundle.sample_rate:
                waveform = torchaudio.functional.resample(
                    waveform, sr, bundle.sample_rate
                )

            with torch.inference_mode():
                emissions, _ = model(waveform)

            logp = emissions[0].cpu().numpy()
            durations = viterbi(logp, target)

            record = {
                "id": str(row["id"]),
                "text": text,
                "char_durations": durations,
                "aligner_frames": int(logp.shape[0]),
                "audio_duration": float(row["duration"])
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            success += 1

        except Exception as e:
            failed += 1
            print("Skipped:", row["id"], str(e)[:100])

        if (success + failed) % 25 == 0:
            print(f"Processed {success + failed}/{len(df)} | Success {success} | Failed {failed}")

print("\nALIGNMENT COMPLETE")
print("Success:", success, "| Failed:", failed)
print("Seconds:", round(time.time() - start, 1))
print("Saved:", OUT)

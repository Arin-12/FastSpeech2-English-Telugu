import torch
import torchaudio
import soundfile as sf
import numpy as np
import pandas as pd
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent.parent
bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H

print("Loading cached English wav2vec2 model...")
model = bundle.get_model().eval().to("cpu")

df = pd.read_csv(ROOT / "English" / "data" / "train.csv")
row = df.iloc[0]
audio_path = ROOT / "English" / "data" / row["audio_path"]

audio, sr = sf.read(str(audio_path), dtype="float32")
if audio.ndim == 2:
    audio = audio.mean(axis=1)

waveform = torch.from_numpy(audio).unsqueeze(0)

if sr != bundle.sample_rate:
    waveform = torchaudio.functional.resample(
        waveform, sr, bundle.sample_rate
    )

print("Audio:", row["id"], "| Duration:", round(len(audio) / sr, 2), "sec")
print("Running CPU alignment-model inference...")
start = time.time()

with torch.inference_mode():
    emissions, lengths = model(waveform)

print("SUCCESS")
print("Emission shape:", tuple(emissions.shape))
print("Output frames:", int(emissions.shape[1]))
print("Inference seconds:", round(time.time() - start, 2))
print("Transcript:", row["text"])

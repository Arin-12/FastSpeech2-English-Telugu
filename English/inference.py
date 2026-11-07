import os
import sys
import json
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FS = ROOT / "FastSpeech2-English"
OUTPUT = ROOT / "outputs" / "english"
CHECKPOINT = OUTPUT / "best_model.pth"
PRE = ROOT / "preprocessed_data"

sys.path.insert(0, str(FS))
os.chdir(FS)

from model.fastspeech2 import FastSpeech2
from text import text_to_sequence
from audio.stft import TacotronSTFT
from audio.tools import inv_mel_spec

tests = [
    {
        "id": "E1",
        "original": "The antidisestablishmentarianism movement gained considerable momentum during the parliamentary proceedings.",
        "input": "THE ANTIDISESTABLISHMENTARIANISM MOVEMENT GAINED CONSIDERABLE MOMENTUM DURING THE PARLIAMENTARY PROCEEDINGS"
    },
    {
        "id": "E2a",
        "original": "The orchestra's conduct during the ceremony was praised by the audience.",
        "input": "THE ORCHESTRA'S CONDUCT DURING THE CEREMONY WAS PRAISED BY THE AUDIENCE"
    },
    {
        "id": "E2b",
        "original": "The researchers will conduct an experiment to evaluate the proposed method.",
        "input": "THE RESEARCHERS WILL CONDUCT AN EXPERIMENT TO EVALUATE THE PROPOSED METHOD"
    },
    {
        "id": "E3",
        "original": "The rhyth m of the orchestra changed unexpectedly.",
        "input": "THE RHYTH M OF THE ORCHESTRA CHANGED UNEXPECTEDLY"
    },
    {
        "id": "E4",
        "original": "The meeting will start at 10:45 AM on 12 November 2026 so Andaroo should arrive on time.",
        "input": "THE MEETING WILL START AT TEN FORTY FIVE AM ON TWELVE NOVEMBER TWO THOUSAND TWENTY SIX SO ANDAROO SHOULD ARRIVE ON TIME"
    }
]

AUDIO_DIR = OUTPUT / "test_generated_audios"
RESULTS = ROOT.parent / "Results" / "English"
DURATION_DIR = RESULTS / "variance_adaptor_plots" / "duration"
PITCH_DIR = RESULTS / "variance_adaptor_plots" / "pitch"
ENERGY_DIR = RESULTS / "variance_adaptor_plots" / "energy"

for folder in [AUDIO_DIR, DURATION_DIR, PITCH_DIR, ENERGY_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

with open(FS / "config" / "LJSpeech" / "preprocess.yaml", encoding="utf-8") as f:
    preprocess_config = yaml.safe_load(f)

with open(FS / "config" / "LJSpeech" / "model.yaml", encoding="utf-8") as f:
    model_config = yaml.safe_load(f)

model_config["max_seq_len"] = 10000

device = torch.device("cpu")
model = FastSpeech2(preprocess_config, model_config).to(device)

checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=False)
model.load_state_dict(checkpoint["model"], strict=True)
model.eval()

audio_cfg = preprocess_config["preprocessing"]["audio"]
stft_cfg = preprocess_config["preprocessing"]["stft"]
mel_cfg = preprocess_config["preprocessing"]["mel"]

stft = TacotronSTFT(
    stft_cfg["filter_length"],
    stft_cfg["hop_length"],
    stft_cfg["win_length"],
    mel_cfg["n_mel_channels"],
    audio_cfg["sampling_rate"],
    mel_cfg["mel_fmin"],
    mel_cfg["mel_fmax"]
).cpu().eval()

stats_path = PRE / "stats.json"
stats = json.loads(stats_path.read_text(encoding="utf-8"))

def get_stat(keys):
    def search(obj):
        if isinstance(obj, dict):
            for key in keys:
                if key in obj:
                    return float(obj[key])
            for value in obj.values():
                result = search(value)
                if result is not None:
                    return result
        return None
    return search(stats)

p_mean = get_stat(["pitch_mean", "p_mean"])
p_std = get_stat(["pitch_std", "p_std"])
e_mean = get_stat(["energy_mean", "e_mean"])
e_std = get_stat(["energy_std", "e_std"])

text_config = preprocess_config.get("preprocessing", {}).get("text", {})
cleaners = text_config.get("text_cleaners", [])

metadata = []

print("Loaded English checkpoint, epoch:", checkpoint.get("epoch"))
print("Device:", device)
print("Generating five English samples...")

for item in tests:
    sid = item["id"]
    text = item["input"].replace(" ", "|")

    try:
        ids = text_to_sequence(text, cleaners)
    except TypeError:
        ids = text_to_sequence(text)

    if not ids:
        raise ValueError(f"{sid}: Empty token sequence")

    texts = torch.LongTensor([ids]).to(device)
    src_lens = torch.LongTensor([len(ids)]).to(device)
    speakers = torch.LongTensor([0]).to(device)

    with torch.no_grad():
        predictions = model(
            speakers,
            texts,
            src_lens,
            len(ids),
            d_control=1.0
        )

    mel = predictions[1][0]
    pitch = predictions[2][0].detach().cpu().numpy()
    energy = predictions[3][0].detach().cpu().numpy()
    durations = predictions[5][0].detach().cpu().numpy()

    mel_len = int(predictions[9][0].item())
    mel = mel[:mel_len].detach().cpu()

    if mel_len <= 0:
        raise RuntimeError(f"{sid}: Model predicted an empty mel sequence")

    wav_path = AUDIO_DIR / f"{sid}.wav"

    inv_mel_spec(
        mel.transpose(0, 1).contiguous(),
        str(wav_path),
        stft,
        griffin_iters=12
    )

    # Duration plot
    plt.figure(figsize=(14, 4))
    plt.bar(range(len(durations[:len(ids)])), durations[:len(ids)])
    plt.xlabel("Input token index")
    plt.ylabel("Predicted duration (mel frames)")
    plt.title(f"{sid} - Predicted Duration")
    plt.tight_layout()
    plt.savefig(DURATION_DIR / f"{sid}_duration.png", dpi=150)
    plt.close()

    # Pitch plot
    pitch_plot = pitch * p_std + p_mean if p_mean is not None and p_std is not None else pitch

    plt.figure(figsize=(12, 4))
    plt.plot(pitch_plot[:mel_len])
    plt.xlabel("Mel frame")
    plt.ylabel("Predicted F0 / normalized pitch")
    plt.title(f"{sid} - Predicted Pitch")
    plt.tight_layout()
    plt.savefig(PITCH_DIR / f"{sid}_pitch.png", dpi=150)
    plt.close()

    # Energy plot
    energy_plot = energy * e_std + e_mean if e_mean is not None and e_std is not None else energy

    plt.figure(figsize=(12, 4))
    plt.plot(energy_plot[:mel_len])
    plt.xlabel("Mel frame")
    plt.ylabel("Predicted energy")
    plt.title(f"{sid} - Predicted Energy")
    plt.tight_layout()
    plt.savefig(ENERGY_DIR / f"{sid}_energy.png", dpi=150)
    plt.close()

    metadata.append({
        "id": sid,
        "original_assignment_text": item["original"],
        "synthesis_input": item["input"],
        "audio_file": str(wav_path),
        "predicted_mel_frames": mel_len,
        "predicted_duration_frames": durations[:len(ids)].tolist()
    })

    print(f"{sid}: audio and 3 plots saved; mel frames={mel_len}")

(OUTPUT / "inference_metadata.json").write_text(
    json.dumps(metadata, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

# Copy training curves into Results
import shutil
shutil.copy2(OUTPUT / "train_loss.png", RESULTS / "training_loss.png")
shutil.copy2(OUTPUT / "validation_loss.png", RESULTS / "validation_loss.png")

print("\nENGLISH INFERENCE COMPLETE")
print("Audio:", AUDIO_DIR)
print("Duration plots:", DURATION_DIR)
print("Pitch plots:", PITCH_DIR)
print("Energy plots:", ENERGY_DIR)

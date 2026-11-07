import os
import sys
import json
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FS = ROOT / "FastSpeech2-Telugu"
OUTPUT = ROOT / "outputs" / "telugu"
CHECKPOINT = OUTPUT / "best_model.pth"
PRE = ROOT / "preprocessed_data"

sys.path.insert(0, str(FS))
os.chdir(FS)

from model.fastspeech2 import FastSpeech2
from text import text_to_sequence
from audio.stft import TacotronSTFT
from audio.tools import inv_mel_spec

# Keep original assignment wording in metadata.
tests = [
    {
        "id": "T1",
        "original": "న్యూ రోఫార్మ కోలాజికల్ పరిశోధనలోనూతన ఔషధాల ప్రభావాన్ని విశ్లషే ించారు.",
        "input": "న్యూ రోఫార్మ కోలాజికల్ పరిశోధనలోనూతన మందుల ప్రభావాన్ని విశ్లషే ించారు"
    },
    {
        "id": "T2a",
        "original": "తాడుపై అతని పట్టుచాలా బలంగా ఉంది.",
        "input": "తాడుపై అతని పట్టుచాలా బలంగా ఉంది"
    },
    {
        "id": "T2b",
        "original": "ఆమె పట్టుచీరను ధరించి వేడుకకు వచ్చ ింది.",
        "input": "ఆమె పట్టుచీరను ధరించి వేడుకకు వచ్చ ింది"
    },
    {
        "id": "T3",
        "original": "పులి అడ విలో నిశ్శబ్ంద గా నడిచింది.",
        "input": "పులి అడ విలో నిశ్శబ్ద గా నడిచింది"
    },
    {
        "id": "T4",
        "original": "ఈ సమావేశం 2026 నవంబర్ 12వ తేదీ ఉదయం 10:45 గంటలకు ప్రారంభమవుతుంది, సమయానికి సిద్ంధ గా ఉంచండి.",
        "input": "ఈ సమావేశం రెండు వేల ఇరవై ఆరు నవంబర్ పన్నెండవ తేదీ ఉదయం పది గంటల నలభై ఐదు నిమిషాలకు ప్రారంభమవుతుంది సమయానికి సిద్ధంగా ఉంచండి"
    }
]

AUDIO_DIR = OUTPUT / "test_generated_audios"
DURATION_DIR = ROOT.parent / "Results" / "Telugu" / "variance_adaptor_plots" / "duration"
PITCH_DIR = ROOT.parent / "Results" / "Telugu" / "variance_adaptor_plots" / "pitch"
ENERGY_DIR = ROOT.parent / "Results" / "Telugu" / "variance_adaptor_plots" / "energy"

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

# Read normalization statistics for pitch and energy.
stats_path = PRE / "stats.json"
stats = json.loads(stats_path.read_text(encoding="utf-8"))

def get_stat(name):
    aliases = {
        "pitch_mean": ["pitch_mean", "p_mean"],
        "pitch_std": ["pitch_std", "p_std"],
        "energy_mean": ["energy_mean", "e_mean"],
        "energy_std": ["energy_std", "e_std"]
    }

    def search(obj):
        if isinstance(obj, dict):
            for key in aliases[name]:
                if key in obj:
                    return float(obj[key])
            for value in obj.values():
                result = search(value)
                if result is not None:
                    return result
        return None

    return search(stats)

p_mean, p_std = get_stat("pitch_mean"), get_stat("pitch_std")
e_mean, e_std = get_stat("energy_mean"), get_stat("energy_std")

metadata = []

print("Loaded Telugu checkpoint, epoch:", checkpoint.get("epoch"))
print("Device:", device)
print("Generating five Telugu samples...")

for item in tests:
    sid = item["id"]
    text = item["input"]

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

    # Griffin-Lim waveform reconstruction.
    wav_path = AUDIO_DIR / f"{sid}.wav"
    inv_mel_spec(
        mel.transpose(0, 1).contiguous(),
        str(wav_path),
        stft,
        griffin_iters=12
    )

    # Duration plot: one predicted duration per input character.
    plt.figure(figsize=(max(10, len(text) * 0.22), 4))
    plt.bar(range(len(durations[:len(text)])), durations[:len(text)])
    plt.xlabel("Character index")
    plt.ylabel("Predicted duration (mel frames)")
    plt.title(f"{sid} - Predicted Duration")
    plt.tight_layout()
    plt.savefig(DURATION_DIR / f"{sid}_duration.png", dpi=150)
    plt.close()

    # Pitch plot: convert normalized predictions back to original scale.
    if p_mean is not None and p_std is not None:
        pitch_plot = pitch * p_std + p_mean
        pitch_label = "Predicted F0 (Hz)"
    else:
        pitch_plot = pitch
        pitch_label = "Normalized predicted pitch"

    plt.figure(figsize=(12, 4))
    plt.plot(pitch_plot[:mel_len])
    plt.xlabel("Mel frame")
    plt.ylabel(pitch_label)
    plt.title(f"{sid} - Predicted Pitch")
    plt.tight_layout()
    plt.savefig(PITCH_DIR / f"{sid}_pitch.png", dpi=150)
    plt.close()

    # Energy plot.
    if e_mean is not None and e_std is not None:
        energy_plot = energy * e_std + e_mean
        energy_label = "Predicted energy"
    else:
        energy_plot = energy
        energy_label = "Normalized predicted energy"

    plt.figure(figsize=(12, 4))
    plt.plot(energy_plot[:mel_len])
    plt.xlabel("Mel frame")
    plt.ylabel(energy_label)
    plt.title(f"{sid} - Predicted Energy")
    plt.tight_layout()
    plt.savefig(ENERGY_DIR / f"{sid}_energy.png", dpi=150)
    plt.close()

    metadata.append({
        "id": sid,
        "original_assignment_text": item["original"],
        "synthesis_input": text,
        "audio_file": str(wav_path),
        "predicted_mel_frames": mel_len,
        "predicted_duration_frames": durations[:len(text)].tolist()
    })

    print(f"{sid}: audio and 3 plots saved; mel frames={mel_len}")

(OUTPUT / "inference_metadata.json").write_text(
    json.dumps(metadata, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

print("\nTELUGU INFERENCE COMPLETE")
print("Audio files:", AUDIO_DIR)
print("Duration plots:", DURATION_DIR)
print("Pitch plots:", PITCH_DIR)
print("Energy plots:", ENERGY_DIR)
print("Metadata:", OUTPUT / "inference_metadata.json")

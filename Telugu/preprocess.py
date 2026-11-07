from pathlib import Path
import os, sys, json, random
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import yaml
import pyworld

ROOT = Path(__file__).resolve().parent
FS = ROOT / "FastSpeech2-Telugu"
DATA = ROOT / "data"
PRE = ROOT / "preprocessed_data"

SR = 16000
HOP = 256
CTC_STRIDE = 320

def allocate(weights, total):
    weights = np.maximum(np.asarray(weights, dtype=float), 1)
    n = len(weights)
    if total < n:
        raise ValueError(f"{total} mel frames cannot represent {n} characters")
    raw = weights / weights.sum() * (total - n)
    result = np.floor(raw).astype(int) + 1
    remainder = total - result.sum()
    order = np.argsort(raw - np.floor(raw))[::-1]
    for i in order[:remainder]:
        result[i] += 1
    return result

def main():
    os.chdir(FS)
    sys.path.insert(0, str(FS))

    from audio.stft import TacotronSTFT

    with open(DATA / "alignments.jsonl", encoding="utf-8") as f:
        alignments = {
            Path((r := json.loads(line))["audio_filepath"]).stem: r
            for line in f
        }

    metadata = pd.read_csv(DATA / "metadata.csv", encoding="utf-8")

    cfg = yaml.safe_load(
        (FS / "config/LJSpeech/preprocess.yaml").read_text(encoding="utf-8")
    )
    audio_cfg = cfg["preprocessing"]["audio"]
    stft_cfg = cfg["preprocessing"]["stft"]
    mel_cfg = cfg["preprocessing"]["mel"]

    stft = TacotronSTFT(
        stft_cfg["filter_length"],
        stft_cfg["hop_length"],
        stft_cfg["win_length"],
        mel_cfg["n_mel_channels"],
        audio_cfg["sampling_rate"],
        mel_cfg["mel_fmin"],
        mel_cfg["mel_fmax"]
    ).cpu().eval()

    for folder in ["mel", "pitch", "energy", "duration"]:
        (PRE / folder).mkdir(parents=True, exist_ok=True)

    records, skipped = [], []
    all_pitch, all_energy = [], []
    vocabulary = set()

    for row in metadata.itertuples(index=False):
        audio_path = Path(row.audio_filepath)
        key = audio_path.stem

        if key not in alignments:
            skipped.append({"id": key, "reason": "alignment missing"})
            continue

        alignment = alignments[key]
        chars = alignment["characters"]

        try:
            # Use the aligned character sequence as the exact training text.
            char_list = [x["character"] for x in chars]
            text = "".join(char_list)

            wav, sr = sf.read(str(audio_path), dtype="float32")
            if wav.ndim == 2:
                wav = wav.mean(axis=1)
            if sr != SR:
                raise ValueError(f"Expected 16000 Hz, received {sr}")

            start = max(0, int(chars[0]["start_frame"]) * CTC_STRIDE)
            end = min(len(wav), int(chars[-1]["end_frame"]) * CTC_STRIDE)
            segment = wav[start:end]

            if len(segment) < 1024:
                raise ValueError("Aligned audio segment is too short")

            audio_tensor = torch.from_numpy(segment).float().unsqueeze(0)

            with torch.no_grad():
                mel_tensor, energy_tensor = stft.mel_spectrogram(audio_tensor)

            mel = mel_tensor.squeeze(0).transpose(0, 1).numpy().astype(np.float32)
            energy = energy_tensor.squeeze(0).numpy().astype(np.float32)
            frames = len(mel)

            durations = allocate(
                [x["duration_frames"] for x in chars],
                frames
            ).astype(np.int64)

            # WORLD pitch extraction, interpolated to the mel frame count.
            f0, times = pyworld.dio(
                segment.astype(np.float64),
                SR,
                frame_period=HOP / SR * 1000
            )
            f0 = pyworld.stonemask(
                segment.astype(np.float64), f0, times, SR
            )

            mel_times = np.arange(frames) * HOP / SR
            pitch = np.interp(
                mel_times, times, f0, left=0, right=0
            ).astype(np.float32)

            if len(energy) != frames:
                energy = np.resize(energy, frames)

            if len(durations) != len(char_list):
                raise ValueError("Character and duration counts differ")

            if durations.sum() != frames:
                raise ValueError("Duration total does not match mel frames")

            np.save(PRE / "mel" / f"{key}.npy", mel)
            np.save(PRE / "pitch" / f"{key}.npy", pitch)
            np.save(PRE / "energy" / f"{key}.npy", energy)
            np.save(PRE / "duration" / f"{key}.npy", durations)

            vocabulary.update(char_list)
            all_pitch.append(pitch)
            all_energy.append(energy)

            records.append({
                "id": key,
                "text": text,
                "speaker": 0,
                "mel_frames": int(frames),
                "characters": len(char_list)
            })

            if len(records) % 20 == 0:
                print(f"Processed {len(records)} recordings")

        except Exception as exc:
            skipped.append({"id": key, "reason": str(exc)})
            print(f"Skipped {key}: {exc}")

    if len(records) < 10:
        raise RuntimeError("Too few usable samples. Check the errors above.")

    # Include all characters found in the aligned transcripts.
    symbols = ["_"] + sorted(vocabulary - {"_"})
    (FS / "text" / "symbols.py").write_text(
        "symbols = " + repr(symbols) + "\n",
        encoding="utf-8"
    )

    # Replace the original English tokenizer with a Telugu character tokenizer.
    tokenizer = '''
from text.symbols import symbols

_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

def text_to_sequence(text, cleaners=None):
    unknown = [c for c in text if c not in _symbol_to_id]
    if unknown:
        raise ValueError("Unknown Telugu characters: " + repr(sorted(set(unknown))))
    return [_symbol_to_id[c] for c in text]

def sequence_to_text(sequence):
    return "".join(_id_to_symbol.get(int(i), "") for i in sequence)
'''
    (FS / "text" / "__init__.py").write_text(
        tokenizer.strip() + "\n", encoding="utf-8"
    )

    # Normalize frame-level pitch and energy.
    p_all = np.concatenate(all_pitch)
    e_all = np.concatenate(all_energy)

    p_mean, p_std = float(p_all.mean()), float(p_all.std() + 1e-6)
    e_mean, e_std = float(e_all.mean()), float(e_all.std() + 1e-6)

    for r in records:
        key = r["id"]
        p = np.load(PRE / "pitch" / f"{key}.npy")
        e = np.load(PRE / "energy" / f"{key}.npy")

        np.save(
            PRE / "pitch" / f"{key}.npy",
            ((p - p_mean) / p_std).astype(np.float32)
        )
        np.save(
            PRE / "energy" / f"{key}.npy",
            ((e - e_mean) / e_std).astype(np.float32)
        )

    p_norm = (p_all - p_mean) / p_std
    e_norm = (e_all - e_mean) / e_std

    stats = {
        "pitch": [
            float(p_norm.min()), float(p_norm.max()), p_mean, p_std
        ],
        "energy": [
            float(e_norm.min()), float(e_norm.max()), e_mean, e_std
        ]
    }
    (PRE / "stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )

    random.Random(42).shuffle(records)
    val_count = max(1, round(len(records) * 0.1))
    validation = records[:val_count]
    training = records[val_count:]

    (PRE / "train.json").write_text(
        json.dumps(training, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    (PRE / "val.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    (PRE / "skipped.json").write_text(
        json.dumps(skipped, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("\nPREPROCESSING FINISHED")
    print("Prepared:", len(records))
    print("Training:", len(training))
    print("Validation:", len(validation))
    print("Skipped:", len(skipped))
    print("Vocabulary size:", len(symbols))
    print("Output:", PRE)

if __name__ == "__main__":
    main()

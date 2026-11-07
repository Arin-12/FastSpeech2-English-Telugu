
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

# Configuration
DATA_DIR = Path("English/data")
OUTPUT_DIR = Path("English/preprocessed")
MEL_DIR = OUTPUT_DIR / "mel"

SAMPLE_RATE = 22050
N_FFT = 1024
HOP_LENGTH = 256
WIN_LENGTH = 1024
N_MELS = 80
FMIN = 0
FMAX = 8000

MEL_DIR.mkdir(parents=True, exist_ok=True)

def process_split(csv_name):
    df = pd.read_csv(DATA_DIR / csv_name)
    records = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=csv_name):
        audio_path = DATA_DIR / row["audio_path"]

        # Load and resample audio
        audio, _ = librosa.load(
            audio_path,
            sr=SAMPLE_RATE,
            mono=True
        )

        # Mel spectrogram
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            n_mels=N_MELS,
            fmin=FMIN,
            fmax=FMAX,
            power=1.0
        )

        # Log mel
        mel = np.log(np.maximum(mel, 1e-5))

        output_path = MEL_DIR / f"{row['id']}.npy"
        np.save(output_path, mel.astype(np.float32))

        records.append({
            "id": row["id"],
            "mel_path": f"mel/{row['id']}.npy",
            "text": row["text"],
            "duration": row["duration"],
            "gender": row["gender"],
            "native_language": row["native_language"]
        })

    output_csv = OUTPUT_DIR / csv_name
    pd.DataFrame(records).to_csv(output_csv, index=False)

    print(f"\nSaved {len(records)} samples to {output_csv}")


if __name__ == "__main__":
    process_split("train.csv")
    process_split("val.csv")
    print("\nEnglish mel preprocessing completed!")

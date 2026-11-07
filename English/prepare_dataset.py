
from datasets import load_dataset, Audio
from pathlib import Path
import pandas as pd

DATASET_NAME = "ai4bharat/Svarah"
DATA_DIR = Path("English/data")
WAV_DIR = DATA_DIR / "wavs"

WAV_DIR.mkdir(parents=True, exist_ok=True)

# Load existing split files
train_df = pd.read_csv(DATA_DIR / "train.csv")
val_df = pd.read_csv(DATA_DIR / "val.csv")

selected_ids = set(train_df["id"]) | set(val_df["id"])
extracted = set()

print("Loading Svarah dataset...")

dataset = load_dataset(DATASET_NAME, split="test")
dataset = dataset.cast_column(
    "audio_filepath", Audio(decode=False)
)

print("Extracting selected WAV files...")

for i, item in enumerate(dataset):
    sample_id = f"svarah_{i:05d}"

    if sample_id not in selected_ids:
        continue

    audio_info = item["audio_filepath"]
    audio_bytes = audio_info["bytes"]

    if audio_bytes is None:
        raise ValueError(f"Audio bytes missing: {sample_id}")

    output_path = WAV_DIR / f"{sample_id}.wav"
    output_path.write_bytes(audio_bytes)

    extracted.add(sample_id)

    if len(extracted) % 500 == 0:
        print(f"Extracted {len(extracted)} files")

# Verify extraction
missing = selected_ids - extracted

if missing:
    raise RuntimeError(f"Missing {len(missing)} audio files")

# Update CSV paths to relative paths
for filename in ["train.csv", "val.csv"]:
    csv_path = DATA_DIR / filename
    df = pd.read_csv(csv_path)
    df["audio_path"] = df["id"].apply(
        lambda x: f"wavs/{x}.wav"
    )
    df.to_csv(csv_path, index=False)

print("\nExtraction completed!")
print("Total extracted:", len(extracted))
print("Training samples:", len(train_df))
print("Validation samples:", len(val_df))

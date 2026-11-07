import os
import csv
import io
from datasets import load_dataset, Audio
import soundfile as sf

DATASET_ID = "SPRINGLab/IndicTTS_Telugu"

OUTPUT_DIR = "Telugu/data"
WAV_DIR = os.path.join(OUTPUT_DIR, "wavs")
METADATA_FILE = os.path.join(OUTPUT_DIR, "metadata.csv")

# First run: process 200 samples to validate the pipeline.
# Change to None to process the entire dataset.
MAX_SAMPLES = 200

os.makedirs(WAV_DIR, exist_ok=True)

print("Loading IndicTTS Telugu dataset...")

ds = load_dataset(
    DATASET_ID,
    split="train",
    streaming=True,
    token=True
)

ds = ds.cast_column("audio", Audio(decode=False))

processed = 0
skipped = 0

print("Starting audio extraction...\n")

with open(METADATA_FILE, "w", newline="", encoding="utf-8") as f:

    writer = csv.writer(f)

    writer.writerow([
        "audio_filepath",
        "text",
        "duration",
        "sampling_rate",
        "gender"
    ])

    for i, sample in enumerate(ds):

        if MAX_SAMPLES is not None and processed >= MAX_SAMPLES:
            break

        audio = sample["audio"]
        audio_bytes = audio.get("bytes")
        text = sample.get("text", "")

        if not audio_bytes or not text.strip():
            skipped += 1
            continue

        filename = f"telugu_{i:06d}.wav"
        filepath = os.path.join(WAV_DIR, filename)

        try:
            # Validate the original audio before saving.
            info = sf.info(io.BytesIO(audio_bytes))

            if info.samplerate != 16000 or info.channels != 1:
                print("Skipping unexpected audio format:", filename)
                skipped += 1
                continue

            # Preserve original audio bytes.
            with open(filepath, "wb") as audio_file:
                audio_file.write(audio_bytes)

            writer.writerow([
                os.path.abspath(filepath),
                text.strip(),
                round(info.duration, 4),
                info.samplerate,
                sample.get("gender", "")
            ])

            processed += 1

            if processed % 25 == 0:
                print(f"Processed: {processed} samples")

        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            skipped += 1

print("\nDataset preparation finished.")
print("Successfully processed:", processed)
print("Skipped:", skipped)
print("Metadata:", METADATA_FILE)
print("Audio directory:", WAV_DIR)
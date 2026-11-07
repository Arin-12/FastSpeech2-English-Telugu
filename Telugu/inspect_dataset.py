from datasets import load_dataset, Audio
import soundfile as sf
import io

DATASET_ID = "SPRINGLab/IndicTTS_Telugu"

print("Loading Telugu dataset without automatic audio decoding...")

ds = load_dataset(
    DATASET_ID,
    split="train",
    streaming=True,
    token=True
)

# Disable TorchCodec-dependent automatic decoding
ds = ds.cast_column("audio", Audio(decode=False))

print("\nInspecting first 5 samples...\n")

for i, sample in enumerate(ds):
    audio = sample["audio"]

    print(f"Sample: {i + 1}")
    print("Text:", sample["text"][:150])
    print("Gender:", sample["gender"])

    audio_bytes = audio.get("bytes")
    audio_path = audio.get("path")

    if audio_bytes:
        info = sf.info(io.BytesIO(audio_bytes))
        print("Sampling rate:", info.samplerate)
        print("Duration:", round(info.duration, 2), "seconds")
        print("Channels:", info.channels)

    elif audio_path:
        print("Audio path:", audio_path)
        try:
            info = sf.info(audio_path)
            print("Sampling rate:", info.samplerate)
            print("Duration:", round(info.duration, 2), "seconds")
        except Exception as e:
            print("Audio metadata unavailable:", e)

    else:
        print("Audio data unavailable")

    print("-" * 50)

    if i == 4:
        break

print("\nDataset inspection completed.")
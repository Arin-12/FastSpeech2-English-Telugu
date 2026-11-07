from datasets import load_dataset, Audio
import soundfile as sf
import io

ds = load_dataset(
    "ai4bharat/Svarah",
    split="test",
    streaming=True
)

ds = ds.cast_column("audio_filepath", Audio(decode=False))

checked = 0

for sample in ds:
    duration = float(sample["duration"])

    # Inspect recordings within our proposed duration range
    if not 1.0 <= duration <= 15.0:
        continue

    audio = sample["audio_filepath"]

    try:
        audio_file = io.BytesIO(audio["bytes"]) if audio.get("bytes") else audio["path"]
        info = sf.info(audio_file)

        print(f"\n--- AUDIO {checked + 1} ---")
        print("Transcript:", sample["text"])
        print("Metadata duration:", duration)
        print("Actual duration:", round(info.duration, 3))
        print("Sample rate:", info.samplerate)
        print("Channels:", info.channels)
        print("Format:", info.format)
        print("Subtype:", info.subtype)

        checked += 1

    except Exception as e:
        print("Audio inspection failed:", str(e))

    if checked == 5:
        break

print("\nAudio inspection complete.")
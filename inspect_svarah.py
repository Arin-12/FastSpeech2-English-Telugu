from datasets import load_dataset, Audio

print("Loading Svarah in streaming mode...")

ds = load_dataset(
    "ai4bharat/Svarah",
    split="test",
    streaming=True
)

# Disable audio decoding; retain audio file references
ds = ds.cast_column("audio_filepath", Audio(decode=False))

for i, sample in enumerate(ds):
    print(f"\n--- SAMPLE {i + 1} ---")

    print("Transcript:", sample["text"])
    print("Duration:", sample["duration"])
    print("Gender:", sample["gender"])
    print("Age group:", sample["age-group"])
    print("Native language:", sample["primary_language"])
    print("State:", sample["native_place_state"])
    print("District:", sample["native_place_district"])

    if i == 4:
        break
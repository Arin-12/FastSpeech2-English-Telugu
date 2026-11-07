from datasets import load_dataset, Audio
from collections import Counter
import numpy as np

print("Loading Svarah metadata in streaming mode...")

ds = load_dataset(
    "ai4bharat/Svarah",
    split="test",
    streaming=True
)

# Do not decode audio during the metadata audit
ds = ds.cast_column("audio_filepath", Audio(decode=False))

durations = []
gender_counts = Counter()
language_counts = Counter()

total = 0
empty_text = 0
short_audio = 0
long_audio = 0
suspicious_spacing = 0

for sample in ds:
    total += 1

    duration = float(sample["duration"])
    durations.append(duration)

    text = sample["text"] or ""

    if not text.strip():
        empty_text += 1

    if duration < 1:
        short_audio += 1

    if duration > 15:
        long_audio += 1

    if text != text.strip() or "  " in text:
        suspicious_spacing += 1

    gender_counts[sample["gender"]] += 1
    language_counts[sample["primary_language"]] += 1

    if total % 500 == 0:
        print(f"Processed {total} records...")

durations = np.array(durations)

print("\n========== SVARAH DATASET AUDIT ==========")
print("Total records:", total)
print("Empty transcripts:", empty_text)
print("Audio shorter than 1 second:", short_audio)
print("Audio longer than 15 seconds:", long_audio)
print("Transcripts with suspicious spacing:", suspicious_spacing)

print("\nDuration statistics (seconds):")
print("Minimum:", round(float(np.min(durations)), 2))
print("Maximum:", round(float(np.max(durations)), 2))
print("Mean:", round(float(np.mean(durations)), 2))
print("Median:", round(float(np.median(durations)), 2))
print("25th percentile:", round(float(np.percentile(durations, 25)), 2))
print("75th percentile:", round(float(np.percentile(durations, 75)), 2))
print("Total audio hours:", round(float(np.sum(durations)) / 3600, 2))

print("\nGender distribution:")
for key, value in gender_counts.most_common():
    print(f"{key}: {value}")

print("\nTop 15 native languages:")
for key, value in language_counts.most_common(15):
    print(f"{key}: {value}")

print("\n========== AUDIT COMPLETE ==========")
import pandas as pd
from pathlib import Path

root = Path.cwd()
raw = pd.read_csv("English/data/train.csv")
prep = pd.read_csv("English/preprocessed/train.csv")

print("Raw columns:", raw.columns.tolist())
print("Preprocessed columns:", prep.columns.tolist())

df = raw.merge(prep[["id", "mel_path"]], on="id")
print("\nMerged rows:", len(df))

for _, r in df.head(5).iterrows():
    print("\nID:", r["id"])
    print("Audio:", r["audio_path"], "exists:", (root / "English/data" / str(r["audio_path"])).exists())
    print("Mel:", r["mel_path"])
    p = Path(str(r["mel_path"]))
    candidates = [p, root / p, root / "English" / p, root / "English/preprocessed" / p, root / "English/preprocessed/mel" / p.name]
    print("Mel candidates:", [(str(x), x.exists()) for x in candidates])

print("\nAlignment IDs:")
with open("English/alignments.jsonl", encoding="utf-8") as f:
    for _ in range(2):
        print(f.readline().strip()[:250])

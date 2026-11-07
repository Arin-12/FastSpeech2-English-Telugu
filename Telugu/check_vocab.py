import csv, json
from pathlib import Path

root = Path("Telugu")
with open(root / "aligner" / "vocab.json", encoding="utf-8") as f:
    vocab = json.load(f)

csv_files = list(root.rglob("*.csv"))
csv_files = [p for p in csv_files if "aligner" not in str(p).lower()]

print("CSV files found:")
for p in csv_files:
    print(" ", p)

for csv_path in csv_files:
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        continue

    text_col = next((k for k in rows[0] if k.lower() in ("text", "transcript", "sentence")), None)
    if not text_col:
        continue

    chars = set()
    for row in rows:
        text = (row.get(text_col) or "").strip().replace(" ", "|")
        chars.update(text)

    missing = sorted(c for c in chars if c not in vocab)

    print("\nMetadata:", csv_path)
    print("Rows:", len(rows))
    print("Transcript column:", text_col)
    print("Unique characters:", len(chars))
    print("Missing characters:", repr(missing))
    print("Vocabulary:", list(vocab.keys()))
    print("First transcript:", repr(rows[0][text_col]))
    break
else:
    print("\nERROR: Could not find a CSV with a recognized transcript column.")

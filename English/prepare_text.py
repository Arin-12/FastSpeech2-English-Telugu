
import re
import pandas as pd
from pathlib import Path
import inflect

DATA_DIR = Path("English/preprocessed")
p = inflect.engine()

def normalize_text(text):
    text = str(text).lower().strip()

    # Expand common abbreviations
    text = re.sub(r"\bdr\.", "doctor", text)
    text = re.sub(r"\bmr\.", "mister", text)
    text = re.sub(r"\bmrs\.", "misses", text)
    text = re.sub(r"\bst\.", "saint", text)

    # Convert numbers up to 4 digits into words
    def replace_number(match):
        number = match.group()
        try:
            if len(number) <= 4:
                return p.number_to_words(int(number))
            return " ".join(p.number_to_words(int(d)) for d in number)
        except (ValueError, TypeError):
            return number

    text = re.sub(r"\b\d+\b", replace_number, text)

    # Remove punctuation while preserving word boundaries
    text = re.sub(r"[^a-zA-Z'\s]", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()

for split in ["train.csv", "val.csv"]:
    path = DATA_DIR / split
    df = pd.read_csv(path)

    df["normalized_text"] = df["raw_text"].apply(normalize_text)

    df.to_csv(path, index=False)
    print(f"{split}: processed {len(df)} transcripts")

print("English text normalization fixed.")

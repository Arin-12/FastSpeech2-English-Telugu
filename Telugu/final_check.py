import json
import inspect
from pathlib import Path

root = Path("Telugu/FastSpeech2-Telugu")

# Alignment record
p = Path("Telugu/data/alignments.jsonl")
with p.open(encoding="utf-8") as f:
    record = json.loads(next(f))

print("===== ALIGNMENT RECORD =====")
print("Keys:", list(record.keys()))
print("Text:", repr(record.get("text")))
print("Duration:", record.get("audio_duration"))
print("Sample rate:", record.get("sample_rate"))
print("Score:", record.get("alignment_score"))
print("First 8 characters:")
for item in record.get("characters", [])[:8]:
    print(item)

# Model forward signature
import sys
sys.path.insert(0, str(root.resolve()))
from model.fastspeech2 import FastSpeech2

print("\n===== FASTSPEECH2 FORWARD =====")
print(inspect.signature(FastSpeech2.forward))

# Text symbols
p = root / "text/symbols.py"
print("\n===== SYMBOL FILE =====")
print(p.read_text(encoding="utf-8")[:3000])

# Model configuration
import yaml
p = root / "config/LJSpeech/model.yaml"
print("\n===== MODEL CONFIG =====")
print(yaml.safe_load(p.read_text(encoding="utf-8")))

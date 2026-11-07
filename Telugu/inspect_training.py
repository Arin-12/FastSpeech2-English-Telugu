from pathlib import Path

root = Path("Telugu/FastSpeech2-Telugu")

files = [
    "model/fastspeech2.py",
    "model/loss.py",
    "audio/tools.py",
    "config/LJSpeech/preprocess.yaml",
    "config/LJSpeech/model.yaml",
    "config/LJSpeech/train.yaml",
]

for filename in files:
    p = root / filename
    print(f"\n{'='*65}\n{filename}\n{'='*65}")
    if not p.exists():
        print("NOT FOUND")
        continue

    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    limit = 240 if filename.endswith(".py") else 180
    for i, line in enumerate(lines[:limit], 1):
        print(f"{i}: {line}")

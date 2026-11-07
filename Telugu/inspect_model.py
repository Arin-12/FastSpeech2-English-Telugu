from pathlib import Path

root = Path("FastSpeech2-master")

files = [
    "utils/model.py",
    "model/fastspeech2.py",
    "model/modules.py",
    "model/loss.py",
    "audio/stft.py",
    "audio/tools.py",
    "train.py",
]

for name in files:
    p = root / name
    print(f"\n{'='*18} {name} {'='*18}")
    if not p.exists():
        print("NOT FOUND")
        continue

    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    if name == "train.py":
        ranges = [(1, 75), (170, 220)]
    elif name == "model/fastspeech2.py":
        ranges = [(1, 240)]
    elif name == "utils/model.py":
        ranges = [(1, 180)]
    elif name == "model/modules.py":
        ranges = [(1, 220)]
    elif name == "model/loss.py":
        ranges = [(1, 160)]
    else:
        ranges = [(1, 150)]

    for start, end in ranges:
        for i in range(start-1, min(end, len(lines))):
            print(f"{i+1}: {lines[i]}")

from pathlib import Path

root = Path("Telugu/FastSpeech2-Telugu")

for filename in [
    "model/modules.py",
    "model/varianceadaptor.py",
    "utils/tools.py",
    "audio/stft.py",
]:
    p = root / filename
    if not p.exists():
        continue

    print(f"\n===== {filename} =====")
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    for i, line in enumerate(lines[:260], 1):
        print(f"{i}: {line}")

print("\n===== ORIGINAL VOCODER CONFIG =====")
p = root / "hifigan/config.json"
if p.exists():
    print(p.read_text(encoding="utf-8", errors="replace"))

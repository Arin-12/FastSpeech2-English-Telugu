from pathlib import Path

root = Path("FastSpeech2-master")

for folder in ["transformer", "hifigan"]:
    p = root / folder
    print(f"\n===== {folder} =====")
    if p.exists():
        for f in p.glob("*.py"):
            print(f"\n--- {f.name} ---")
            lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
            for i, line in enumerate(lines[:180], 1):
                print(f"{i}: {line}")

print("\n===== VOCODER CHECK =====")
for f in ["hifigan/config.json", "hifigan/generator_LJSpeech.pth.tar",
          "hifigan/generator_universal.pth.tar"]:
    p = root / f
    print(f"{f}: {'FOUND' if p.exists() else 'MISSING'}")

print("\n===== TELUGU ALIGNMENT CHECK =====")
p = Path("Telugu/data/alignments.jsonl")
print("Alignment file:", p.exists())
if p.exists():
    print("Size:", round(p.stat().st_size / 1024, 1), "KB")

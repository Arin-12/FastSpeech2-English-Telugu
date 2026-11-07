from pathlib import Path
import importlib.util

root = Path("FastSpeech2-master")

print("=== REQUIRED LIBRARIES ===")
for package in ["torch", "torchaudio", "librosa", "pyworld", "soundfile", "yaml", "scipy"]:
    try:
        mod = __import__(package)
        print(package, getattr(mod, "__version__", "available"))
    except Exception as e:
        print(package, "ERROR:", str(e)[:150])

files = [
    "config/LJSpeech/preprocess.yaml",
    "config/LJSpeech/model.yaml",
    "config/LJSpeech/train.yaml",
    "dataset.py",
    "preprocessor/preprocessor.py",
    "train.py",
    "text/__init__.py",
    "text/symbols.py",
]

for name in files:
    p = root / name
    print(f"\n{'='*20} {name} {'='*20}")
    if not p.exists():
        print("FILE NOT FOUND")
        continue

    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    if name.endswith(".yaml") or name.endswith("symbols.py") or name.endswith("text/__init__.py"):
        print("\n".join(f"{i+1}: {line}" for i, line in enumerate(lines[:180])))
    else:
        keywords = (
            "class Dataset", "def __init__", "def __getitem__",
            "def collate_fn", "def process", "def build_from_path",
            "TextGrid", "duration", "pitch", "energy", "mel",
            "ArgumentParser", "argparse", "FastSpeech2Loss",
            "get_model", "get_data"
        )
        selected = set()
        for i, line in enumerate(lines):
            if any(k in line for k in keywords):
                selected.update(range(max(0, i-3), min(len(lines), i+7)))
        for i in sorted(selected):
            print(f"{i+1}: {lines[i]}")

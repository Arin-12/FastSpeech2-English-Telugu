from pathlib import Path
import shutil

source = Path("FastSpeech2-master")
target = Path("Telugu/FastSpeech2-Telugu")

if not target.exists():
    shutil.copytree(
        source,
        target,
        ignore=shutil.ignore_patterns(
            ".git", "__pycache__", "*.pyc",
            "output", "preprocessed_data"
        )
    )
    print("Created separate Telugu FastSpeech2 copy.")
else:
    print("Telugu copy already exists.")

# Show text processing and dataset expectations
for filename in ["text/__init__.py", "text/cleaners.py", "dataset.py"]:
    p = target / filename
    print(f"\n===== {filename} =====")
    if p.exists():
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        for i, line in enumerate(lines[:180], 1):
            print(f"{i}: {line}")
    else:
        print("NOT FOUND")

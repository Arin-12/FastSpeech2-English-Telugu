from huggingface_hub import hf_hub_download
from pathlib import Path

repo = "XDcobra/wav2vec2-large-xlsr-53-telugu-ONNX"
target = Path("Telugu/aligner")
target.mkdir(parents=True, exist_ok=True)

files = ["onnx/model_int8.onnx", "vocab.json", "config.json"]

for name in files:
    print(f"\nDownloading {name} ...", flush=True)
    path = hf_hub_download(
        repo_id=repo,
        filename=name,
        local_dir=str(target)
    )
    print("Saved:", path, flush=True)

print("\nDownloads complete.")

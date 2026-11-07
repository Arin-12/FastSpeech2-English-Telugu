import json
import onnxruntime as ort
from pathlib import Path

root = Path("Telugu/aligner")

with open(root / "vocab.json", encoding="utf-8") as f:
    vocab = json.load(f)

with open(root / "config.json", encoding="utf-8") as f:
    config = json.load(f)

print("\nVOCABULARY")
print("Type:", type(vocab).__name__)
print("Size:", len(vocab))
print("First entries:", list(vocab.items())[:25])

print("\nMODEL CONFIG")
for key in ["sampling_rate", "vocab_size", "pad_token_id", "bos_token_id", "eos_token_id"]:
    print(f"{key}: {config.get(key)}")

print("\nLOADING ONNX MODEL...")
model_path = root / "onnx" / "model_int8.onnx"
session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

print("INPUTS:")
for item in session.get_inputs():
    print(item.name, item.shape, item.type)

print("OUTPUTS:")
for item in session.get_outputs():
    print(item.name, item.shape, item.type)

print("\nMODEL INSPECTION COMPLETE")

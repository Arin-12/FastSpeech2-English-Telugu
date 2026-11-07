from pathlib import Path
import shutil, yaml, json, collections

root = Path.cwd()
src = root/"Telugu"/"FastSpeech2-Telugu"
dst = root/"English"/"FastSpeech2-English"

if not dst.exists():
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns(
        ".git", "__pycache__", "outputs", "checkpoints"
    ))

symbols = ["_"] + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["'", "|"]
(dst/"text"/"symbols.py").write_text(
    "symbols = " + repr(symbols) + "\n", encoding="utf-8"
)

p = dst/"config"/"LJSpeech"/"preprocess.yaml"
cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
cfg["dataset"] = "English"
cfg["preprocessing"]["text"]["language"] = "en"
cfg["preprocessing"]["audio"]["sampling_rate"] = 22050
cfg["preprocessing"]["mel"]["mel_fmax"] = 8000
p.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

m = dst/"config"/"LJSpeech"/"model.yaml"
mcfg = yaml.safe_load(m.read_text(encoding="utf-8"))
mcfg["max_seq_len"] = 10000
m.write_text(yaml.safe_dump(mcfg, sort_keys=False), encoding="utf-8")

manifest = root/"English"/"preprocessed"/"english_manifest.jsonl"
records = [json.loads(x) for x in manifest.read_text(encoding="utf-8").splitlines()]
chars = sorted(set("".join(x["text"] for x in records)))
print("Model copy:", dst)
print("Vocabulary:", symbols)
print("Characters in manifest:", chars)
print("Unsupported:", sorted(set(chars)-set(symbols)))
print("Samples:", len(records))

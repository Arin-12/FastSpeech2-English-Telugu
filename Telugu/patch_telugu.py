from pathlib import Path
import yaml

root = Path("Telugu/FastSpeech2-Telugu")

# 1. Make STFT work on CPU and GPU
p = root / "audio/stft.py"
s = p.read_text(encoding="utf-8")

old = """        forward_transform = F.conv1d(
            input_data.cuda(),
            torch.autograd.Variable(self.forward_basis, requires_grad=False).cuda(),
            stride=self.hop_length,
            padding=0,
        ).cpu()"""

new = """        forward_basis = self.forward_basis.to(input_data.device)
        forward_transform = F.conv1d(
            input_data,
            forward_basis,
            stride=self.hop_length,
            padding=0,
        )"""

if old in s:
    s = s.replace(old, new)
    p.write_text(s, encoding="utf-8")
    print("Patched STFT for CPU/GPU.")
elif "input_data.cuda()" not in s:
    print("STFT already appears patched.")
else:
    raise RuntimeError("Could not find the expected STFT code. No replacement made.")

# 2. Fix energy control parameter
p = root / "model/modules.py"
s = p.read_text(encoding="utf-8")
old = """                x, pitch_target, mel_mask, p_control
            )
            x = x + energy_embedding"""
new = """                x, energy_target, mel_mask, e_control
            )
            x = x + energy_embedding"""

# Replace only the frame-level energy block if it matches
old_block = """            energy_prediction, energy_embedding = self.get_energy_embedding(
                x, energy_target, mel_mask, p_control
            )"""
new_block = """            energy_prediction, energy_embedding = self.get_energy_embedding(
                x, energy_target, mel_mask, e_control
            )"""

if old_block in s:
    s = s.replace(old_block, new_block)
    p.write_text(s, encoding="utf-8")
    print("Fixed frame-level energy control.")
else:
    print("Energy-control code already changed or pattern not found.")

# 3. Configure Telugu preprocessing
p = root / "config/LJSpeech/preprocess.yaml"
cfg = yaml.safe_load(p.read_text(encoding="utf-8"))

cfg["dataset"] = "Telugu"
cfg["path"]["corpus_path"] = "../data"
cfg["path"]["raw_path"] = "../data"
cfg["path"]["preprocessed_path"] = "../preprocessed_data"

cfg["preprocessing"]["val_size"] = 20
cfg["preprocessing"]["text"]["text_cleaners"] = ["basic_cleaners"]
cfg["preprocessing"]["text"]["language"] = "te"

cfg["preprocessing"]["audio"]["sampling_rate"] = 16000
cfg["preprocessing"]["stft"]["filter_length"] = 1024
cfg["preprocessing"]["stft"]["hop_length"] = 256
cfg["preprocessing"]["stft"]["win_length"] = 1024
cfg["preprocessing"]["mel"]["n_mel_channels"] = 80
cfg["preprocessing"]["mel"]["mel_fmin"] = 0
cfg["preprocessing"]["mel"]["mel_fmax"] = 7600

cfg["preprocessing"]["pitch"]["feature"] = "frame_level"
cfg["preprocessing"]["energy"]["feature"] = "frame_level"

p.write_text(
    yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
    encoding="utf-8"
)
print("Updated Telugu preprocessing configuration.")

print("\nConfiguration patch completed.")

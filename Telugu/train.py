from pathlib import Path
import os, sys, json, random, argparse
import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parent
FS = ROOT / "FastSpeech2-Telugu"
PRE = ROOT / "preprocessed_data"

class TeluguDataset(Dataset):
    def __init__(self, split):
        self.items = json.loads(
            (PRE / f"{split}.json").read_text(encoding="utf-8")
        )
        from text.symbols import symbols
        self.lookup = {s: i for i, s in enumerate(symbols)}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        r = self.items[index]
        k = r["id"]
        return (
            k, r["text"],
            np.load(PRE/"mel"/f"{k}.npy"),
            np.load(PRE/"pitch"/f"{k}.npy"),
            np.load(PRE/"energy"/f"{k}.npy"),
            np.load(PRE/"duration"/f"{k}.npy")
        )

def collate(items):
    B = len(items)
    lookup = TeluguDataset.lookup

    src_lens = np.array([len(x[1]) for x in items], dtype=np.int64)
    mel_lens = np.array([len(x[2]) for x in items], dtype=np.int64)
    max_src, max_mel = int(src_lens.max()), int(mel_lens.max())

    texts = np.zeros((B, max_src), dtype=np.int64)
    mels = np.zeros((B, max_mel, 80), dtype=np.float32)
    pitches = np.zeros((B, max_mel), dtype=np.float32)
    energies = np.zeros((B, max_mel), dtype=np.float32)
    durations = np.zeros((B, max_src), dtype=np.int64)

    for i, (key, text, mel, pitch, energy, dur) in enumerate(items):
        ids = [lookup[c] for c in text]

        if len(ids) != len(dur):
            raise ValueError(f"{key}: text/duration lengths differ")

        if int(dur.sum()) != len(mel):
            raise ValueError(
                f"{key}: duration sum {dur.sum()} != mel frames {len(mel)}"
            )

        texts[i, :len(ids)] = ids
        mels[i, :len(mel)] = mel
        pitches[i, :len(pitch)] = pitch
        energies[i, :len(energy)] = energy
        durations[i, :len(dur)] = dur

    ids = [x[0] for x in items]
    raw_texts = [x[1] for x in items]
    speakers = np.zeros(B, dtype=np.int64)

    return (
        ids, raw_texts, speakers, texts, src_lens, max_src,
        mels, mel_lens, max_mel, pitches, energies, durations
    )

def run_epoch(model, criterion, loader, optimizer, device, training):
    model.train(training)
    total = np.zeros(6, dtype=np.float64)
    batches = 0

    for batch in loader:
        (ids, raw, speakers, texts, src_lens, max_src,
         mels, mel_lens, max_mel, pitches, energies, durations) = batch

        speakers = torch.as_tensor(speakers, dtype=torch.long, device=device)
        texts = torch.as_tensor(texts, dtype=torch.long, device=device)
        src_lens = torch.as_tensor(src_lens, dtype=torch.long, device=device)
        mels = torch.as_tensor(mels, dtype=torch.float32, device=device)
        mel_lens = torch.as_tensor(mel_lens, dtype=torch.long, device=device)
        pitches = torch.as_tensor(pitches, dtype=torch.float32, device=device)
        energies = torch.as_tensor(energies, dtype=torch.float32, device=device)
        durations = torch.as_tensor(durations, dtype=torch.long, device=device)

        inputs = (
            ids, raw, speakers, texts, src_lens, max_src,
            mels, mel_lens, max_mel, pitches, energies, durations
        )

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            predictions = model(
                speakers, texts, src_lens, max_src,
                mels, mel_lens, max_mel,
                pitches, energies, durations
            )
            losses = criterion(inputs, predictions)
            loss = losses[0]

            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        total += np.array([
            float(x.detach().cpu()) for x in losses
        ])
        batches += 1

    return (total / max(batches, 1)).tolist()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    args = parser.parse_args()

    os.chdir(FS)
    sys.path.insert(0, str(FS))

    from model.fastspeech2 import FastSpeech2
    from model.loss import FastSpeech2Loss

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(4)

    device = torch.device("cpu")

    preprocess_config = yaml.safe_load(
        (FS/"config/LJSpeech/preprocess.yaml").read_text(encoding="utf-8")
    )
    model_config = yaml.safe_load(
        (FS/"config/LJSpeech/model.yaml").read_text(encoding="utf-8")
    )

    # Allow positional encodings to accommodate long Telugu recordings.
    model_config["max_seq_len"] = 10000

    train_ds = TeluguDataset("train")
    val_ds = TeluguDataset("val")

    if args.max_train_samples:
        train_ds.items = train_ds.items[:args.max_train_samples]
    if args.max_val_samples:
        val_ds.items = val_ds.items[:args.max_val_samples]

    # Make the symbol lookup available to the collate function.
    TeluguDataset.lookup = train_ds.lookup

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True, num_workers=0, collate_fn=collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=1,
        shuffle=False, num_workers=0, collate_fn=collate
    )

    model = FastSpeech2(preprocess_config, model_config).to(device)
    criterion = FastSpeech2Loss(preprocess_config, model_config).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-4,
        betas=(0.9, 0.98), eps=1e-9
    )

    output = ROOT/"outputs"/"telugu"
    output.mkdir(parents=True, exist_ok=True)

    history = {"train": [], "validation": []}
    best_loss = float("inf")

    print("Device:", device)
    print("Training samples:", len(train_ds))
    print("Validation samples:", len(val_ds))
    print("Epochs:", args.epochs)

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(
            model, criterion, train_loader, optimizer, device, True
        )
        val_loss = run_epoch(
            model, criterion, val_loader, optimizer, device, False
        )

        history["train"].append(train_loss)
        history["validation"].append(val_loss)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train loss: {train_loss[0]:.4f} | "
            f"Validation loss: {val_loss[0]:.4f}"
        )

        checkpoint = {
            "model": model.state_dict(),
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": val_loss
        }

        torch.save(checkpoint, output/"latest_model.pth")

        if val_loss[0] < best_loss:
            best_loss = val_loss[0]
            torch.save(checkpoint, output/"best_model.pth")

        (output/"loss_history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )

    # Plot the training and validation loss curves.
    names = [
        "Total", "Mel", "Postnet Mel",
        "Pitch", "Energy", "Duration"
    ]

    for split in ["train", "validation"]:
        values = np.asarray(history[split])
        plt.figure(figsize=(10, 5))

        for i, name in enumerate(names):
            if i < values.shape[1]:
                plt.plot(
                    range(1, len(values) + 1),
                    values[:, i],
                    marker="o",
                    label=name
                )

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Telugu FastSpeech 2 - {split.title()} Loss")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output/f"{split}_loss.png", dpi=160)
        plt.close()

    print("\nTRAINING FINISHED")
    print("Checkpoint folder:", output)
    print("Best model:", output/"best_model.pth")

if __name__ == "__main__":
    main()


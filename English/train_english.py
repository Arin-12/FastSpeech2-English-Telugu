from pathlib import Path
import os, sys, json, random, argparse
import numpy as np
import torch, yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

ROOT=Path(__file__).resolve().parent
FS=ROOT/"FastSpeech2-English"
MANIFEST=ROOT/"preprocessed"/"english_manifest.jsonl"
OUT=ROOT/"outputs"/"english"

class EnglishDataset(Dataset):
    def __init__(self, records):
        self.items=records
    def __len__(self):
        return len(self.items)
    def __getitem__(self,i):
        r=self.items[i]; k=r["id"]
        mel=np.load(ROOT/"preprocessed"/"mel"/(k+".npy")).astype(np.float32)
        if mel.shape[0]==80: mel=mel.T
        return (k,r["text"],mel,
            np.load(ROOT/"preprocessed"/"pitch"/(k+".npy")).astype(np.float32),
            np.load(ROOT/"preprocessed"/"energy"/(k+".npy")).astype(np.float32),
            np.asarray(r["durations"],dtype=np.int64))

def collate(items):
    B=len(items)
    sl=np.array([len(x[1]) for x in items],dtype=np.int64)
    ml=np.array([len(x[2]) for x in items],dtype=np.int64)
    ms,mm=int(sl.max()),int(ml.max())
    tx=np.zeros((B,ms),np.int64); mel=np.zeros((B,mm,80),np.float32)
    pi=np.zeros((B,mm),np.float32); en=np.zeros((B,mm),np.float32)
    du=np.zeros((B,ms),np.int64)
    for i,(k,t,m,p,e,d) in enumerate(items):
        ids=[EnglishDataset.lookup[c] for c in t]
        if len(ids)!=len(d) or int(d.sum())!=len(m):
            raise ValueError(f"{k}: text/duration/mel mismatch")
        tx[i,:len(ids)]=ids; mel[i,:len(m)]=m
        pi[i,:len(p)]=p; en[i,:len(e)]=e; du[i,:len(d)]=d
    return ([x[0] for x in items],[x[1] for x in items],
        np.zeros(B,np.int64),tx,sl,ms,mel,ml,mm,pi,en,du)

def run_epoch(model,criterion,loader,opt,device,training):
    model.train(training); total=np.zeros(6); count=0
    for b in loader:
        ids,raw,sp,tx,sl,ms,mel,ml,mm,pi,en,du=b
        def ten(x,d): return torch.as_tensor(x,dtype=d,device=device)
        sp=ten(sp,torch.long); tx=ten(tx,torch.long); sl=ten(sl,torch.long)
        mel=ten(mel,torch.float32); ml=ten(ml,torch.long)
        pi=ten(pi,torch.float32); en=ten(en,torch.float32); du=ten(du,torch.long)
        inp=(ids,raw,sp,tx,sl,ms,mel,ml,mm,pi,en,du)
        if training: opt.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            pred=model(sp,tx,sl,ms,mel,ml,mm,pi,en,du)
            losses=criterion(inp,pred); loss=losses[0]
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                opt.step()
        total+=np.array([float(x.detach().cpu()) for x in losses]); count+=1
    return (total/max(count,1)).tolist()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--epochs",type=int,default=1)
    ap.add_argument("--batch-size",type=int,default=1)
    ap.add_argument("--max-train-samples",type=int,default=0)
    args=ap.parse_args()
    os.chdir(FS); sys.path.insert(0,str(FS))
    from text.symbols import symbols
    from model.fastspeech2 import FastSpeech2
    from model.loss import FastSpeech2Loss
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    torch.set_num_threads(4); device=torch.device("cpu")
    pc=yaml.safe_load((FS/"config/LJSpeech/preprocess.yaml").read_text(encoding="utf-8"))
    mc=yaml.safe_load((FS/"config/LJSpeech/model.yaml").read_text(encoding="utf-8"))
    mc["max_seq_len"]=10000
    records=[json.loads(x) for x in MANIFEST.read_text(encoding="utf-8").splitlines()]
    random.shuffle(records); nv=max(1,int(.1*len(records)))
    val=records[:nv]; train=records[nv:]
    if args.max_train_samples: train=train[:args.max_train_samples]
    lookup={s:i for i,s in enumerate(symbols)}
    missing=set("".join(r["text"] for r in records))-set(lookup)
    if missing: raise ValueError("Missing symbols: "+repr(missing))
    EnglishDataset.lookup=lookup
    tr=DataLoader(EnglishDataset(train),batch_size=args.batch_size,shuffle=True,num_workers=0,collate_fn=collate)
    va=DataLoader(EnglishDataset(val),batch_size=1,shuffle=False,num_workers=0,collate_fn=collate)
    model=FastSpeech2(pc,mc).to(device); criterion=FastSpeech2Loss(pc,mc).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=1e-4,betas=(.9,.98),eps=1e-9)
    OUT.mkdir(parents=True,exist_ok=True); hist={"train":[],"validation":[]}; best=float("inf")
    print("CPU | Train:",len(train),"| Validation:",len(val),flush=True)
    for ep in range(1,args.epochs+1):
        a=run_epoch(model,criterion,tr,opt,device,True)
        b=run_epoch(model,criterion,va,opt,device,False)
        hist["train"].append(a); hist["validation"].append(b)
        print(f"Epoch {ep}: train={a[0]:.4f}, validation={b[0]:.4f}",flush=True)
        ck={"model":model.state_dict(),"epoch":ep,"train_loss":a,"validation_loss":b}
        torch.save(ck,OUT/"latest_model.pth")
        if b[0]<best: best=b[0]; torch.save(ck,OUT/"best_model.pth")
        (OUT/"loss_history.json").write_text(json.dumps(hist,indent=2),encoding="utf-8")
    names=["Total","Mel","Postnet Mel","Pitch","Energy","Duration"]
    for split in ("train","validation"):
        v=np.asarray(hist[split]); plt.figure(figsize=(9,5))
        for i,n in enumerate(names): plt.plot(range(1,len(v)+1),v[:,i],marker="o",label=n)
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("English FastSpeech 2 - "+split.title())
        plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
        plt.savefig(OUT/(split+"_loss.png"),dpi=150); plt.close()
    print("TRAINING COMPLETE:",OUT,flush=True)

if __name__=="__main__": main()

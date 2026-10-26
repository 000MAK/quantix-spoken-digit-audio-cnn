"""Train a compact 1-D CNN on the online Free Spoken Digit Dataset."""
from __future__ import annotations
import argparse, json, re, subprocess
from pathlib import Path

FSDD_URL="https://github.com/Jakobovski/free-spoken-digit-dataset.git"
PATTERN=re.compile(r"^(?P<label>[0-9])_(?P<speaker>[A-Za-z]+)_(?P<index>\d+)\.wav$")

def parse_recording_name(filename: str):
    """Return (digit, speaker) for an FSDD filename or raise ValueError."""
    match=PATTERN.match(Path(filename).name)
    if not match: raise ValueError(f"not a recognized FSDD recording filename: {filename}")
    return int(match.group("label")), match.group("speaker").lower()

def validate_epochs(epochs: int) -> int:
    if int(epochs)<1: raise ValueError("epochs must be at least 1")
    return int(epochs)

def ensure_dataset(destination="data/free-spoken-digit-dataset"):
    path=Path(destination)
    if not (path/"recordings").is_dir():
        path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists(): raise RuntimeError(f"{path} exists but has no recordings/ folder; remove it and retry")
        subprocess.run(["git","clone","--depth","1",FSDD_URL,str(path)],check=True)
    return path/"recordings"

def main(output_dir="outputs", epochs=8, batch_size=32, seed=42, sample_rate=8000, duration=1.0):
    validate_epochs(epochs)
    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import Dataset,DataLoader
    from scipy.io import wavfile
    torch.manual_seed(seed); np.random.seed(seed)
    recordings=ensure_dataset()
    files=sorted(recordings.glob("*.wav"))
    parsed=[]
    for path in files:
        try: label,speaker=parse_recording_name(path.name); parsed.append((path,label,speaker))
        except ValueError: continue
    speakers=sorted({item[2] for item in parsed})
    if len(speakers)<2: raise RuntimeError("FSDD download must contain recordings from at least two speakers")
    held_out=speakers[-1]  # Hold out all recordings from one speaker to reduce speaker leakage.
    train_items=[r for r in parsed if r[2]!=held_out]; test_items=[r for r in parsed if r[2]==held_out]
    target_len=int(sample_rate*duration)
    class AudioRows(Dataset):
        def __init__(self,rows): self.rows=rows
        def __len__(self): return len(self.rows)
        def __getitem__(self,index):
            path,label,_=self.rows[index]; rate,wave=wavfile.read(path)
            wave=np.asarray(wave,dtype=np.float32)
            if wave.ndim>1: wave=wave.mean(axis=1)
            if rate!=sample_rate:
                from scipy.signal import resample_poly
                from math import gcd
                div=gcd(int(rate),int(sample_rate)); wave=resample_poly(wave,sample_rate//div,int(rate)//div).astype(np.float32)
            peak=float(np.max(np.abs(wave))) if wave.size else 0.0
            if peak: wave=wave/peak
            if len(wave)<target_len: wave=np.pad(wave,(0,target_len-len(wave)))
            else: wave=wave[:target_len]
            return torch.from_numpy(wave.copy()).unsqueeze(0),label
    train_loader=DataLoader(AudioRows(train_items),batch_size=batch_size,shuffle=True,num_workers=0)
    test_loader=DataLoader(AudioRows(test_items),batch_size=batch_size,shuffle=False,num_workers=0)
    class AudioCNN(nn.Module):
        def __init__(self):
            super().__init__(); self.net=nn.Sequential(
              nn.Conv1d(1,16,15,stride=4,padding=7),nn.BatchNorm1d(16),nn.ReLU(),nn.MaxPool1d(4),
              nn.Conv1d(16,32,9,padding=4),nn.BatchNorm1d(32),nn.ReLU(),nn.MaxPool1d(4),
              nn.Conv1d(32,64,5,padding=2),nn.ReLU(),nn.AdaptiveAvgPool1d(1),nn.Flatten(),nn.Linear(64,10))
        def forward(self,x): return self.net(x)
    model=AudioCNN(); optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3); loss_fn=nn.CrossEntropyLoss()
    for epoch in range(validate_epochs(epochs)):
        model.train(); total=0.0
        for x,y in train_loader:
            optimizer.zero_grad(); loss=loss_fn(model(x),y); loss.backward(); optimizer.step(); total+=loss.item()*len(y)
        print(f"epoch={epoch+1} train_loss={total/max(len(train_items),1):.4f}")
    model.eval(); correct=seen=0
    with torch.no_grad():
        for x,y in test_loader:
            pred=model(x).argmax(1); correct+=int((pred==y).sum()); seen+=len(y)
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True); torch.save(model.state_dict(),out/"spoken_digit_cnn.pt")
    report={"dataset":"FSDD","train_rows":len(train_items),"test_rows":seen,"held_out_speaker":held_out,
      "speakers":speakers,"accuracy":correct/max(seen,1),"epochs":epochs,"seed":seed,
      "warning":"Speaker-held-out educational benchmark only; inspect class-wise scores and upstream dataset terms before reuse."}
    (out/"metrics.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    return report

if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--output-dir",default="outputs"); p.add_argument("--epochs",type=int,default=8); p.add_argument("--batch-size",type=int,default=32); p.add_argument("--seed",type=int,default=42); p.add_argument("--sample-rate",type=int,default=8000); p.add_argument("--duration",type=float,default=1.0)
    a=p.parse_args(); print(json.dumps(main(a.output_dir,a.epochs,a.batch_size,a.seed,a.sample_rate,a.duration),indent=2))

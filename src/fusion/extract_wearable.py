# src/fusion/extract_wearable.py
import numpy as np, torch, pandas as pd, pathlib
from lightning.pytorch import LightningModule

ROOT  = pathlib.Path(__file__).resolve().parents[2]
EMBED_DIM = 128

class EncoderOnly(LightningModule):
    def __init__(self, ckpt):
        super().__init__()
        state = torch.load(ckpt, map_location="cpu")["state_dict"]
        # solo-learn keys look like "backbone.encoder.0.weight"
        self.encoder = torch.nn.Sequential()
        for k, v in state.items():
            if k.startswith("backbone"):
                new_k = k.replace("backbone.", "")
                self.encoder.state_dict()[new_k] = v
        self.encoder.eval()

def main(n_windows=5000, out="data/wearable_embeddings.npy"):
    X = np.load(ROOT/"data/pamap_windows/windows.npy")[:n_windows]   # (N,1000,3)
    enc = EncoderOnly(ROOT/"models/wearable_simclr.ckpt")
    with torch.no_grad():
        embs = []
        for batch in np.array_split(X, len(X)//256):
            t = torch.tensor(batch).permute(0,2,1).float()   # (N,C,T)
            embs.append(enc.encoder(t).numpy())
    embs = np.concatenate(embs, 0)          # (N,128)
    np.save(ROOT/out, embs)
    print("wearable_embeddings.npy:", embs.shape)

if __name__ == "__main__":
    main()

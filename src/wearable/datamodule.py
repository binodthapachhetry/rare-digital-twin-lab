# src/wearable/datamodule.py
import torch, numpy as np
from torch.utils.data import DataLoader, Dataset, random_split
from lightning import LightningDataModule
from torchvision.transforms import Compose
import random

def jitter(x, sigma=0.01): return x + sigma*np.random.randn(*x.shape)
def scale(x, sigma=0.1):   return x * (1 + sigma*np.random.randn(1,1))
def flip_axis(x):          return x * np.array([1,-1,1])   # occasional sign invert

class WearDataset(Dataset):
    def __init__(self, X, aug=False):
        self.X = X; self.aug = aug
    def __len__(self): return len(self.X)
    def _augment(self, x):
        if not self.aug: return x
        funcs = [jitter, scale, flip_axis]
        for f in funcs:
            if random.random() < .5: x = f(x)
        return x
    def __getitem__(self, idx):
        x = self.X[idx]
        x1 = self._augment(x.copy()); x2 = self._augment(x.copy())
        return torch.tensor(x1).permute(1,0), torch.tensor(x2).permute(1,0) # (C,T)

class PamapDataModule(LightningDataModule):
    def __init__(self, path="data/pamap_windows", batch=256):
        super().__init__(); self.path=path; self.batch=batch
    def setup(self, stage=None):
        X = np.load(f"{self.path}/windows.npy")
        N = len(X); train_len = int(.9*N)
        self.train, self.val = random_split(WearDataset(X, aug=True),
                                            [train_len, N-train_len],
                                            generator=torch.Generator().manual_seed(42))
    def train_dataloader(self): return DataLoader(self.train, batch_size=self.batch, shuffle=True, num_workers=4)
    def val_dataloader(self):   return DataLoader(self.val,   batch_size=self.batch, shuffle=False, num_workers=4)

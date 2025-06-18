# src/wearable/linear_eval.py
import torch, lightning as L, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from wearable.datamodule import PamapDataModule
from sklearn.metrics import f1_score, confusion_matrix

class LinearProbe(L.LightningModule):
    def __init__(self, encoder_ckpt):
        super().__init__()
        ckpt = torch.load(encoder_ckpt, map_location='cpu')
        self.encoder = nn.Sequential(*list(ckpt['state_dict'].values())[:])  # hack: load encoder weights
        for p in self.encoder.parameters(): p.requires_grad=False
        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        with torch.no_grad(): h = self.encoder(x)
        return self.fc(h)

    def training_step(self,b,y):
        logits=self(b); loss=nn.CrossEntropyLoss()(logits,y)
        self.log("train_loss",loss); return loss
    def configure_optimizers(self):
        return torch.optim.Adam(self.fc.parameters(), lr=1e-3)

def run_linear():
    dm = PamapDataModule(batch=256); dm.setup()
    # small downstream labelled subset: 50 train / 50 val
    X_lab = np.load('data/pamap_windows/windows.npy')
    y_lab = np.load('data/pamap_windows/labels.npy')
    idx = np.random.choice(len(X_lab), 100, replace=False)
    Xsup, ysup = X_lab[idx], y_lab[idx]
    Ds = TensorDataset(torch.tensor(Xsup).permute(0,2,1).float(),
                       torch.tensor(ysup).long())
    train_loader = DataLoader(Ds[:80], batch_size=32, shuffle=True)
    val_loader   = DataLoader(Ds[80:], batch_size=32)

    model = LinearProbe("models/wearable_simclr.ckpt")
    trainer = L.Trainer(max_epochs=20, log_every_n_steps=5)
    trainer.fit(model, train_loader, val_loader)

    # final eval on hold-out 200 windows
    idx_test = np.random.choice(len(X_lab), 200, replace=False)
    Xt, yt = torch.tensor(X_lab[idx_test]).permute(0,2,1).float(), torch.tensor(y_lab[idx_test])
    preds = torch.argmax(model(Xt),1).cpu().numpy()
    f1 = f1_score(yt, preds)
    print("Frozen-encoder F1:", round(f1,2))

if __name__ == "__main__":
    run_linear()

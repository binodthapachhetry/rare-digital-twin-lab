# src/federated/fl_client.py
import os, torch, echonet, glob, numpy as np
from echonet.models import r2plus1d_18
import torch.nn.functional as F
from torch.utils.data import DataLoader
import flwr as fl

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    mdl = r2plus1d_18(num_classes=1)
    ckpt = torch.load("models/fewshot_best.pt", map_location=DEVICE)
    mdl.load_state_dict(ckpt["state_dict"])
    return mdl.to(DEVICE)

def make_loader(site_path, batch=4):
    files = glob.glob(os.path.join(site_path, "*.avi"))
    ds = echonet.datasets.Echo(files, clips=32, period=2)
    return DataLoader(ds, batch_size=batch, num_workers=4, shuffle=True)

class EchoNetClient(fl.client.NumPyClient):
    def __init__(self, site):
        self.site = site
        self.model = load_model()
        self.loader = make_loader(site)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=1e-4)

    # Flower required methods
    def get_parameters(self, config):      # → list of NumPy
        return [v.cpu().numpy() for v in self.model.state_dict().values()]

    def set_parameters(self, params):
        sd = self.model.state_dict()
        for k, p in zip(sd.keys(), params):
            sd[k] = torch.tensor(p)
        self.model.load_state_dict(sd)

    def fit(self, params, config):
        self.set_parameters(params)
        self.model.train()
        for _ in range(1):                 # one local epoch per round
            for x, y, *_ in self.loader:
                x, y = x.to(DEVICE), y.to(DEVICE)/100
                self.opt.zero_grad()
                pred = self.model(x).squeeze()
                loss = F.l1_loss(pred, y)
                loss.backward(); self.opt.step()
        return self.get_parameters({}), len(self.loader.dataset), {}

    def evaluate(self, params, config):
        self.set_parameters(params)
        self.model.eval()
        maes = []
        with torch.no_grad():
            for x, y, *_ in self.loader:
                x, y = x.to(DEVICE), y.to(DEVICE)/100
                pred = self.model(x).squeeze()
                maes.append(torch.abs(pred-y).cpu().numpy())
        mae = float(np.mean(np.concatenate(maes)))*100
        return mae, len(self.loader.dataset), {"mae": mae}

def start_client(site_dir):
    fl.client.start_numpy_client(server_address="127.0.0.1:8080",
                                 client=EchoNetClient(site_dir))

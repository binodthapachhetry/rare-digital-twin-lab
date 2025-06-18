# src/gnn/gcn_train.py
import torch, torch.nn as nn, torch.nn.functional as F, pathlib, random, numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.metrics import roc_auc_score, classification_report
from torch_geometric.utils import train_test_split_edges
ROOT = pathlib.Path(__file__).resolve().parents[2]

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

class GCN(nn.Module):
    def __init__(self, in_dim, hidden=64, n_classes=2, dropout=0.3):
        super().__init__()
        self.gcn1 = GCNConv(in_dim, hidden)
        self.lin  = nn.Linear(hidden, n_classes)
        self.do = nn.Dropout(dropout)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.gcn1(x, edge_index))
        x = self.do(x)
        return self.lin(x)

def main():
    data: Data = torch.load(ROOT/"data/patient_graph.pt")
    # 25 % hold-out node split
    N = data.num_nodes
    idx = np.random.permutation(N)
    val_idx   = idx[:int(0.15*N)]
    test_idx  = idx[int(0.15*N):int(0.25*N)]
    train_idx = idx[int(0.25*N):]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GCN(in_dim=data.num_node_features).to(device)
    data = data.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-3)

    for epoch in range(201):
        model.train(); opt.zero_grad()
        out = model(data)[train_idx]
        loss = F.cross_entropy(out, data.y[train_idx])
        loss.backward(); opt.step()
        if epoch%40==0:
            model.eval()
            pred = model(data)[val_idx].softmax(1)[:,1].detach().cpu()
            auc = roc_auc_score(data.y[val_idx].cpu(), pred)
            print(f"Epoch {epoch:3d}  train_loss {loss.item():.3f}  val_AUC {auc:.3f}")

    torch.save(model.state_dict(), ROOT/"models/gcn_fusion.pt")
    # final test metrics
    model.eval()
    prob = model(data)[test_idx].softmax(1)[:,1].cpu().numpy()
    y_test = data.y[test_idx].cpu().numpy()
    auc = roc_auc_score(y_test, prob)
    print("\nTest ROC-AUC:", round(auc,3))
    preds = (prob>0.5).astype(int)
    print(classification_report(y_test, preds))

if __name__ == "__main__":
    main()

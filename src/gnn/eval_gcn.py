# src/gnn/eval_gcn.py
import torch, pathlib
from gnn.gcn_train import GCN
ROOT = pathlib.Path(__file__).resolve().parents[2]
data = torch.load(ROOT/"data/patient_graph.pt")
model = GCN(data.num_node_features)
model.load_state_dict(torch.load(ROOT/"models/gcn_fusion.pt", map_location="cpu"))
model.eval()
probs = model(data).softmax(1)[:,1].detach().numpy()
for i,p in enumerate(probs[:10]):   # first 10 patients
    print(f"Patient {i:02d}  P(low-EF)={p:.2f}")

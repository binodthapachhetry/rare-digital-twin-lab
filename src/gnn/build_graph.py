# src/gnn/build_graph.py
"""
Builds k-NN graph from fusion_df.csv and writes:
    data/patient_graph.pt    # PyG Data object
"""
import pandas as pd, numpy as np, torch, pathlib, networkx as nx
from torch_geometric.utils import from_networkx
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

ROOT = pathlib.Path(__file__).resolve().parents[2]

def main(k=5, out="data/patient_graph.pt"):
    df = pd.read_csv(ROOT/"data/fusion_df.csv")
    # -------- 1. feature matrix X  --------
    embed_cols = [c for c in df.columns if c.startswith("E")]
    X = df[embed_cols + ["EF_pred","Age","Sex"]].to_numpy()   # shape (N, 132)
    X = StandardScaler().fit_transform(X)                     # centre/scale
    # -------- 2. build k-NN graph --------
    nn = NearestNeighbors(n_neighbors=k+1, metric="euclidean").fit(X)
    neigh_idx = nn.kneighbors(X, return_distance=False)[:,1:] # skip self (0)
    G = nx.Graph()
    for i in range(len(X)):
        for j in neigh_idx[i]:
            G.add_edge(i, j)
    # -------- 3. convert to PyG ---------
    data = from_networkx(G)
    data.x = torch.tensor(X, dtype=torch.float32)
    data.y = torch.tensor((df["EF_true"] < 40).astype(int).to_numpy(),
                          dtype=torch.long)
    torch.save(data, ROOT/out)
    print("patient_graph.pt:", data)

if __name__ == "__main__":
    main()

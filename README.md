# rare-digital-twin-lab

Building a holistic “digital twin” based on multimodal data.

## Project Structure

```
src/
  wearable/
    datamodule.py
    linear_eval.py
    make_windows.py
    simclr_train.py
  fusion/
    build_df.py
    extract_echo.py
    extract_wearable.py
    train_fusion.py
  gnn/
    build_graph.py
    eval_gcn.py
    gcn_train.py
  federated/
    fl_client.py
    fl_server.py
    split_EchoNet.py
  train.py
  eval.py
env.yaml
notebooks/
  03_late_fusion_ensemble.ipynb
  04_gnn_patient_graph.ipynb
data/
  pamap_windows/
    windows.npy
    labels.npy
  EchoNet_Dynamic/
  wearable_embeddings.npy
  echo_features.csv
  fusion_df.csv
models/
  wearable_simclr.ckpt
  gcn_fusion.pt
  fusion_gbm.pkl
  fewshot_best.pt
output/
  flwr_history.json
```

## Usage Examples

### 1. Wearable Data Preprocessing

Generate sliding windows from raw PAMAP2 data:

```bash
python src/wearable/make_windows.py --src data/PAMAP2_Dataset/Protocol --out data/pamap_windows
```

### 2. Self-Supervised Pretraining (SimCLR)

Train a SimCLR encoder on wearable windows:

```bash
python src/wearable/simclr_train.py
```

### 3. Linear Evaluation

Evaluate frozen encoder with a linear probe:

```bash
python src/wearable/linear_eval.py
```

### 4. Extract Wearable Embeddings

```bash
python src/fusion/extract_wearable.py
```

### 5. Extract EchoNet Features

```bash
python src/fusion/extract_echo.py
```

### 6. Build Fusion DataFrame

```bash
python src/fusion/build_df.py
```

### 7. Train Late Fusion Model

```bash
python src/fusion/train_fusion.py
```

### 8. Build Patient Graph

```bash
python src/gnn/build_graph.py
```

### 9. Train GCN on Patient Graph

```bash
python src/gnn/gcn_train.py
```

### 10. Evaluate GCN

```bash
python src/gnn/eval_gcn.py
```

### 11. Federated Learning

Start server:

```bash
python src/federated/fl_server.py
```

Start client (for each site):

```bash
python src/federated/fl_client.py
```

Split EchoNet data for federated sites:

```bash
python src/federated/split_EchoNet.py
```

## Requirements

Install dependencies:

```bash
conda env create -f env.yaml
conda activate rare-digital-twin-lab
```

## Notes

- Data files and pretrained models are required in the `data/` and `models/` directories, respectively.
- See `notebooks/` for exploratory analysis and additional usage examples.

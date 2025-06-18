# src/wearable/make_windows.py
import pandas as pd, numpy as np, pathlib, pickle, os, argparse

def slice_windows(df, fps=100, win_sec=10, stride_sec=2):
    win, stride = fps*win_sec, fps*stride_sec
    xs, ys = [], []
    for start in range(0, len(df)-win, stride):
        chunk = df.iloc[start:start+win]
        x = chunk[['acc_hand_x', 'acc_hand_y', 'acc_hand_z']].to_numpy().astype('float32')
        y = (chunk['activity_id']==4).any()   # walking window?
        xs.append(x)
        ys.append(int(y))
    return np.stack(xs), np.array(ys)

def main(src_root, out_root):
    src_root, out_root = pathlib.Path(src_root), pathlib.Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    wins, labels = [], []
    for file in src_root.glob("subject*.dat"):
        df = pd.read_csv(file, sep=' ', header=None)
        df = df.rename(columns={1:'timestamp', 4:'acc_hand_x',5:'acc_hand_y',6:'acc_hand_z',1:'activity_id'})
        x, y = slice_windows(df)
        wins.append(x); labels.append(y)
    X = np.concatenate(wins); y = np.concatenate(labels)
    idx = np.random.permutation(len(X))
    np.save(out_root/'windows.npy', X[idx])
    np.save(out_root/'labels.npy',  y[idx])
    print("windows.npy / labels.npy saved:", X.shape, y.shape)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', default='data/PAMAP2_Dataset/Protocol')
    parser.add_argument('--out', default='data/pamap_windows')
    args = parser.parse_args()
    main(args.src, args.out)

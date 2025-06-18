# src/fusion/build_df.py
import pandas as pd, numpy as np, pathlib, json

ROOT = pathlib.Path(__file__).resolve().parents[2]

def main(out="data/fusion_df.csv"):
    echo = pd.read_csv(ROOT/"data/echo_features.csv")
    wearable = np.load(ROOT/"data/wearable_embeddings.npy")[:len(echo)]
    # fake demographics for demo
    np.random.seed(0)
    age  = np.random.randint(30,80,size=len(echo))
    sex  = np.random.choice([0,1], size=len(echo))   # 0 F, 1 M

    df = echo.copy()
    df["Age"]  = age
    df["Sex"]  = sex
    # add embedding columns E0..E127
    for j in range(wearable.shape[1]):
        df[f"E{j}"] = wearable[:,j]
    df["y"] = (df["EF_true"] < 40).astype(int)       # target
    df.to_csv(ROOT/out, index=False)
    print("fusion_df.csv shape:", df.shape)

if __name__ == "__main__":
    main()

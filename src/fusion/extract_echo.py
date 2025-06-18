# src/fusion/extract_echo.py
import pandas as pd, torch, echonet
from echonet.models import r2plus1d_18
import torchvision.transforms as T, pathlib, json, os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ROOT   = pathlib.Path(__file__).resolve().parents[2]   # repo root

def main(split_csv="fewshot_test.csv", out="data/echo_features.csv"):
    # 1. load test list
    test = pd.read_csv(split_csv, header=None, names=["File"])
    # 2. load network
    model = r2plus1d_18(num_classes=1)
    ckpt  = torch.load(ROOT/"models/fewshot_best.pt", map_location=DEVICE)
    model.load_state_dict(ckpt["state_dict"]); model.to(DEVICE).eval()

    transform = T.Compose([
        T.CenterCrop(112),   # default EchoNet settings
        T.Normalize(mean=[0.485], std=[0.229]),
    ])

    rows = []
    for f in test["File"]:
        vid, _, ef_true = echonet.utils.loadvideo(os.path.join(
            ROOT, "data/EchoNet_Dynamic", f))
        # sample middle 32 frames @ 2-frame stride (≈ EchoNet default)
        vid = vid[:, :, :, :112]                  # crop width
        clip = torch.from_numpy(vid).permute(3, 0, 1, 2)  # (T,C,H,W)
        clip = transform(clip/255.).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            ef_pred = model(clip).squeeze().item() * 100   # rescale
        rows.append({"File": f, "EF_true": ef_true, "EF_pred": ef_pred})
    pd.DataFrame(rows).to_csv(out, index=False)
    print("echo_features.csv written:", len(rows))

if __name__ == "__main__":
    main()

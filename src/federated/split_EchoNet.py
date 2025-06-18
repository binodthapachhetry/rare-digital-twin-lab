import pandas as pd, pathlib, random, shutil, os
ROOT = pathlib.Path.cwd()
echodir = ROOT/"data"/"EchoNet_Dynamic"
splits  = ["siteA", "siteB", "siteC"]
for s in splits:
    (ROOT/f"data/{s}").mkdir(parents=True, exist_ok=True)

csv = pd.read_csv(ROOT/"fewshot_train.csv", header=None, names=["File"])
files = csv["File"].tolist()
random.seed(42); random.shuffle(files)

third = len(files)//3
site_map = {"siteA":files[:third],
            "siteB":files[third:2*third],
            "siteC":files[2*third:]}

for site, lst in site_map.items():
    for f in lst:
        src = echodir/f
        dst = ROOT/f"data/{site}/{f}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy(src, dst)      # hard-link if same disk
print({k: len(v) for k,v in site_map.items()})
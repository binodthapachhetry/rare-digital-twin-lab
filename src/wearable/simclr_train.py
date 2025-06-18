# src/wearable/simclr_train.py
import torch, torch.nn as nn
from solo.methods import SimCLR
from solo.args.setup import parse_args_simclr
from wearable.datamodule import PamapDataModule
from lightning.pytorch import Trainer, seed_everything

def main():
    seed_everything(42, workers=True)
    args = parse_args_simclr()
    args.dataset = "custom"          # we supply our own datamodule
    args.backbone = "convnet1d"
    args.batch_size = 256
    args.max_epochs = 50
    args.num_classes = 2             # downstream task placeholder
    args.optimizer = "adam"
    args.lr = 1e-3
    args.data_dir = ""               # unused

    datamodule = PamapDataModule(batch=args.batch_size)
    model = SimCLR(**args.__dict__)

    trainer = Trainer(accelerator="auto", devices=1,
                      max_epochs=args.max_epochs,
                      log_every_n_steps=20, default_root_dir="output/simclr")
    trainer.fit(model, datamodule=datamodule)

    trainer.save_checkpoint("models/wearable_simclr.ckpt")

if __name__ == "__main__":
    main()

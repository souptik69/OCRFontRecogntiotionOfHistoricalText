import os
from argparse import ArgumentParser
import wandb

import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
from pytorch_lightning.loggers import WandbLogger
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset.entityTransformer.ETDataModule import ETDataModule
from model.EntityTransformer.entity_module import EntityModule

from utils.CustomCollate import collate_unet
from utils.metrics.LineDetectionMetricUtils import compute_line_detection_from_batch
from utils.constants import *

print("Cuda available?", torch.cuda.is_available())

def run(args):
    wandb.login(key='37ddeaddf9516e4d369f2a533c11ad51817a0ede')
    os.makedirs(args.save_dir, exist_ok=True)
    wandb_logger = WandbLogger(project=args.name, save_dir=args.save_dir)

    if args.books is not None:
        books = args.books.split(",")

    dataModule = ETDataModule(root_dir=args.root_dir, batch_size=args.batch_size, multiple_books=books,
                              line_height=args.line_height, mode="line")
    dataModule.prepare_data()
    dataModule.setup(stage="test")

    test_data = dataModule.test_dataloader()

    model = EntityModule.load_from_checkpoint(args.et_checkpoint, strict=False)

    trainer = pl.Trainer(gpus=1, logger=wandb_logger)
    trainer.test(model=model, test_dataloaders=test_data)


def parse_arguments():
    args = ArgumentParser()
    args.add_argument('--name', type=str)
    args.add_argument('--root_dir', type=str, default="/cluster/mayr/nbb/vanilla/")
    args.add_argument('--batch_size', type=int, default=16)
    args.add_argument('--n_head', type=int, default=1)
    args.add_argument('--line_height', type=int, default=64)
    args.add_argument('--hidden_size', type=int, default=256)
    args.add_argument('--dropout', type=float, default=0.1)
    args.add_argument('--save_dir', type=str, default="tmp/models/ET_test")
    args.add_argument('--books', type=str, default="Band2,Band3,Band4")
    args.add_argument('--lr', type=float, default=0.0001)
    args.add_argument('--eps', type=float, default=0.4)
    args.add_argument('--noise_teacher_forcing', type=float, default=0.2)
    args.add_argument('--et_checkpoint', type=str, default="tmp/models/et_lre-4_e25.ckpt")
    return args.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    run(args)
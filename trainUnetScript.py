from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from dataset.unet.UnetDataModule import UnetDataModule
from model.unet.UnetModel import UnetModel
from utils.type.LrScheduleType import LrScheduleType

print("Cuda available?", torch.cuda.is_available())

RANDOM_SEED = 47

torch.backends.cudnn.deterministic = True
seed_everything(RANDOM_SEED)


def run(args):
    if args.books is not None:
        books = args.books.split(",")
    dataModule = UnetDataModule(root_dir=args.root_dir, batch_size=args.batch_size, multiple_books=books,
                                pos_weight=args.balance)
    dataModule.prepare_data()
    dataModule.setup()

    train_data = dataModule.train_dataloader()
    val_data = dataModule.val_dataloader()


    pos_weight = dataModule.train_dataset.pos_weight

    model = UnetModel(lr_schedule_type=LrScheduleType.REDUCE_ON_PLATEAU, architecture=args.architecture,
                      balance=pos_weight)

    checkpoint_callback = ModelCheckpoint(dirpath=args.model_dir, filename='unet-{epoch}-{val_loss:.2f}')
    early_stopping = EarlyStopping(monitor="val_loss", patience=10)

    trainer = pl.Trainer(gpus=1, auto_lr_find=True, callbacks=[checkpoint_callback, early_stopping],
                         default_root_dir=args.model_dir)
    trainer.tune(model, train_dataloader=train_data)
    trainer.fit(model, train_data, val_data)

    test_data = dataModule.test_dataloader()
    trainer.test(test_dataloaders=test_data)


def parse_arguments():
    args = ArgumentParser()
    args.add_argument('--root_dir', type=str, default="/cluster/mayr/nbb/vanilla/")
    args.add_argument('--batch_size', type=int, default=8)
    args.add_argument('--model_dir', type=str, default="tmp/models/att_unet_bin_20211223")
    args.add_argument('--books', type=str, default="Band2,Band3,Band4")
    args.add_argument('--architecture', type=str, default="att_unet")
    args.add_argument('--balance', type=bool, default=False)
    return args.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    run(args)

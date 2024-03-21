import os
from argparse import ArgumentParser
import wandb
import pytorch_lightning as pl
import torch
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from dataset.entityTransformer.ETDataModule import ETDataModule
from model.EntityTransformer.entity_module import EntityModule
from utils.alphabet import Alphabet
from utils.type.LrScheduleType import LrScheduleType

print("Cuda available?", torch.cuda.is_available())
#
# RANDOM_SEED = 47
#
# torch.backends.cudnn.deterministic = True
# seed_everything(RANDOM_SEED)


def run(args):
    # logging
    wandb.login(key='37ddeaddf9516e4d369f2a533c11ad51817a0ede')
    os.makedirs(args.save_dir, exist_ok=True)
    wandb_logger = WandbLogger(project=args.name, save_dir=args.save_dir)

    if args.books is not None:
        books = args.books.split(",")
    dataModule = ETDataModule(root_dir=args.root_dir, batch_size=args.batch_size, multiple_books=books,
                                line_height=args.line_height, mode="line", augment=args.augment)
    dataModule.prepare_data()
    dataModule.setup()

    train_data = dataModule.train_dataloader()
    val_data = dataModule.val_dataloader()

    A = Alphabet(dataset="NBB", mode="s2s_recipient")
    model = EntityModule(vocab_size=len(A.toPosition), hidden_size=args.hidden_size, n_head=args.n_head,
                         dropout=args.dropout, lr=args.lr, eps=args.eps, noise_teacher_forcing=args.noise_teacher_forcing)

    checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(args.save_dir,args.name), filename='et-{epoch}',
                                          monitor="val/cer", save_top_k=1, mode="min")
    early_stopping = EarlyStopping(monitor="val/cer", patience=30)

    trainer = pl.Trainer(gpus=1, auto_lr_find=False, callbacks=[checkpoint_callback, early_stopping],
                         logger=wandb_logger)
    trainer.tune(model, train_dataloader=train_data)
    trainer.fit(model, train_data, val_data)


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
    args.add_argument('--augment', action='store_true')
    return args.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    run(args)

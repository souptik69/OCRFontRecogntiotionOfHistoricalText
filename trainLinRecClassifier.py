from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from dataset.img2seq.Img2SeqDataModule import Img2SeqDataModule

from model.LinRecClassifier.LinRecModel import LinRecModel

print("Cuda available?", torch.cuda.is_available())

RANDOM_SEED = 47

torch.backends.cudnn.deterministic = True
seed_everything(RANDOM_SEED)


def run(args):
    data_module = Img2SeqDataModule(img_height=64, batch_size=args.batch_size, use_custom_augmentation=True,
                                           content_file=args.content_file, img_dir=args.img_dir, root_dir=args.root_dir,
                                           books=args.books, recipient_information=True)

    data_module.prepare_data()
    data_module.setup()

    train_data = data_module.train_dataloader()
    val_data = data_module.val_dataloader()
    test_data = data_module.test_dataloader()

    model = LinRecModel(loss_balancing=data_module.complete_dataset.balancing_factor,
                        architecture=args.classifier_architecture, htr_checkpoint_path=args.htr_checkpoint_path)

    checkpoint_callback = ModelCheckpoint(dirpath=args.model_dir, filename='text_class_hth_cnn-{epoch}-{val_f1:.2f}')
    early_stopping = EarlyStopping(monitor="val_loss", patience=50)

    trainer = pl.Trainer(gpus=1, auto_lr_find=True, callbacks=[checkpoint_callback, early_stopping])
    trainer.tune(model, train_dataloader=train_data, val_dataloaders=val_data)
    trainer.fit(model, train_dataloader=train_data, val_dataloaders=val_data)
    trainer.test(test_dataloaders=test_data)


def parse_arguments():
    args = ArgumentParser()
    args.add_argument('--root_dir', type=str, default="/cluster/mayr/nbb/vanilla/")
    args.add_argument('--batch_size', type=int, default=32)
    args.add_argument('--img_dir', type=str, default="")
    args.add_argument('--content_file', type=str, default="gt.txt")
    args.add_argument('--model_dir', type=str, default="tmp/models")
    args.add_argument('--books', type=str, default="Band2,Band3,Band4")
    args.add_argument('--htr_checkpoint_path', type=str, default="wp3/htr/htr_model_backup.ckpt")
    args.add_argument('--classifier_architecture', type=str, default="cnn")
    return args.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    run(args)

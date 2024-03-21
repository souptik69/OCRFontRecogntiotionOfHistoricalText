import os
from argparse import ArgumentParser
import wandb
import pytorch_lightning as pl
import torch
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from dataset.FontDetection.ETDataModule import FontDataModule
from model.EntityTransformer.entity_module import EntityModuleFont
from utils.alphabet import Alphabet
print("Cuda available?", torch.cuda.is_available())

def run(args):
    # logging
    wandb.login(key='5a429a803de01cc018d885bec7b3696b40715445')
    os.makedirs(args.save_dir, exist_ok=True)
    wandb_logger = WandbLogger(project=args.name, save_dir=args.save_dir)
    data_module = FontDataModule(zip_file_path=args.root_dir,image_size=args.image_size, batch_size=args.batch_size, augment=args.augment)
    data_module.prepare_data()
    data_module.setup()
    train_data = data_module.train_dataloader()
    val_data = data_module.val_dataloader()

    A = Alphabet(dataset="NBB", mode="attention")
    font_class_vocabulary = list(range(14))
    model = EntityModuleFont(vocab_size=len(A.toPosition), 
                             font_classes=len(font_class_vocabulary), hidden_size=args.hidden_size,
                             n_head=args.n_head,dropout=args.dropout, lr=args.lr, eps=args.eps, 
                             noise_teacher_forcing=args.noise_teacher_forcing)
    checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(args.save_dir,args.name), filename='et-{epoch}',
                                          monitor="val/total_loss", save_top_k=1, mode="min")
    early_stopping = EarlyStopping(monitor="val/total_loss", patience=10)

    trainer = pl.Trainer(accelerator="gpu",devices= 2,callbacks=[checkpoint_callback, early_stopping],
                         logger=wandb_logger, strategy="ddp")
    # trainer.tune(model, train_dataloader=train_data)
    trainer.fit(model, train_data, val_data)
    
def parse_arguments():
    args = ArgumentParser()
    args.add_argument('--name', type=str, default="SingleMultipleFonts13")
    args.add_argument('--root_dir', type=str, default="/home/hpc/iwi5/iwi5144h/nbb-recipient-detection/SingleMultiple.zip")
    args.add_argument('--batch_size', type=int, default=4)
    args.add_argument('--n_head', type=int, default=2)
    args.add_argument('--image_size', type=int, default=256)
    args.add_argument('--hidden_size', type=int, default=256)
    args.add_argument('--dropout', type=float, default=0.1)
    args.add_argument('--save_dir', type=str, default="/home/hpc/iwi5/iwi5144h/nbb-recipient-detection/SingleMultipleFontModel")
    args.add_argument('--lr', type=float, default=0.0001)
    args.add_argument('--eps', type=float, default=0.4)
    args.add_argument('--noise_teacher_forcing', type=float, default=0.2)
    args.add_argument('--augment', action='store_true')
    return args.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    run(args)
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
    data_module = FontDataModule(zip_file_path=args.root_dir,image_size=args.image_size, batch_size=args.batch_size)
    data_module.prepare_data()
    data_module.setup(stage="test")
    test_data = data_module.test_dataloader()
    model = EntityModuleFont.load_from_checkpoint(args.et_checkpoint, strict=False)
    trainer = pl.Trainer(devices=1, num_nodes=1, logger=wandb_logger)
    trainer.test(model=model, dataloaders= test_data)
    
def parse_arguments():
    args = ArgumentParser()
    args.add_argument('--name', type=str, default="SingleMultipleFonts10")
    args.add_argument('--root_dir', type=str, default="/home/hpc/iwi5/iwi5144h/nbb-recipient-detection/SingleMultiple.zip")
    args.add_argument('--batch_size', type=int, default=16)
    args.add_argument('--n_head', type=int, default=1)
    args.add_argument('--image_size', type=int, default=256)
    args.add_argument('--hidden_size', type=int, default=256)
    args.add_argument('--dropout', type=float, default=0.1)
    args.add_argument('--save_dir', type=str, default="/home/hpc/iwi5/iwi5144h/nbb-recipient-detection/SingleMultiple_test")
    args.add_argument('--lr', type=float, default=0.0001)
    args.add_argument('--eps', type=float, default=0.4)
    args.add_argument('--noise_teacher_forcing', type=float, default=0.2)
    args.add_argument('--et_checkpoint', type=str, default="/home/hpc/iwi5/iwi5144h/nbb-recipient-detection/SingleMultipleFontModel/SingleMultipleFonts13/et-epoch=66.ckpt")
    return args.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    run(args)
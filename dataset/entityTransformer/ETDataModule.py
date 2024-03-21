from typing import Optional, Union, List, Any
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from dataset.entityTransformer.ETDataset import ETDataset
from utils.CustomCollate import collate_s2s, collate_s2s_both


class ETDataModule(pl.LightningDataModule):

    def __init__(self, root_dir:str, xml_dir:str="alto_recipient", batch_size:int=24, multiple_books:List=None,
                 return_coords:bool=False, compute_pos_weight:bool=False, line_height:int=64, img_size:int=256,
                 mode:str="line", augment=False):
        self.root_dir = root_dir
        self.multiple_books = multiple_books
        self.return_coords = return_coords
        self.compute_pos_weight = compute_pos_weight
        self.mode = mode
        self.xml_dir = xml_dir
        self.batch_size = batch_size
        self.line_height = line_height
        self.img_size = img_size
        self.augment = augment

        self.train_dataset: ETDataset = None
        self.val_dataset: ETDataset = None
        self.test_dataset: ETDataset = None

        if mode=="line":
            self.custom_collate = collate_s2s
        elif mode=="page":
            self.custom_collate = None
        elif mode=="both":
            self.custom_collate = collate_s2s_both

    def setup(self, stage: Optional[str] = None):
        if stage != "test":
            self.train_dataset = ETDataset(root_dir=self.root_dir,multiple_books=self.multiple_books, split="train",
                                           return_coords=self.return_coords, compute_pos_weight=self.compute_pos_weight,
                                           mode=self.mode, xml_dir=self.xml_dir, line_height=self.line_height,
                                           img_size=self.img_size, augment=self.augment)
            self.val_dataset = ETDataset(root_dir=self.root_dir, multiple_books=self.multiple_books, split="validation",
                                           return_coords=self.return_coords, compute_pos_weight=self.compute_pos_weight,
                                           mode=self.mode, xml_dir=self.xml_dir, line_height=self.line_height,
                                           img_size=self.img_size)
            print(f"Training samples: {len(self.train_dataset)} \n"
                  + f"Validation samples: {len(self.val_dataset)}")

        else:
            self.test_dataset = ETDataset(root_dir=self.root_dir, multiple_books=self.multiple_books, split="test",
                                           return_coords=self.return_coords, compute_pos_weight=self.compute_pos_weight,
                                           mode=self.mode, xml_dir=self.xml_dir, line_height=self.line_height,
                                           img_size=self.img_size)

            print(f"Test samples: {len(self.test_dataset)}")

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("You need to prepare and setup the data module before loading a dataloader.")

        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8,
                          collate_fn=self.custom_collate)

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        if self.val_dataset is None:
            raise RuntimeError("You need to prepare and setup the data module before loading a dataloader.")

        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8,
                          collate_fn=self.custom_collate)

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        if self.test_dataset is None:
            raise RuntimeError("You need to prepare and setup the data module before loading a dataloader.")

        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8,
                          collate_fn=self.custom_collate)
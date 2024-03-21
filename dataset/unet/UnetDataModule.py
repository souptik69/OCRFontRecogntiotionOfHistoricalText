from typing import Optional, Union, List, Any

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from utils.CustomCollate import collate_unet

from dataset.unet.UnetDataset import UnetDataset


class UnetDataModule(pl.LightningDataModule):

    def __init__(self, root_dir: str = "../../tmp/data_original", batch_size: int = 8, train_percentage: float = 0.8,
                 multiple_books=None, return_coords=False, pos_weight=False):
        self.root_dir = root_dir
        self.train_percentage = train_percentage
        self.batch_size = batch_size
        self.multiple_books = multiple_books
        self.return_coords = return_coords
        self.pos_weight = pos_weight

        self.complete_dataset: UnetDataset = None
        self.train_dataset: UnetDataset = None
        self.valid_dataset: UnetDataset = None
        self.test_dataset: UnetDataset = None

    # def prepare_data(self, *args, **kwargs):
    #     self.complete_dataset = UnetDataset(self.root_dir, multiple_books=self.multiple_books,
    #                                         return_coords=self.return_coords, compute_pos_weight=self.pos_weight)

    def setup(self, stage: Optional[str] = None):
        # train_size = int(self.train_percentage * len(self.complete_dataset))
        # valid_test_size = len(self.complete_dataset) - train_size
        # valid_size = int(valid_test_size/2)
        # test_size = valid_test_size - valid_size

        self.train_dataset =  UnetDataset(self.root_dir, multiple_books=self.multiple_books, split="train",
                                          return_coords=self.return_coords, compute_pos_weight=self.pos_weight)
        self.valid_dataset = UnetDataset(self.root_dir, multiple_books=self.multiple_books, split="validation",
                                          return_coords=self.return_coords, compute_pos_weight=self.pos_weight)
        self.test_dataset = UnetDataset(self.root_dir, multiple_books=self.multiple_books, split="test",
                                          return_coords=self.return_coords, compute_pos_weight=self.pos_weight)

        print(f"Training samples: {len(self.train_dataset)} \n"
              + f"Validation samples: {len(self.valid_dataset)} \n"
              + f"Test samples: {len(self.test_dataset)}")

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("You need to prepare and setup the data module before loading a dataloader.")

        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8)

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        if self.valid_dataset is None:
            raise RuntimeError("You need to prepare and setup the data module before loading a dataloader.")

        return DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        if self.test_dataset is None:
            raise RuntimeError("You need to prepare and setup the data module before loading a dataloader.")

        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8,
                          collate_fn=collate_unet)

    def transfer_batch_to_device(self, batch: Any, device: torch.device) -> Any:
        pass

from typing import Any, Union, List, Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split

from dataset.img2seq.Img2SeqDataset import Img2SeqDataset
from utils.CustomCollate import collate_with_padded_width


class Img2SeqDataModule(pl.LightningDataModule):

    def __init__(self, img_height: int, root_dir: str = "tmp/data_binarized", content_file: str = "train.txt",
                 img_dir: str = "lines_invert", batch_size: int = 64, train_percentage: float = 0.8,
                 use_custom_augmentation: bool = True, books="Band2,Band3", recipient_information=False):
        super().__init__()
        self.image_height = img_height
        self.root_dir, self.img_dir, self.content_file = root_dir, img_dir, content_file
        self.train_percentage = train_percentage
        self.batch_size = batch_size
        self.use_custom_augmentation = use_custom_augmentation
        self.books = books
        self.recipient_information = recipient_information

        self.complete_dataset: Img2SeqDataset = None
        self.train_dataset: Img2SeqDataset = None
        self.valid_dataset: Img2SeqDataset = None
        self.test_dataset: Img2SeqDataset = None

    def prepare_data(self, *args, **kwargs):
        self.complete_dataset = Img2SeqDataset(self.root_dir, self.img_dir, self.content_file, self.image_height,
                                               self.use_custom_augmentation, books=self.books, recipient_information=self.recipient_information)

    def setup(self, stage: Optional[str] = None):
        train_size = int(self.train_percentage * len(self.complete_dataset))
        valid_test_size = len(self.complete_dataset) - train_size
        valid_size = int(valid_test_size/2)
        test_size = valid_test_size - valid_size

        self.train_dataset, self.valid_dataset, self.test_dataset = \
            random_split(self.complete_dataset, [train_size, valid_size, test_size])

        print(f"Complete sample size: {len(self.complete_dataset)} \n"
              + f"Training samples: {len(self.train_dataset)} \n"
              + f"Validation samples: {len(self.valid_dataset)} \n"
              + f"Test samples: {len(self.test_dataset)}")

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("You need to prepare and setup the data module before loading a dataloader.")

        return DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=collate_with_padded_width,
                          num_workers=8, shuffle=True)

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        if self.train_dataset is None:
            raise RuntimeError("You need to prepare and setup the data module before loading a dataloader.")

        return DataLoader(self.valid_dataset, batch_size=self.batch_size, collate_fn=collate_with_padded_width,
                          num_workers=8, shuffle=False)

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        if self.train_dataset is None:
            raise RuntimeError("You need to prepare and setup the data module before loading a dataloader.")

        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,
                          collate_fn=collate_with_padded_width, num_workers=8)

    def transfer_batch_to_device(self, batch: Any, device: torch.device) -> Any:
        pass

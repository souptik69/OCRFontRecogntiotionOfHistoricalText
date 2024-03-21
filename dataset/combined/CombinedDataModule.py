from typing import Optional, Union, List, Any

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split

from dataset.combined.CombinedDataset import CombinedDataset
from utils.CustomCollate import collate_combined


class CombinedDataModule(pl.LightningDataModule):

    def __init__(self, root_dir: str = "../../tmp/dataset/vanilla", pos_weight=False, train_percentage=0.8):
        super().__init__()
        self.root_dir = root_dir
        self.pos_weight = pos_weight
        self.train_percentage = train_percentage

        # only for testing
        self.complete_dataset: CombinedDataset = None

    def prepare_data(self, *args, **kwargs):
        self.complete_dataset = CombinedDataset(self.root_dir, compute_pos_weight=self.pos_weight)

    def setup(self, stage: Optional[str] = None):
        train_size = int(self.train_percentage * len(self.complete_dataset))
        valid_test_size = len(self.complete_dataset) - train_size
        valid_size = int(valid_test_size / 2)
        test_size = valid_test_size - valid_size

        self.train_dataset, self.valid_dataset, self.test_dataset = \
            random_split(self.complete_dataset, [train_size, valid_size, test_size])

        print(f"Complete sample size: {len(self.complete_dataset)} \n"
              + f"Training samples: {len(self.train_dataset)} \n"
              + f"Validation samples: {len(self.valid_dataset)} \n"
              + f"Test samples: {len(self.test_dataset)}")

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        if self.test_dataset is None:
            raise RuntimeError("You need to prepare and setup the data module before loading a dataloader.")

        # batch size needs to be fixed to 1
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False, collate_fn=collate_combined)

    def transfer_batch_to_device(self, batch: Any, device: torch.device) -> Any:
        pass

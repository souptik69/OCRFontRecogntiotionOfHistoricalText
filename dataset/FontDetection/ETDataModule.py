import torch 
from torch.utils.data import DataLoader
from typing import Optional, Union, List
import pytorch_lightning as pl
from dataset.FontDetection.ETDataset import FontDataset, FontDatasetSingle
from utils.CustomCollate import collate_s2s_font,collate_s2s_singlefont

class FontDataModule(pl.LightningDataModule):
    def __init__(self, zip_file_path,image_size = 256, batch_size:int=16, augment=False):
        self.zip_file_path = zip_file_path
        # self.line_height = line_height
        self.image_size = image_size
        self.batch_size = batch_size
        self.augment = augment
        
        self.train_dataset: FontDataset = None
        self.val_dataset: FontDataset = None
        self.test_dataset: FontDataset = None
        self.custom_collate = collate_s2s_font
    
    def setup(self, stage: Optional[str]= None):
        if stage != "test":
            self.train_dataset = FontDataset(zip_file_path=self.zip_file_path, 
                                             image_size=self.image_size,split ="train", 
                                             augment=self.augment)
            self.val_dataset = FontDataset(zip_file_path=self.zip_file_path, 
                                             image_size=self.image_size,split ="valid", 
                                             augment=self.augment)
            print(f"Training samples: {len(self.train_dataset)} \n"
                  + f"Validation samples: {len(self.val_dataset)}")
        
        else :
            self.test_dataset = FontDataset(zip_file_path=self.zip_file_path, 
                                             image_size=self.image_size,split ="test", 
                                             augment=self.augment)
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
    

class FontDataModuleSingle(pl.LightningDataModule):
    def __init__(self, zip_file_path,image_size = 256, batch_size:int=16, augment=False):
        self.zip_file_path = zip_file_path
        self.image_size = image_size
        self.batch_size = batch_size
        self.augment = augment
        
        self.train_dataset: FontDatasetSingle = None
        self.val_dataset: FontDatasetSingle = None
        self.test_dataset: FontDatasetSingle = None
        self.custom_collate = collate_s2s_singlefont
    
    def setup(self, stage: Optional[str]= None):
        if stage != "test":
            self.train_dataset = FontDatasetSingle(zip_file_path=self.zip_file_path, 
                                             image_size=self.image_size,split ="train", 
                                             augment=self.augment)
            self.val_dataset = FontDatasetSingle(zip_file_path=self.zip_file_path, 
                                             image_size=self.image_size,split ="valid", 
                                             augment=self.augment)
            print(f"Training samples: {len(self.train_dataset)} \n"
                  + f"Validation samples: {len(self.val_dataset)}")
        
        else :
            self.test_dataset = FontDatasetSingle(zip_file_path=self.zip_file_path, 
                                             image_size=self.image_size,split ="test", 
                                             augment=self.augment)
            print(f"Test samples: {len(self.test_dataset)}")

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("You need to prepare and setup the data module before loading a dataloader.")

        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=32,
                          collate_fn=self.custom_collate)

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        if self.val_dataset is None:
            raise RuntimeError("You need to prepare and setup the data module before loading a dataloader.")

        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=32,
                          collate_fn=self.custom_collate)

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        if self.test_dataset is None:
            raise RuntimeError("You need to prepare and setup the data module before loading a dataloader.")

        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8,
                          collate_fn=self.custom_collate)
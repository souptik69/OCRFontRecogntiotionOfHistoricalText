import os
import zipfile
from io import BytesIO
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from dataset.transforms.Binarize import Binarize
from utils.constants import *
from torchvision.transforms import Resize, ToTensor, ToPILImage
from dataset.augmentation.composed_data_augmentation import ComposedDataAugmentation
from utils.clean_txt import clean_string
from utils.FixedHeight import FixedHeightResize
import gc
class FontDataset(Dataset):
    def __init__(self, zip_file_path, image_size=256, split="train", augment=False):
        self.zip_file_path = zip_file_path
        self.image_size = image_size
        self.data = []

        if split not in ["train", "valid", "test"]:
            raise ValueError("Invalid split. Use 'train', 'valid', or 'test'.")

        data_path = os.path.join("SingleMultiple", split, "multiple")
        # data_path = os.path.join("data", split, "single","textura")

        with zipfile.ZipFile(self.zip_file_path, 'r') as zip_ref:
            image_files = [f for f in zip_ref.namelist() if f.startswith(data_path)]

            for img_filename in image_files:
                img_name, ext = os.path.splitext(img_filename)
                numerical_name = img_name.split('/')[-1]

                if ext == '.jpg':
                    img_data = zip_ref.read(img_filename)
                    txt_filename = f"SingleMultiple/{split}/multiple/{numerical_name}.txt"
                    pf_filename = f"SingleMultiple/{split}/multiple/{numerical_name}.pf"
                    cf_filename = f"SingleMultiple/{split}/multiple/{numerical_name}.cf"

                    # txt_filename = f"data/{split}/single/textura/{numerical_name}.txt"
                    # pf_filename = f"data/{split}/single/textura/{numerical_name}.pf"
                    # cf_filename = f"data/{split}/single/textura/{numerical_name}.cf"

                    if txt_filename in zip_ref.namelist() and pf_filename in zip_ref.namelist() and cf_filename in zip_ref.namelist():
                        text_data = zip_ref.read(txt_filename).decode('utf-8')
                        # text_data = clean_string(text_data)

                        with BytesIO(zip_ref.read(pf_filename)) as pf_buffer:
                            pf_array = np.load(pf_buffer, allow_pickle=True)

                        with BytesIO(zip_ref.read(cf_filename)) as cf_buffer:
                            cf_array = np.load(cf_buffer, allow_pickle=True)

                        img = Image.open(BytesIO(img_data))
                        binarizer = Binarize()
                        img = Resize(size=[self.image_size, self.image_size])(binarizer(img.convert("L")))
                        self.data.append((img, text_data, cf_array))

        if augment and split == "train":
            print("augmenting")
            self.line_transform = transforms.Compose([ComposedDataAugmentation(), transforms.ToTensor()])
        else:
            self.line_transform = transforms.ToTensor()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        line_img, text, cf = self.data[index]
        line_img = self.line_transform(line_img)
        return {"LINE_IMAGE": line_img, "TEXT": text, "CF": cf}



class FontDatasetSingle(Dataset):
    def __init__(self, zip_file_path,image_size = 256, split="train", augment=False):
        self.zip_file_path = zip_file_path
        self.image_size = image_size
        self.content_infos = []
        self.data = []

        if split in ["train", "valid", "test"]:
            data_path = os.path.join(split, "single")
            # print(data_path)  # Folder structure within the zip file
        else:
            raise ValueError("Invalid split. Use 'train', 'valid', or 'test'.")
        
        with zipfile.ZipFile(self.zip_file_path, 'r') as zip_ref:
            image_files = [f for f in zip_ref.namelist() if f.startswith(data_path)]
            image_groups = {}  # Dictionary to group corresponding files

            for img_filename in image_files:
                img_name, ext = os.path.splitext(img_filename)
                numerical_name = img_name.split('/')[-1]  # Extract numerical name
                if numerical_name not in image_groups:
                    image_groups[numerical_name] = {}
                if ext == '.jpg':
                    image_groups[numerical_name]['jpg'] = zip_ref.read(img_filename)
                elif ext == '.pf':
                    image_groups[numerical_name]['pf'] = zip_ref.read(img_filename)
                # break

            for index, numerical_name in enumerate(image_groups):
                group = image_groups[numerical_name]
                if 'jpg' in group and 'pf' in group:
                    content_info = self._UContentInfo(index, group['jpg'],group['pf'])
                    self.content_infos.append(content_info)
        
        
        self.data += self._preload_lines()

        self.toTensor = transforms.ToTensor()

        if augment==True and split == "train":
            print("augmenting")
            self.line_transform = transforms.Compose([ComposedDataAugmentation(),
                                                      transforms.ToTensor()])
        else:
            self.line_transform = transforms.ToTensor()                    

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        # toTensor transforms and augmentations
        line_img, pf = self.data[index]
        line_img = self.line_transform(line_img)
        return {"LINE_IMAGE": line_img, "PF": pf}
    
    def _UContentInfo(self, index, img_data, pf_data):
        img = Image.open(BytesIO(img_data))  # Open image from bytes
        with BytesIO(pf_data) as pf_buffer:
            pf_array = np.load(pf_buffer, allow_pickle=True)

        return {
            "index": index,
            "img": img,
            "PF": pf_array,
        }

    def _preload_lines(self):
        resize = Resize(size=[self.image_size,self.image_size])
        lines = list()
        for content_info in self.content_infos:
            image = content_info["img"]
            index = content_info["index"]
            pf = content_info["PF"]
            bin = Binarize()
            image = bin(image.convert("L"))
            image = resize(image)
            lines.append((image,pf))
        return lines

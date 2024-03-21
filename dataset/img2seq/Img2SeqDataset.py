import os.path

import xml.etree.ElementTree as et

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
import os.path as path

from dataset.img2seq.Img2SeqItem import Img2SeqItem
from dataset.transforms.ComposedDataAugmentation import ComposedDataAugmentation
from dataset.transforms.FixedHeightResize import FixedHeightResize
from dataset.transforms.Binarize import Binarize
from utils.alphabet import Alphabet
from utils.constants import *


class Img2SeqDataset(Dataset):

    # TODO: preload everything into memory
    def __init__(self, root_dir: str, img_dir: str, content_file: str, fixed_img_height: int,
                 use_custom_augmentation: bool, books="Band2,Band3", recipient_information=False):

        self.alphabet = Alphabet(dataset="NBB")
        self.img2seq_items: Img2SeqItem = []
        self.root_dir, self.img_dir = root_dir, img_dir
        self.max_content_len = 0
        self.fixed_img_height = fixed_img_height

        with open(path.join(self.root_dir, content_file), encoding='utf-8') as cf:
            content = cf.readlines()

        balancing_factor = 0
        for c in content:
            img_file_name, content = c.split(" ", 1)
            if img_file_name[:5] not in books.split(","):
                continue
            # load rec info
            if recipient_information:
                recipient = self.get_recipient_information(img_file_name, root_dir)
            else:
                recipient = False
            if recipient == -1:
                continue
            elif recipient:
                balancing_factor += 1
            content = content.rstrip("\n")
            if len(content) == 0 or "Latein".lower() in content.lower():
                continue
            img2seq_item = Img2SeqItem(path.splitext(img_file_name)[0], content, recipient)
            if len(img2seq_item.content) > 0:
                self.img2seq_items.append(img2seq_item)

        if recipient_information:
            self.balancing_factor = torch.Tensor([1 / (balancing_factor / len(self.img2seq_items))])
        self.compute_dataset_params()

        if use_custom_augmentation:
            self.transforms = transforms.Compose([Binarize(), ComposedDataAugmentation(), FixedHeightResize(self.fixed_img_height),
                                                  transforms.ToTensor()])
        else:
            self.transforms = transforms.Compose([Binarize(), FixedHeightResize(self.fixed_img_height), transforms.ToTensor()])

    def __len__(self):
        # return 1000
        return len(self.img2seq_items)

    def __getitem__(self, index):
        img2seq_item: Img2SeqItem = self.img2seq_items[index]

        with Image.open(path.join(self.root_dir, self.img_dir, f"{img2seq_item.img_file_name}.png")).convert("L") as im:
            transformed_image = self.transforms(im)

        data = {IMAGE: transformed_image,
                TEXT: img2seq_item.content, # just to evaluate
                UNPADDED_TEXT_LEN: len(img2seq_item.content),
                RECIPIENT: img2seq_item.recipient}
        return data

    def get_recipient_information(self, line_name, root_dir):
        line_name = os.path.splitext(line_name)[0]
        path_to_line, full_name = os.path.split(line_name)
        book, _ = os.path.split(path_to_line)
        parts = full_name.split("_")
        img, id = "_".join(parts[:-2]), "_".join(parts[-2:])
        img_parts = img.split("-")
        img = " ".join([img_parts[0], "-".join(img_parts[1:-1]), img_parts[-1]])
        xml_file = os.path.join(root_dir, book, "alto", "{}.xml".format(img))
        if os.path.isfile(xml_file) == False:
            return -1
        xml_root = et.parse(xml_file).getroot()
        for text_line_tag in xml_root.iter("{http://www.loc.gov/standards/alto/ns-v2#}TextLine"):
            string_id = text_line_tag.find("{http://www.loc.gov/standards/alto/ns-v2#}String").attrib["ID"]
            if string_id == id:
                return True if text_line_tag.attrib["RECIPIENT"] == "True" else False

    def compute_dataset_params(self):
        print("Computing dataset params ...")
        for img2seq_item in self.img2seq_items:

            content_length = len(img2seq_item.content)
            if content_length > self.max_content_len:
                self.max_content_len = content_length

        print("Calculated maximal content length:", self.max_content_len)

import os

import torch
from PIL import Image
from torchvision.transforms import Resize, transforms, ToTensor, Grayscale
from pathlib import Path

from dataset.transforms.Binarize import Binarize
from dataset.transforms.FixedHeightResize import FixedHeightResize


class CombinedLineInfo:

    def __init__(self, text_line_tag, png_file_name, index, root_dir, fixed_img_height):
        self.index = index
        string_tag = text_line_tag.find("{http://www.loc.gov/standards/alto/ns-v2#}String")
        self.root_dir = root_dir
        self.fixed_img_height = fixed_img_height

        self.content = string_tag.attrib["CONTENT"]
        self.unpadded_text_len = len(self.content)

        self.is_recipient = True if text_line_tag.attrib["RECIPIENT"] == "True" else False

        png_file_path = f"{png_file_name}_{string_tag.attrib['ID']}".replace(" ", "-")
        book, file_name = os.path.split(os.path.splitext(png_file_path)[0])

        self.line_image_path = f"{os.path.join(self.root_dir, book, 'lines', file_name)}.png"
        self.line_image = None
        self.unpadded_image_with = None

        self.height = int(string_tag.attrib["HEIGHT"])
        self.width = int(string_tag.attrib["WIDTH"])
        self.hpos = int(string_tag.attrib["HPOS"])
        self.vpos = int(string_tag.attrib["VPOS"])


    def __repr__(self):
        return f"""{self.content}
Image file: {self.png_file_name}
Recipient: {self.is_recipient}
Heigth: {self.height} - Width: {self.width}
HPos: {self.hpos} - VPos: {self.vpos}"""

    def load_image(self):

        with Image.open(self.line_image_path).convert("L") as im:
            trans = transforms.Compose([Binarize(), FixedHeightResize(self.fixed_img_height), transforms.ToTensor()])
            self.line_image = trans(im)
            self.unpadded_image_with = self.line_image.shape[2]


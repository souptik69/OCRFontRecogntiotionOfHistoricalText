import os
import xml.etree.ElementTree as et
from typing import List

from PIL import Image
from torchvision.transforms import transforms, Resize, ToTensor

from .pageLineEntry import PageLineEntry


class PageEntry:

    def __init__(self, base_path, xml_file, image_file_name, book, img_size=256):
        xml_file_path = os.path.join(base_path, book, "alto", xml_file)
        xml_root = et.parse(xml_file_path).getroot()
        self.image_file = image_file_name
        self.base_path = base_path
        self.book = book
        self.page = self.image_file.split("Nr")[-1].strip().split(".")[0]
        self.page_line_entries: List[PageLineEntry] = []

        for i, text_line_tag in enumerate(xml_root.iter("{http://www.loc.gov/standards/alto/ns-v4#}TextLine")):
            page_slice_entry = PageLineEntry(self.base_path, self.image_file, text_line_tag, book, i)
            self.page_line_entries.append(page_slice_entry)

        self.transforms = transforms.Compose([Resize(size=[img_size, img_size]), ToTensor()])
        self.image = None
        self.original_size = None

    def compute(self, visualization=False, connection=None):
        if visualization:
            for line in self.page_line_entries:
                line.compute(visualization, connection)
        else:
            image = Image.open(f"{self.base_path}/{self.book}/{self.image_file}")
            self.original_size = image.size
            self.image = self.transforms(image).unsqueeze(0)


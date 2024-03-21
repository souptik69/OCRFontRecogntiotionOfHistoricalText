from PIL import Image
from torchvision.transforms import transforms, ToTensor, ToPILImage

from .Binarize import Binarize
from .FixedHeightResize import FixedHeightResize
from .database_helper import select_line_info

class PageLineEntry:
    def __init__(self, base_path, page_file_name, text_line_tag, book, line_number, fixed_img_height=64):
        self.book = book
        self.base_path = base_path
        self.page_file_name = page_file_name
        self.page = page_file_name.split("Nr")[-1].strip().split(".")[0]
        self.fixed_img_height = fixed_img_height
        self.line_number = line_number

        self.hpos = int(text_line_tag.attrib["HPOS"])
        self.vpos = int(text_line_tag.attrib["VPOS"])
        self.height = int(text_line_tag.attrib["HEIGHT"])
        self.width = int(text_line_tag.attrib["WIDTH"])
        self.id = text_line_tag.attrib["ID"]

        self.transforms = transforms.Compose([Binarize(), FixedHeightResize(self.fixed_img_height), ToTensor()])
        self.image = None

        self.is_recipient = None

    def compute(self, visualization=True, connection=None):
        if visualization:

            if connection is None:
                raise Exception("Connection can not be none!")

            row = select_line_info(connection, self)
            self.is_recipient = row[0]
        else:
            self.image = self.transforms(self._get_slice()).unsqueeze(0)

    def _get_slice(self):
        with Image.open(f"{self.base_path}/{self.book}/{self.page_file_name}").convert("L") as image:
            image_tensor = ToTensor()(image)
            slice = image_tensor[:, self.vpos:(self.vpos + self.height), self.hpos: (self.hpos + self.width)]
            return ToPILImage()(slice)

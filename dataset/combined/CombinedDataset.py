import os
import xml.etree.ElementTree as et

import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import Resize, ToTensor, ToPILImage

from dataset.combined.CombinedLineInfo import CombinedLineInfo
from dataset.transforms.ApplyRecipientInfoToMask import ApplyRecipientInfoToMask
from utils.alphabet import Alphabet
from utils.constants import MASK, IMAGE, CONTENT_INFOS, ORIGINAL_IMAGE_SIZE, BLANK, PAD, START_OF_SEQUENCE, \
    END_OF_SEQUENCE, ORIGINAL_IMAGE
from tqdm import tqdm


class CombinedDataset(Dataset):
    def __init__(self, root_dir, xml_dir="alto", img_size=256, line_info_img_height=64, compute_pos_weight=False):
        self.root_dir = root_dir
        self.xml_dir = xml_dir
        self.img_size = img_size

        self.alphabet = Alphabet(dataset="NBB")

        # data is dict of type png_file_name --> content infos []
        self.data = []
        self.all_content_infos = []

        xml_file_names = []
        for book in ["Band2", "Band3", "Band4"]:
            xml_file_names += [os.path.join(book, xml_dir, f) for f in
                               os.listdir(os.path.join(root_dir, book, xml_dir))]

        for xml_file_name in xml_file_names:
            img_path, file_name = os.path.split(os.path.splitext(xml_file_name)[0])
            book = os.path.split(os.path.split(img_path)[0])[1]
            png_file_name = os.path.join(book, file_name)
            xml_root = et.parse(os.path.join(root_dir, xml_file_name)).getroot()

            content_infos = []

            for i, text_line in enumerate(xml_root.iter("{http://www.loc.gov/standards/alto/ns-v2#}TextLine")):
                content_info = CombinedLineInfo(text_line, png_file_name, i, self.root_dir, line_info_img_height)

                # there are some infos without a img in the lines directory
                content_infos.append(content_info)
                # needed to compute dataset params
                self.all_content_infos.append(content_info)

            self.data.append((png_file_name, content_infos))

        if compute_pos_weight:
            self.pos_weight = self._get_pos_weight()
        else:
            self.pos_weight = torch.Tensor([1])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        file_name, content_infos = self.data[index]
        image = Image.open(f"{os.path.join(self.root_dir, file_name)}.jpg")

        # create target mask
        mask = torch.zeros((1, image.size[1], image.size[0]))
        recipient_content_infos = [info for info in content_infos if info.is_recipient]

        image_transforms = transforms.Compose([Resize(size=[self.img_size, self.img_size]), ToTensor()])
        mask_transforms = transforms.Compose([ApplyRecipientInfoToMask(recipient_content_infos), ToPILImage(),
                                              Resize(size=[self.img_size, self.img_size]), ToTensor()])

        image_output = image_transforms(image)
        mask_output = mask_transforms(mask)

        # get and transform all images for corresponding
        for ci in content_infos:
            ci.load_image()

        return {IMAGE: image_output, MASK: mask_output, ORIGINAL_IMAGE: image,
                ORIGINAL_IMAGE_SIZE: image.size, CONTENT_INFOS: content_infos}

    def _get_pos_weight(self):
        pos_weight = []
        print("computing pos weight")
        for i in tqdm(range(len(self.data))):
            item = self.__getitem__(i)
            pos_weight.append(torch.mean(item[MASK]))
        pos_weight = torch.mean(torch.Tensor(pos_weight))
        return torch.true_divide(1, pos_weight)

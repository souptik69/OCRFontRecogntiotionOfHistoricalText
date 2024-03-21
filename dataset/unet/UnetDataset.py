import os
import xml.etree.ElementTree as et

import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import Resize, ToTensor, ToPILImage

from dataset.transforms.ApplyRecipientInfoToMask import ApplyRecipientInfoToMask
from dataset.transforms.Binarize import Binarize
from dataset.unet.UnetContentInfo import UnetContentInfo
from utils.constants import MASK, IMAGE
from utils.efficiency import load
from tqdm import tqdm


class UnetDataset(Dataset):
    def __init__(self, root_dir, xml_dir="alto_recipient", img_size=256, multiple_books=None, return_coords=False,
                 compute_pos_weight=False, split="test"):
        self.root_dir = root_dir
        self.xml_dir = xml_dir
        self.img_size = img_size
        self.content_infos = []
        self.return_coords = return_coords
        if split in ["train","validation"]:
            split_filenames = load(os.path.join(root_dir, "NBB_splits.pkl.gz"))[split]
        else:
            all_files = load(os.path.join(root_dir, "NBB_splits.pkl.gz"))
            split_filenames = list()
            for k in [k for k in all_files.keys() if "test" in k]:
                split_filenames += all_files[k]
        # data is dict of type png_file_name --> content infos []
        self.data = []

        if multiple_books is None:
            xml_file_names = [os.path.join(xml_dir, f) for f in os.listdir(os.path.join(root_dir, xml_dir))]
        else:
            xml_file_names = []
            for book in multiple_books:
                xml_file_names += [os.path.join(book, xml_dir, f) for f in
                                   os.listdir(os.path.join(root_dir, book, xml_dir))]

        xml_file_names = [n for n in xml_file_names if os.path.split(os.path.splitext(n)[0])[1] in split_filenames]
        for xml_file_name in tqdm(xml_file_names):
            if multiple_books is None:
                png_file_name = os.path.split(os.path.splitext(xml_file_name)[0])[1]
            else:
                img_path, file_name = os.path.split(os.path.splitext(xml_file_name)[0])
                book = os.path.split(os.path.split(img_path)[0])[1]
                png_file_name = os.path.join(book, file_name)
            xml_root = et.parse(os.path.join(root_dir, xml_file_name)).getroot()

            content_infos = []

            for i, text_line in enumerate(xml_root.iter("{http://www.loc.gov/standards/alto/ns-v2#}TextLine")):
                content_info = UnetContentInfo(text_line, png_file_name, i)
                content_infos.append(content_info)

            # self.data.append((png_file_name, content_infos))
            image, mask, image_size = self._preload_image_and_mask(png_file_name, content_infos)
            self.data.append((image, image_size, mask, content_infos))
        if compute_pos_weight:
            self.pos_weight = self._get_pos_weight()
        else:
            self.pos_weight = torch.Tensor([1])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # file_name, content_infos = self.data[index]
        # image = Image.open(f"{os.path.join(self.root_dir, file_name)}.jpg")
        #
        # # create target mask
        # mask = torch.zeros((1, image.size[1], image.size[0]))
        # recipient_content_infos = [info for info in content_infos if info.is_recipient]
        #
        # image_transforms = transforms.Compose([Resize(size=[self.img_size, self.img_size]), ToTensor()])
        # mask_transforms = transforms.Compose([ApplyRecipientInfoToMask(recipient_content_infos), ToPILImage(),
        #                                       Resize(size=[self.img_size, self.img_size]), ToTensor()])
        #
        # image_output = image_transforms(image)
        # mask_output = mask_transforms(mask)

        image_output, image_size, mask_output, content_infos =  self.data[index]
        if self.return_coords:
            return {IMAGE: image_output, MASK: mask_output, "original_height": image_size[1],
                    "original_width": image_size[0],
                    "content_infos": content_infos}
        else:
            return {IMAGE: image_output, MASK: mask_output}

    def _preload_image_and_mask(self, file_name, content_infos):
        bin = Binarize()
        image = bin(Image.open(f"{os.path.join(self.root_dir, file_name)}.jpg").convert("L"))

        # create target mask
        mask = torch.zeros((1, image.size[1], image.size[0]))
        recipient_content_infos = [info for info in content_infos if info.is_recipient]

        image_transforms = transforms.Compose([Resize(size=[self.img_size, self.img_size]), ToTensor()])
        mask_transforms = transforms.Compose([ApplyRecipientInfoToMask(recipient_content_infos), ToPILImage(),
                                              Resize(size=[self.img_size, self.img_size]), ToTensor()])

        image_output = image_transforms(image)
        mask_output = mask_transforms(mask)
        return image_output, mask_output, image.size

    def _get_pos_weight(self):
        pos_weight = []
        print("computing pos weight")
        for i in tqdm(range(len(self.data))):
            item = self.__getitem__(i)
            pos_weight.append(torch.mean(item[MASK]))
        pos_weight = torch.mean(torch.Tensor(pos_weight))
        return torch.true_divide(1, pos_weight)

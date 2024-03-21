import os
from tqdm import tqdm
import xml.etree.ElementTree as et
from PIL import Image
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision.transforms import Resize, ToTensor, ToPILImage

from dataset.augmentation.composed_data_augmentation import ComposedDataAugmentation
from dataset.transforms.ApplyRecipientInfoToMask import ApplyRecipientInfoToMask
from dataset.transforms.Binarize import Binarize
from dataset.unet.UnetContentInfo import UnetContentInfo
from utils.constants import *
from utils.CustomCollate import collate_s2s, collate_s2s_both
from utils.efficiency import load
from utils.FixedHeight import FixedHeightResize

class ETDataset(Dataset):
    """
    mode = {"line","page","both"}
    """
    def __init__(self, root_dir, xml_dir="alto_recipient", img_size=256, line_height=64, multiple_books=None, mode="line", return_coords=False, compute_pos_weight=False,
                 split="test", augment=False):
        self.root_dir = root_dir
        self.mode = mode
        self.xml_dir = xml_dir
        self.img_size = img_size
        self.line_height = line_height
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


            image_unet, mask, image, image_size = self._preload_image_and_mask(png_file_name, content_infos)
            if mode=="page":
                self.data.append((image_unet, image_size, mask, content_infos))
            elif mode in ["line","both"]:
                if mode=="line":
                    self.data += self._preload_lines(image,content_infos)
                else:
                    self.data.append((image, image_unet, image_size, mask, content_infos, self._preload_lines(image,content_infos)))

            # if len(self.data) > 2:
            #     break

        if compute_pos_weight:
            self.pos_weight = self._get_pos_weight()
        else:
            self.pos_weight = torch.Tensor([1])

        # transforms
        self.toTensor = transforms.ToTensor()

        if augment and split == "train":
            print("augmenting")
            self.line_transform = transforms.Compose([ComposedDataAugmentation(),
                                                      transforms.ToTensor()])
        else:
            self.line_transform = transforms.ToTensor()


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # toTensor transforms and augmentations
        if self.mode=="page":
            image_output, image_size, image, mask_output, content_infos = self.data[index]
            image_output = self.toTensor(image_output)
            mask_output = self.toTensor(mask_output)
            if self.return_coords:
                return {IMAGE: image_output, MASK: mask_output, "original_height": image_size[1],
                        "original_width": image_size[0],
                        "content_infos": content_infos}
            else:
                return {IMAGE: image_output, MASK: mask_output}
        elif self.mode=="line":
            line_img, content, is_recipient, file_name = self.data[index]
            line_img = self.line_transform(line_img)
            # y to torch LongTensor
            return {LINE_IMAGE: line_img, TEXT: content, RECIPIENT: is_recipient}
        elif self.mode=="both":
            image, image_unet, image_size, mask_unet, content_infos, line_data = self.data[index]
            image_unet, mask_unet = self.toTensor(image_unet), self.toTensor(mask_unet)
            line_batch = []
            for line in line_data:
                line_batch.append({LINE_IMAGE: self.line_transform(line[0]), TEXT: line[1], RECIPIENT: line[2]})
            line_batch = collate_s2s(line_batch)
            return {IMAGE: image_unet, MASK: mask_unet, "content_infos": content_infos, "original_image": image,
                    "line_batch": line_batch}

    # helpers
    def _preload_lines(self, image, content_infos):
        resize = FixedHeightResize(height=self.line_height)
        lines = list()
        for c in content_infos:
            # crop out image
            area = (c.hpos,c.vpos,c.hpos+c.width,c.vpos+c.height)
            line_img = resize(image.crop(area))
            lines.append((line_img,c.content,c.is_recipient, c.png_file_name))
        return lines

    def _preload_image_and_mask(self, file_name, content_infos):
        bin = Binarize()
        image = bin(Image.open(f"{os.path.join(self.root_dir, file_name)}.jpg").convert("L"))

        # create target mask
        mask = torch.zeros((1, image.size[1], image.size[0]))
        recipient_content_infos = [info for info in content_infos if info.is_recipient]

        image_transforms = transforms.Compose([Resize(size=[self.img_size, self.img_size])])
        mask_transforms = transforms.Compose([ApplyRecipientInfoToMask(recipient_content_infos), ToPILImage(),
                                              Resize(size=[self.img_size, self.img_size])])

        image_output = image_transforms(image)
        mask_output = mask_transforms(mask)
        return image_output, mask_output, image, image.size

    def _get_pos_weight(self):
        pos_weight = []
        print("computing pos weight")
        for i in tqdm(range(len(self.data))):
            item = self.__getitem__(i)
            pos_weight.append(torch.mean(item[MASK]))
        pos_weight = torch.mean(torch.Tensor(pos_weight))
        return torch.true_divide(1, pos_weight)

if __name__ == "__main__":
    # ds_line = ETDataset(root_dir="/cluster/mayr/nbb/vanilla/", multiple_books=["Band2", "Band3", "Band4"],
    #                     split="train", mode="both", augment=True)
    # dl_line = DataLoader(ds_line, batch_size=2, shuffle=False, collate_fn=collate_s2s_both, num_workers=8)
    # toPIL = transforms.ToPILImage()
    # for idx, b in enumerate(dl_line):
    #     image_unet, mask_unet, image = b[IMAGE], b[MASK], b["original_image"]
    #     content_infos, line_batch = b["content_infos"], b["line_batch"]
    #     # print(image_unet.shape, mask_unet.shape, len(image))
    #     # print(len(content_infos[0]), line_batch[0][LINE_IMAGE].shape)


    ds_line = ETDataset(root_dir="/cluster/mayr/nbb/vanilla/", multiple_books=["Band3","Band4"], split="train", augment=True)
    dl_line = DataLoader(ds_line, batch_size=16, shuffle=False, collate_fn=collate_s2s, num_workers=8)
    toPIL = transforms.ToPILImage()
    biggest_width = 0
    biggest_idx = 0
    # for idx, b in enumerate(dl_line):
    #     img = b[LINE_IMAGE]
    #     if img.shape[-1]>2500:
    #         biggest_width=img.shape[-1]
    #         biggest_idx = idx
    #         _,content,_,file_name = ds_line.data[biggest_idx]
    #         print(idx, biggest_width, img.shape, file_name, content)

    for b in dl_line:
        text = b[S2S_TEXT]
        print(text)
        key_masking = b[TGT_KEY_PADDING_MASK]
        print(key_masking)
        tgt_mask = b[TGT_MASK]
        print(tgt_mask)
        imgs = b[LINE_IMAGE]
        for i in imgs:
            img = toPIL(i)
            img.show()
        break

    # ds_page = ETDataset(root_dir="/cluster/mayr/nbb/vanilla/", multiple_books=["Band2"], mode="page")
    # print(len(ds_line), len(ds_page))

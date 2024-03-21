import os
import pickle
from typing import List

import torch
from matplotlib import pyplot as plt
from torchvision.transforms import transforms
from tqdm import tqdm

from application.helper.database_helper import connect, insert_line_infos
from application.helper.pageEntry import PageEntry
from model.LinRecClassifier.LinRecModel import LinRecModel
from model.unet.UnetModel import UnetModel
from utils.alphabet import Alphabet
from utils.metrics.TextConverter import convert_vector_to_text


def load_models_and_predict(params: dict, connection, plot=False):
    unet_path = params.get("unet_path")
    unet_arch = params.get("unet_arch")
    lin_rec_path = params.get("lin_rec_path")
    lin_rec_arch = params.get("lin_rec_arch")

    htr_model_path = params.get("htr_model_path")

    alphabet = Alphabet(dataset="NBB")

    unet = UnetModel.load_from_checkpoint(unet_path, architecture=unet_arch, strict=False, img_chan=3).cuda()
    lin_rec = LinRecModel.load_from_checkpoint(lin_rec_path, architecture=lin_rec_arch,
                                               htr_checkpoint_path=htr_model_path, alphabet=alphabet,
                                               max_content_len=102, strict=False).cuda()

    data_path = "tmp/new_dataset"
    entries: List[PageEntry] = _build_entries(data_path, "Band6") # TODO: Band -> args

    threshold = 0.5

    print(f"Used threshold: {threshold}")

    for entry in tqdm(entries):
        entry.compute(visualization=False)
        sigmoid = torch.nn.Sigmoid()
        unet_output = sigmoid(unet(entry.image.cuda()).squeeze(0))

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((entry.original_size[1], entry.original_size[0])),
            transforms.ToTensor(),
        ])

        original_size_output = transform(unet_output.cpu())
        sem_pred_placeholder = torch.zeros(original_size_output.shape)

        for line_entry in entry.page_line_entries:
            line_entry.compute(visualization=False)
            seg_pred = torch.mean(original_size_output[:, line_entry.vpos:(line_entry.vpos + line_entry.height),
                                  line_entry.hpos: (line_entry.hpos + line_entry.width)].float())

            if line_entry.image.shape[3] <= 22:
                print("found too short line image")
                sem_pred = 0
                content = "Sequence to short!"
            else:
                pred_content, sem_pred = lin_rec(line_entry.image.cuda())
                sem_pred = sigmoid(sem_pred)
                content = alphabet.logits_to_string(pred_content[0])

            sem_pred_placeholder[:, line_entry.vpos:(line_entry.vpos + line_entry.height),
            line_entry.hpos: (line_entry.hpos + line_entry.width)] = sem_pred
            insert_line_infos(connection, line_entry, content, threshold, sem_pred, seg_pred)

        if plot:
            img_to_plot = [transform(entry.image.squeeze()), original_size_output, sem_pred_placeholder]
            titles_to_plot = ["Original", "Unet sigmoids", "Semantic mask"]
            fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 6))
            for i, axi in enumerate(ax.flat):
                img = img_to_plot[i]
                axi.imshow(img.permute(1, 2, 0).detach(), alpha=0.25, cmap=plt.get_cmap("binary"))
                axi.set_title(titles_to_plot[i])
            plt.savefig(f"./tmp/plots/fig_{entry.image_file}")
            plt.close()


def _build_entries(data_path, book) -> List[PageEntry]:
    base_path = os.path.join(data_path, book)
    if not os.path.isdir(base_path):
        raise Exception(f"Path '{base_path}' not found.")
    xml_path = os.path.join(base_path, "alto")
    xml_files = os.listdir(xml_path)
    xml_files.sort()
    entries: List[PageEntry] = []
    for xml_file in xml_files:
        image_file_name = f"{xml_file.split('.')[0]}.jpg"
        pageEntry = PageEntry(data_path, xml_file, image_file_name, book)
        if len(pageEntry.page_line_entries) > 0:
            entries.append(pageEntry)
    return entries


if __name__ == '__main__':
    connection = connect()

    params = {
        "unet_path": "./../tmp/models/unet_unbalanced.ckpt",
        "unet_arch": "unet",
        "lin_rec_path": "./../tmp/models/text_class_cnn.ckpt",
        "lin_rec_arch": "cnn",
        "htr_model_path": "./../tmp/models/htr_ctc.ckpt",
    }

    load_models_and_predict(params, connection, plot=True)

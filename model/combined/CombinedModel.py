import os
from typing import Any, List

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from PIL import ImageDraw
from sklearn.metrics import f1_score
from torch import nn
from torchvision import transforms
from torchvision.transforms import Resize, ToTensor, ToPILImage

from application.helper.page_helper import draw_box
from dataset.combined.CombinedLineInfo import CombinedLineInfo
from model.LinRecClassifier.LinRecModel import LinRecModel
from model.unet.UnetModel import UnetModel
from utils.constants import IMAGE, CONTENT_INFOS, MASK, ORIGINAL_IMAGE_SIZE, ORIGINAL_IMAGE
from utils.metrics.LineDetectionMetricUtils import compute_line_detection_accuracy
from utils.metrics.UnetMetricUtils import compute_pixel_accuracy, compute_dice_coefficient


class CombinedModel(pl.LightningModule):

    def __init__(self, params: dict):
        super().__init__()
        # load pretrained models
        unet_path = params.get("unet_path")
        unet_arch = params.get("unet_arch")
        lin_rec_path = params.get("lin_rec_path")
        lin_rec_arch = params.get("lin_rec_arch")

        htr_model_path = params.get("htr_model_path")
        self.img_size = params.get("image_size", 256)
        self.plot_path = params.get("plot_path")

        self.unet = UnetModel.load_from_checkpoint(unet_path, architecture=unet_arch, strict=False)
        self.lin_rec = LinRecModel.load_from_checkpoint(lin_rec_path, architecture=lin_rec_arch,
                                                        htr_checkpoint_path=htr_model_path,
                                                        max_content_len=102, strict=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        raise Exception("Not implemented - Should not be called")

    def training_step(self, batch, batch_idx):
        raise Exception("Not implemented - Should not be called")

    def configure_optimizers(self):
        raise Exception("Not implemented - Should not be called")

    def test_step(self, test_batch, test_idx):
        y_mask = test_batch[MASK].cpu()
        unet_logits = self.unet(test_batch[IMAGE].unsqueeze(0))
        unet_sigmoids = self.sigmoid(unet_logits).cpu().squeeze(0)

        unet_preds = self.compute_unet_metrics(unet_sigmoids, test_batch)
        semantic_sigmoids = self.construct_semantic_map(test_batch)
        semantic_preds = self.compute_unet_metrics(semantic_sigmoids, test_batch, type="semantic")

        combined_sigmoids = unet_sigmoids * semantic_sigmoids
        threshold = 0.5 ** 2

        combined_preds = self.compute_unet_metrics(combined_sigmoids, test_batch, type="combined", threshold=threshold)

        #process image for plotting
        original_image = test_batch[ORIGINAL_IMAGE]
        draw = ImageDraw.Draw(test_batch[ORIGINAL_IMAGE])
        for content_info in test_batch[CONTENT_INFOS]:
            if content_info.is_recipient:
                draw_box(draw, content_info, (255, 0, 0))

        resize = Resize(size=[self.img_size, self.img_size])
        resized_img = resize(original_image)

        # Plot results
        img_to_plot = \
            [resized_img, y_mask, unet_sigmoids, unet_preds, semantic_sigmoids, semantic_preds, combined_sigmoids, combined_preds]
        titles_to_plot = \
            ["Input image", "Ground truth mask", "Unet sigmoids", "Unet prediction", "Semantic sigmoids", "Semantic prediction",
             "Combined sigmoids", "Combined prediction"]

        fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(6, 8))
        for i, axi in enumerate(ax.flat):

            axi.get_xaxis().set_visible(False)
            axi.get_yaxis().set_visible(False)
            img = img_to_plot[i]

            if titles_to_plot[i] == "Input image":
                axi.imshow(img)
            else:
                axi.imshow(img.permute(1, 2, 0), alpha=0.25, cmap=plt.get_cmap("binary"))

            axi.set_title(titles_to_plot[i])

        if not os.path.exists(f"./{self.plot_path}"):
            os.makedirs(f"./{self.plot_path}")

        plt.savefig(f"./{self.plot_path}/fig_{test_idx}")
        plt.close()

    def construct_semantic_map(self, test_batch):
        line_infos: List[CombinedLineInfo] = test_batch[CONTENT_INFOS]
        semantic_map = torch.zeros((1, test_batch[ORIGINAL_IMAGE_SIZE][1], test_batch[ORIGINAL_IMAGE_SIZE][0]))
        trans = transforms.Compose([ToPILImage(), Resize(size=[self.img_size, self.img_size]), ToTensor()])

        if len(line_infos) == 0:
            return trans(semantic_map)

        images = torch.stack([li.line_image for li in line_infos]).cuda()
        paddings = [li.unpadded_image_with for li in line_infos]
        targets = torch.tensor([li.is_recipient for li in line_infos])

        _, logits = self.lin_rec(images, paddings)
        sigmoid_output = self.sigmoid(logits)

        # compute f1 score - just for interest
        pred = (sigmoid_output > 0.5).cpu()
        f1 = f1_score(pred, targets)
        self.log(f"test_f1", f1)

        for idx, li in enumerate(line_infos):
            pred_value = sigmoid_output[idx]
            semantic_map[:, li.vpos:(li.vpos + li.height), li.hpos: (li.hpos + li.width)] = pred_value

        # transform back to shape that matches unet
        output = trans(semantic_map)
        return output

    def compute_unet_metrics(self, input, test_batch, type="unet", threshold=0.5):
        mask = test_batch[MASK].cpu()
        preds = input > threshold
        pixel_acc = compute_pixel_accuracy(preds, mask)
        dice_coefficient = compute_dice_coefficient(preds, mask)

        content_infos = test_batch[CONTENT_INFOS]

        if len(content_infos) > 0:
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(test_batch[ORIGINAL_IMAGE_SIZE]),
                transforms.ToTensor(),
            ])
            out = compute_line_detection_accuracy(transform(input), content_infos, seg_threshold=threshold)
            line_detection_accuracy = out["accuracy"]
        else:
            line_detection_accuracy = 1

        self.log(f"{type}_line_detection_acc", line_detection_accuracy)
        self.log(f"{type}_point_acc", pixel_acc)
        self.log(f"{type}_dice_coefficient", dice_coefficient)

        return preds


    def _forward_unimplemented(self, *input: Any) -> None:
        pass
